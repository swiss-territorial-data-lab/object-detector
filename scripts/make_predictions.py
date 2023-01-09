#!/usr/bin/env python
# coding: utf-8


import argparse
import yaml, json
import os, sys
import gdal
import time
import logging, logging.config
import pickle

import numpy as np
import torch

from cv2 import imwrite
from tqdm import tqdm

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode

# the following lines allow us to import modules from within this file's parent folder
from inspect import getsourcefile
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from helpers.detectron2 import CocoPredictor
from helpers.detectron2 import dt2predictions_to_list


logging.config.fileConfig('logging.conf')
logger = logging.getLogger('root')


if __name__ == "__main__":
    
    tic = time.time()
    logger.info('Starting...')

    parser = argparse.ArgumentParser(description="This script makes predictions, using a previously trained model.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]
        
    # ---- parse config file  
    if 'pth_file' in cfg['model_weights'].keys():
        MODEL_PTH_FILE = cfg['model_weights']['pth_file']
    else:
        logger.critical("A model pickle file (\"pth_file\") must be provided")
        sys.exit(1)
         
        
    COCO_FILES_DICT = cfg['COCO_files']
    DETECTRON2_CFG_FILE = cfg['detectron2_config_file']
    
    WORKING_DIR = cfg['working_folder']
    SAMPLE_TAGGED_IMG_SUBDIR = cfg['sample_tagged_img_subfolder']
    LOG_SUBDIR = cfg['log_subfolder']

    if 'num_channels' in cfg.keys():
        NUM_CHANNELS = cfg['num_channels']
        logger.info("A special DatasetMapper will be used to handle the additional bands.")
    else:
        NUM_CHANNELS = 3
    
    os.chdir(WORKING_DIR)
    # let's make the output directories in case they don't exist
    for DIR in [SAMPLE_TAGGED_IMG_SUBDIR, LOG_SUBDIR]:
        if not os.path.exists(DIR):
            os.makedirs(DIR)

    written_files = []

    
    # ---- register datasets
    for dataset_key, coco_file in COCO_FILES_DICT.items():
        register_coco_instances(dataset_key, {}, coco_file, "")

    # ---- set up Detectron2's configuration

    # cf. https://detectron2.readthedocs.io/modules/config.html#config-references
    cfg = get_cfg()
    cfg.merge_from_file(DETECTRON2_CFG_FILE)
    cfg.OUTPUT_DIR = LOG_SUBDIR
    cfg.NUM_CHANNELS = NUM_CHANNELS

     # get the number of classes to make prediction for
    classes={"file":[COCO_FILES_DICT['trn'], COCO_FILES_DICT['tst'], COCO_FILES_DICT['val']], "num_classes":[]}

    for filepath in classes["file"]:
        file = open(filepath)
        coco_json = json.load(file)
        classes["num_classes"].append(len(coco_json["categories"]))
        file.close()

    # test if it is the same number of classes in all datasets
    try:
        assert classes["num_classes"][0]==classes["num_classes"][1] and classes["num_classes"][0]==classes["num_classes"][2]
    except AssertionError:
        logger.info(f"The number of classes is not equal in the training ({classes['num_classes'][0]}), testing ({classes['num_classes'][1]}) and validation ({classes['num_classes'][2]}) datasets. The program will not continue.")
        sys.exit(1)

   # set the number of classes to detect 
    num_classes=classes["num_classes"][0]
    logger.info(f"Making predictions for {num_classes} classe(s)")

    cfg.MODEL.ROI_HEADS.NUM_CLASSES=num_classes
    
    cfg.MODEL.WEIGHTS = MODEL_PTH_FILE
    predictor = CocoPredictor(cfg)
    
    # ---- make predictions
    threshold = 0.05
    threshold_str = str( round(threshold, 2) ).replace('.', 'dot')

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold   # set the testing threshold for this model
    
    for dataset in COCO_FILES_DICT.keys():

        predictions = {}
        
        logger.info(f"Making predictions over the entire {dataset} dataset...")
        
        prediction_filename = f'{dataset}_predictions_at_{threshold_str}_threshold.pkl'
    
        for d in tqdm(DatasetCatalog.get(dataset)):
            
            ds = gdal.Open(d["file_name"])
            im_cwh = ds.ReadAsArray()
            im = np.transpose(im_cwh, (1, 2, 0))

            try:
                outputs = predictor(im)
            except Exception as e:
                print(f"Exception: {e}, file: {d['file_name']}")
                sys.exit(1)
                
            predictions[d['file_name']] = dt2predictions_to_list(outputs)
            
        with open(prediction_filename, 'wb') as fp:
            pickle.dump(predictions, fp)
            
        written_files.append(os.path.join(WORKING_DIR, prediction_filename))
        
        logger.info('...done.')
        
        logger.info("Let's tag some sample images...")
        for d in DatasetCatalog.get(dataset)[0:min(len(DatasetCatalog.get(dataset)), 10)]:
            output_filename = f'{dataset}_pred_{d["file_name"].split("/")[-1]}'
            output_filename = output_filename.replace('tif', 'png')

            ds = gdal.Open(d["file_name"])
            im_cwh = ds.ReadAsArray()
            im = np.transpose(im_cwh, (1, 2, 0))
            outputs = predictor(im)
            im_rgb=im[:,:,0:3]

            v = Visualizer(im_rgb, 
                           metadata=MetadataCatalog.get(dataset), 
                           scale=1.0, 
                           instance_mode=ColorMode.IMAGE_BW # remove the colors of unsegmented pixels
            )   
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            imwrite(os.path.join(SAMPLE_TAGGED_IMG_SUBDIR, output_filename),
                    v.get_image()[:, :, ::-1]) # [:, :, ::-1] is for RGB -> BGR conversion, cf. https://stackoverflow.com/questions/14556545/why-opencv-using-bgr-colour-space-instead-of-rgb
            written_files.append( os.path.join(WORKING_DIR, os.path.join(SAMPLE_TAGGED_IMG_SUBDIR, output_filename)) )
        logger.info('...done.')

        
    # ------ wrap-up

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    print()

    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()


