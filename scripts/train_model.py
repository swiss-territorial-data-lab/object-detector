#!/usr/bin/env python
# coding: utf-8


import argparse
import yaml, json
import os, sys
import cv2
import gdal
import time
import logging, logging.config

import numpy as np

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
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

from helpers.detectron2 import CocoTrainer, CocoPredictor


logging.config.fileConfig('logging.conf')
logger = logging.getLogger('root')


if __name__ == "__main__":
    
    tic = time.time()
    logger.info('Starting...')

    parser = argparse.ArgumentParser(description="This script trains a predictive models.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]
        
    # ---- parse config file    
    
    if 'model_zoo_checkpoint_url' in cfg['model_weights'].keys():
        MODEL_ZOO_CHECKPOINT_URL = cfg['model_weights']['model_zoo_checkpoint_url']
    else:
        MODEL_ZOO_CHECKPOINT_URL = None
    
    if 'pth_file' in cfg['model_weights'].keys():
        MODEL_PTH_FILE = cfg['model_weights']['pth_file']
    else:
        MODEL_PTH_FILE = None
    
    if MODEL_ZOO_CHECKPOINT_URL == None:
        logger.critical("A model zoo checkpoint URL (\"model_zoo_checkpoint_url\") must be provided")
        sys.exit(1)

    if 'num_channels' in cfg.keys():
        NUM_CHANNELS = cfg['num_channels']
        logger.info(f"Working with {NUM_CHANNELS} bands...")
        logger.info("A special DatasetMapper will be used to handle the additional bands.")
    else:
        NUM_CHANNELS = 3
        
    COCO_TRN_FILE = cfg['COCO_files']['trn']
    COCO_VAL_FILE = cfg['COCO_files']['val']
    COCO_TST_FILE = cfg['COCO_files']['tst']
        
    DETECTRON2_CFG_FILE = cfg['detectron2_config_file']
    

    WORKING_DIR = cfg['working_folder']
    SAMPLE_TAGGED_IMG_SUBDIR = cfg['sample_tagged_img_subfolder']
    LOG_SUBDIR = cfg['log_subfolder']
        
    
    os.chdir(WORKING_DIR)
    # let's make the output directories in case they don't exist
    for DIR in [SAMPLE_TAGGED_IMG_SUBDIR, LOG_SUBDIR]:
        if not os.path.exists(DIR):
            os.makedirs(DIR)

    

    written_files = []

    
    # ---- register datasets
    register_coco_instances("trn_dataset", {}, COCO_TRN_FILE, "")
    register_coco_instances("val_dataset", {}, COCO_VAL_FILE, "")
    register_coco_instances("tst_dataset", {}, COCO_TST_FILE, "")
    
    registered_datasets = ['trn_dataset', 'val_dataset', 'tst_dataset']

        
    registered_datasets_prefixes = [x.split('_')[0] for x in registered_datasets]


    for dataset in registered_datasets:
    
        for d in DatasetCatalog.get(dataset)[0:min(len(DatasetCatalog.get(dataset)), 4)]:
            output_filename = "tagged_" + d["file_name"].split('/')[-1]
            output_filename = output_filename.replace('tif', 'png')
            
            img = cv2.imread(d["file_name"])  
            
            visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(dataset), scale=1.0)
            
            vis = visualizer.draw_dataset_dict(d)
            cv2.imwrite(os.path.join(SAMPLE_TAGGED_IMG_SUBDIR, output_filename), vis.get_image()[:, :, ::-1])
            written_files.append( os.path.join(WORKING_DIR, os.path.join(SAMPLE_TAGGED_IMG_SUBDIR, output_filename)) )
            

    # ---- set up Detectron2's configuration

    # cf. https://detectron2.readthedocs.io/modules/config.html#config-references
    cfg = get_cfg()
    cfg.merge_from_file(DETECTRON2_CFG_FILE)
    cfg.OUTPUT_DIR = LOG_SUBDIR
    cfg.NUM_CHANNELS = NUM_CHANNELS
    
    # get the number of classes to detect
    classes={"file":[COCO_TRN_FILE,COCO_TST_FILE, COCO_VAL_FILE], "num_classes":[]}

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
    logger.info(f"Training with {num_classes} classe(s)")

    cfg.MODEL.ROI_HEADS.NUM_CLASSES=num_classes
    
    
    # ---- do training
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_ZOO_CHECKPOINT_URL)
    trainer = CocoTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    TRAINED_MODEL_PTH_FILE = os.path.join(LOG_SUBDIR, 'model_final.pth')
    written_files.append(os.path.join(WORKING_DIR, TRAINED_MODEL_PTH_FILE))

        
    # ---- evaluate model on the test dataset    
    #evaluator = COCOEvaluator("tst_dataset", cfg, False, output_dir='.')
    #val_loader = build_detection_test_loader(cfg, "tst_dataset")
    #inference_on_dataset(trainer.model, val_loader, evaluator)
   
    cfg.MODEL.WEIGHTS = TRAINED_MODEL_PTH_FILE
    logger.info("Make some sample predictions over the test dataset...")

    predictor = CocoPredictor(cfg)
     
    for d in DatasetCatalog.get("tst_dataset")[0:min(len(DatasetCatalog.get("tst_dataset")), 10)]:
        output_filename = "pred_" + d["file_name"].split('/')[-1]
        output_filename = output_filename.replace('tif', 'png')

        ds = gdal.Open(d["file_name"]) # We suppose the 1st bands are RGB 
        im_cwh = ds.ReadAsArray()
        im = np.transpose(im_cwh, (1, 2, 0))
        outputs = predictor(im)
        im_rgb=im[:,:,0:3]
        
        v = Visualizer(im_rgb, 
                       metadata=MetadataCatalog.get("tst_dataset"), 
                       scale=1.0, 
                       instance_mode=ColorMode.IMAGE_BW # remove the colors of unsegmented pixels
        )   
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(SAMPLE_TAGGED_IMG_SUBDIR, output_filename),
                    v.get_image()[:, :, ::-1]) # [:, :, ::-1] is for RGB -> BGR conversion, cf. https://stackoverflow.com/questions/14556545/why-opencv-using-bgr-colour-space-instead-of-rgb
        written_files.append( os.path.join(WORKING_DIR, os.path.join(SAMPLE_TAGGED_IMG_SUBDIR, output_filename)) )
    
    logger.info("...done.")

        
    # ------ wrap-up

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    print()

    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()


