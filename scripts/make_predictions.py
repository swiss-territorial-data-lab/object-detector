#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.insert(0, '.')

import argparse
import yaml
import numpy as np
import os, json, cv2, random
import time
import datetime
import logging, logging.config
import pickle
import gzip

import torch, torchvision
assert torch.__version__.startswith("1.7")

from tqdm import tqdm

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.modeling import build_model
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode

from helpers.detectron2 import LossEvalHook, CocoTrainer
from helpers.detectron2 import dt2predictions_to_list


logging.config.fileConfig('logging.conf')
logger = logging.getLogger('root')


if __name__ == "__main__":
    
    tic = time.time()
    logger.info('Starting...')

    parser = argparse.ArgumentParser(description="This script assesses the quality of predictions with respect to ground-truth/other labels.")
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
        
    DO_TRAIN = cfg['do_train']
    
    if DO_TRAIN and MODEL_ZOO_CHECKPOINT_URL == None:
        logger.critical("A model zoo checkpoint URL (\"model_zoo_checkpoint_url\") must be provided in case \"do_train: True\"")
        sys.exit(1)
        
    if not DO_TRAIN and MODEL_PTH_FILE == None:
        logger.critical("A model pickle file (\"pickle_file\") must be provided in case \"do_train: False\"")
        sys.exit(1)
        
    COCO_TRN_FILE = cfg['COCO_files']['trn']
    COCO_VAL_FILE = cfg['COCO_files']['val']
    COCO_TST_FILE = cfg['COCO_files']['tst']
    
    # optional parameter
    if 'oth' in cfg['COCO_files']:
        COCO_OTH_FILE = cfg['COCO_files']['oth']
    else:
        COTO_OTH_FILE = None
        
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
    
    if COCO_OTH_FILE is not None:
        register_coco_instances("oth_dataset", {}, COCO_OTH_FILE, "")
        registered_datasets.append('oth_dataset')


    for dataset in registered_datasets:
    
        for d in random.sample(DatasetCatalog.get(dataset), min(len(DatasetCatalog.get(dataset)), 4)):
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
    
    
    if DO_TRAIN:
        # ---- do training
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_ZOO_CHECKPOINT_URL)
        trainer = CocoTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
        written_files.append(os.path.join(WORKING_DIR, LOG_SUBDIR, 'model_final.pth'))

    else:
        # ---- load pre-trained model
        cfg.MODEL.WEIGHTS = MODEL_PTH_FILE
        
    # ---- evaluate model on the test dataset    
    #evaluator = COCOEvaluator("tst_dataset", cfg, False, output_dir='.')
    #val_loader = build_detection_test_loader(cfg, "tst_dataset")
    #inference_on_dataset(trainer.model, val_loader, evaluator)
   
    logger.info("Make some sample predictions over the test dataset...")

    predictor = DefaultPredictor(cfg)
     
    for d in random.sample(DatasetCatalog.get("tst_dataset"), min(len(DatasetCatalog.get("tst_dataset")), 10)):
        output_filename = "pred_" + d["file_name"].split('/')[-1]
        output_filename = output_filename.replace('tif', 'png')
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        v = Visualizer(im[:, :, ::-1], # [:, :, ::-1] is for RGB -> BGR conversion, cf. https://stackoverflow.com/questions/14556545/why-opencv-using-bgr-colour-space-instead-of-rgb
                       metadata=MetadataCatalog.get("tst_dataset"), 
                       scale=1.0, 
                       instance_mode=ColorMode.IMAGE_BW # remove the colors of unsegmented pixels
        )   
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite(os.path.join(SAMPLE_TAGGED_IMG_SUBDIR, output_filename), v.get_image()[:, :, ::-1])
        written_files.append( os.path.join(WORKING_DIR, os.path.join(SAMPLE_TAGGED_IMG_SUBDIR, output_filename)) )
    
    logger.info("...done.")
    
    
    # ---- make predictions
    
    
    threshold = 0.05
    threshold_str = str( round(threshold, 2) ).replace('.', 'dot')

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold   # set the testing threshold for this model
    


    for dataset_prefix in ["trn", "val", "tst", "oth"]:

        predictions = {}
        
        logger.info(f"Making predictions over the entire {dataset_prefix} dataset...")
        
        prediction_filename = f'{dataset_prefix}_predictions_at_{threshold_str}_threshold.pkl'
        #prediction_file = gzip.open(prediction_filename, 'wt', encoding='utf8')
  
        for d in tqdm(DatasetCatalog.get( f"{dataset_prefix}_dataset" ) ):
            
            im = cv2.imread(d["file_name"])
            try:
                outputs = predictor(im)
            except Exception as e:
                print(f"Exception: {e}, file: {d['file_name']}")
                sys.exit(1)
                
            predictions[d['file_name']] = dt2predictions_to_list(outputs)
#             this_img_predictions = {
#                 d['file_name']: dt2predictions_to_list(outputs)
#             }
            #print(tmp)
            #prediction_file.write(json.dumps(this_img_predictions)+"\n")
            
        
        #prediction_file.close()   
        with open(prediction_filename, 'wb') as fp:
            pickle.dump(predictions, fp)
            
        written_files.append(os.path.join(WORKING_DIR, prediction_filename))
        
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


