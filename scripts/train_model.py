#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import cv2
import time
import yaml

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# the following lines allow us to import modules from within this file's parent folder
from inspect import getsourcefile
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from helpers.detectron2 import AugmentedCocoTrainer, CocoTrainer
from helpers.misc import format_logger, get_number_of_classes
from helpers.constants import DONE_MSG

from loguru import logger
logger = format_logger(logger)


def main(cfg_file_path):

    tic = time.time()
    logger.info('Starting...')

    logger.info(f"Using {cfg_file_path} as config file.")
    
    with open(cfg_file_path) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]
        
    # ---- parse config file    

    DEBUG = cfg['debug_mode'] if 'debug_mode' in cfg.keys() else False
    RESUME_TRAINING = cfg['resume_training'] if 'resume_training' in cfg.keys() else False
    DATA_AUGMENTATION = cfg['data_augmentation'] if 'data_augmentation' in cfg.keys() else False
    
    MODEL_WEIGHTS = cfg['model_weights']
    
    COCO_FILES_DICT = cfg['COCO_files']
    COCO_TRN_FILE = COCO_FILES_DICT['trn']
    COCO_VAL_FILE = COCO_FILES_DICT['val']
    COCO_TST_FILE = COCO_FILES_DICT['tst']
        
    DETECTRON2_CFG_FILE = cfg['detectron2_config_file']
    

    WORKING_DIR = cfg['working_directory']
    SAMPLE_TAGGED_IMG_SUBDIR = cfg['sample_tagged_img_subfolder']
    LOG_SUBDIR = cfg['log_subfolder']
        
    
    os.chdir(WORKING_DIR)
    written_files = []

    
    # ---- register datasets
    register_coco_instances("trn_dataset", {}, COCO_TRN_FILE, "")
    register_coco_instances("val_dataset", {}, COCO_VAL_FILE, "")
    register_coco_instances("tst_dataset", {}, COCO_TST_FILE, "")
    
    registered_datasets = ['trn_dataset', 'val_dataset', 'tst_dataset']

    if not RESUME_TRAINING:
        # Erase folder if exists and make them anew
        for dir in [SAMPLE_TAGGED_IMG_SUBDIR, LOG_SUBDIR]:
            if os.path.exists(dir):
                os.system(f"rm -r {dir}")
            os.makedirs(dir)

        for dataset in registered_datasets:
        
            for d in DatasetCatalog.get(dataset)[0:min(len(DatasetCatalog.get(dataset)), 4)]:
                output_filename = "tagged_" + d["file_name"].split('/')[-1]
                output_filename = output_filename.replace('tif', 'png')
                
                img = cv2.imread(d["file_name"])  
                
                visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get(dataset), scale=1.0)
                
                vis = visualizer.draw_dataset_dict(d)
                cv2.imwrite(os.path.join(SAMPLE_TAGGED_IMG_SUBDIR, output_filename), vis.get_image()[:, :, ::-1])
                written_files.append(os.path.join(WORKING_DIR, SAMPLE_TAGGED_IMG_SUBDIR, output_filename))
            

    # ---- set up Detectron2's configuration

    # cf. https://detectron2.readthedocs.io/modules/config.html#config-references
    cfg = get_cfg()
    cfg.merge_from_file(DETECTRON2_CFG_FILE)
    cfg.OUTPUT_DIR = LOG_SUBDIR
    
    num_classes = get_number_of_classes(COCO_FILES_DICT)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = num_classes

    if DATA_AUGMENTATION and (cfg.INPUT.CROP.ENABLE or cfg.INPUT.RANDOM_FLIP):
        logger.critical("Augmentation must be either in detectron2 config or in dataloader, mix currently not supported.")
        sys.exit(1)

    if DEBUG:
        logger.warning('Setting a configuration for DEBUG only.')
        cfg.IMS_PER_BATCH = 2
        cfg.SOLVER.STEPS = (100, 200, 250, 300, 350, 375, 400, 425, 450, 460, 470, 480, 490)
        cfg.SOLVER.MAX_ITER = 500
    
    # ---- do training
    TRAINED_MODEL_PTH_FILE = os.path.join(LOG_SUBDIR, 'model_final.pth')
    if RESUME_TRAINING:
        if 'pth_file' in MODEL_WEIGHTS.keys():
            PICK_UP_MODEL = MODEL_WEIGHTS['pth_file']
            logger.info(f"Resuming training from {PICK_UP_MODEL}")
        else:
            logger.warning(F'No model path to resume from. Using {TRAINED_MODEL_PTH_FILE}.')
            PICK_UP_MODEL = TRAINED_MODEL_PTH_FILE
        cfg.MODEL.WEIGHTS = PICK_UP_MODEL
        trainer = CocoTrainer(cfg)
        trainer.resume_or_load(resume=True)
    else:
        PASSED_ZOO_MODEL = 'model_zoo_checkpoint_url' in MODEL_WEIGHTS.keys()
        if PASSED_ZOO_MODEL and cfg.MODEL.WEIGHTS:
            logger.critical(f"A model zoo checkpoint URL is provided from parameters and detectron2 config. Remove weights from {cfg_file_path} or {DETECTRON2_CFG_FILE}.")
            sys.exit(1)
        elif PASSED_ZOO_MODEL:
            MODEL_ZOO_CHECKPOINT_URL = MODEL_WEIGHTS['model_zoo_checkpoint_url']
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_ZOO_CHECKPOINT_URL)
        elif not (PASSED_ZOO_MODEL or cfg.MODEL.WEIGHTS):
            logger.critical("A model zoo checkpoint URL  must be provided in parameters (\"model_zoo_checkpoint_url\") or detectron2 config (\"model.weights\").")
            sys.exit(1)
    
        logger.info(f"Fine-tuning from {cfg.MODEL.WEIGHTS}")
        trainer = AugmentedCocoTrainer(cfg) if DATA_AUGMENTATION else CocoTrainer(cfg)
        trainer.resume_or_load(resume=False)
    trainer.train()
    written_files.append(os.path.join(WORKING_DIR, TRAINED_MODEL_PTH_FILE))
   
    logger.info("Make some sample detections over the test dataset...")
    cfg.MODEL.WEIGHTS = TRAINED_MODEL_PTH_FILE

    predictor = DefaultPredictor(cfg)
     
    for d in DatasetCatalog.get("tst_dataset")[0:min(len(DatasetCatalog.get("tst_dataset")), 10)]:
        output_filename = "det_tst_" + d["file_name"].split('/')[-1]
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
        written_files.append(os.path.join(WORKING_DIR, SAMPLE_TAGGED_IMG_SUBDIR, output_filename))
    
    logger.success(DONE_MSG)

        
    # ------ wrap-up

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    print()

    toc = time.time()
    logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This script trains an object detection model.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    main(args.config_file)
