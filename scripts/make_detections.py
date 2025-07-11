#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import os
import sys
import argparse
import cv2
import json
import time
import yaml
from tqdm import tqdm

import geopandas as gpd
import pandas as pd

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

from helpers.detectron2 import detectron2dets_to_features
from helpers.misc import image_metadata_to_affine_transform, format_logger, get_number_of_classes, add_geohash, remove_overlap_poly
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
    if 'pth_file' in cfg['model_weights'].keys():
        MODEL_PTH_FILE = cfg['model_weights']['pth_file']
    else:
        logger.critical("A model pickle file (\"pth_file\") must be provided")
        sys.exit(1)
         
        
    COCO_FILES_DICT = cfg['COCO_files']
    DETECTRON2_CFG_FILE = cfg['detectron2_config_file']
    
    WORKING_DIR = cfg['working_directory']
    OUTPUT_DIR = cfg['output_folder'] if 'output_folder' in cfg.keys() else '.'
    SAMPLE_TAGGED_IMG_SUBDIR = cfg['sample_tagged_img_subfolder'] if 'sample_tagged_img_subfolder' in cfg.keys() else False
    LOG_SUBDIR = cfg['log_subfolder'] if 'log_subfolder' in cfg.keys() else False

    SCORE_LOWER_THR = cfg['score_lower_threshold'] 

    IMG_METADATA_FILE = cfg['image_metadata_json']
    RDP_SIMPLIFICATION_ENABLED = cfg['rdp_simplification']['enabled']
    RDP_SIMPLIFICATION_EPSILON = cfg['rdp_simplification']['epsilon']
    REMOVE_OVERLAP = cfg['remove_det_overlap'] if 'remove_det_overlap' in cfg.keys() else False

    os.chdir(WORKING_DIR)
    # let's make the output directories in case they don't exist
    for directory in [OUTPUT_DIR, LOG_SUBDIR, SAMPLE_TAGGED_IMG_SUBDIR]:
        if directory:
            os.makedirs(directory, exist_ok=True)

    written_files = []

    # ------ Loading image metadata
    with open(IMG_METADATA_FILE, 'r') as fp:
        tmp = json.load(fp)

    # let's extract filenames (w/o path)
    img_metadata_dict = {os.path.split(k)[-1]: v for (k, v) in tmp.items()}

    # ---- register datasets
    for dataset_key, coco_file in COCO_FILES_DICT.items():
        register_coco_instances(dataset_key, {}, coco_file, "")

    # ---- set up Detectron2's configuration

    # cf. https://detectron2.readthedocs.io/modules/config.html#config-references
    cfg = get_cfg()
    cfg.merge_from_file(DETECTRON2_CFG_FILE)
    cfg.OUTPUT_DIR = LOG_SUBDIR

    cfg.MODEL.WEIGHTS = MODEL_PTH_FILE
    logger.info(f'Using model {MODEL_PTH_FILE}.')

    # get the number of classes
    num_classes = get_number_of_classes(COCO_FILES_DICT)

   # set the number of classes to detect 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes    

    # set the testing threshold for this model
    threshold = SCORE_LOWER_THR
    threshold_str = str( round(threshold, 2) ).replace('.', 'dot')
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold   

    predictor = DefaultPredictor(cfg)
    
    # ---- make detections   
    for dataset in COCO_FILES_DICT.keys():

        all_feats = []
        crs = None
        
        logger.info(f"Making detections over the entire {dataset} dataset...")
        
        detections_filename = os.path.join(OUTPUT_DIR, f'{dataset}_detections_at_{threshold_str}_threshold.gpkg')
    
        for d in tqdm(DatasetCatalog.get(dataset)):
            
            im = cv2.imread(d["file_name"])
            try:
                outputs = predictor(im)
            except Exception as e:
                print(f"Exception: {e}, file: {d['file_name']}")
                sys.exit(1)
                  
            kk = d["file_name"].split('/')[-1]
            im_md = img_metadata_dict[kk]

            _crs = f"EPSG:{im_md['extent']['spatialReference']['latestWkid']}"

            # let's make sure all the images share the same CRS
            if crs is not None: # iterations other than the 1st
                assert crs == _crs, "Mismatching CRS"
            crs = _crs

            transform = image_metadata_to_affine_transform(im_md)
            if 'year' in im_md.keys():
                year = im_md['year']
                this_image_feats = detectron2dets_to_features(outputs, d['file_name'], transform, RDP_SIMPLIFICATION_ENABLED, RDP_SIMPLIFICATION_EPSILON, year=year)
            else:
                this_image_feats = detectron2dets_to_features(outputs, d['file_name'], transform, RDP_SIMPLIFICATION_ENABLED, RDP_SIMPLIFICATION_EPSILON)

            all_feats += this_image_feats

        gdf = gpd.GeoDataFrame.from_features(all_feats, crs=crs)
        gdf['dataset'] = dataset

        # Filter detection to avoid overlapping detection polygons due to multi-class detection 
        if REMOVE_OVERLAP:
            id_to_keep = []
            gdf = add_geohash(gdf)
            if 'year_det' in gdf.keys():
                for year in gdf.year_det.unique():
                    gdf_temp = gdf.copy()
                    gdf_temp = gdf_temp[gdf_temp['year_det']==year] 
                    gdf_temp['geom'] = gdf_temp.geometry
                    ids = remove_overlap_poly(gdf_temp, id_to_keep)
                    id_to_keep.append(ids)
            else:
                id_to_keep = remove_overlap_poly(gdf_temp, id_to_keep)  
            # Keep only polygons with the highest detection score
            gdf = gdf[gdf.geohash.isin(id_to_keep)]
        gdf.to_file(detections_filename, driver='GPKG')
        written_files.append(os.path.join(WORKING_DIR, detections_filename))
            
        logger.success(DONE_MSG)

        if SAMPLE_TAGGED_IMG_SUBDIR:
            logger.info("Let's tag some sample images...")
            for d in DatasetCatalog.get(dataset)[0:min(len(DatasetCatalog.get(dataset)), 10)]:
                output_filename = f'{dataset}_det_{d["file_name"].split("/")[-1]}'
                output_filename = output_filename.replace('tif', 'png')
                im = cv2.imread(d["file_name"])
                outputs = predictor(im)
                v = Visualizer(im[:, :, ::-1], # RGB -> BGR conversion for open-cv
                    metadata=MetadataCatalog.get(dataset), 
                    scale=1.0, 
                    instance_mode=ColorMode.IMAGE_BW # remove the colors of unsegmented pixels
                )   
                v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                filepath = os.path.join(SAMPLE_TAGGED_IMG_SUBDIR, output_filename)
                cv2.imwrite(filepath, v.get_image()[:, :, ::-1])
                written_files.append(os.path.join(WORKING_DIR, filepath))
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
    
    parser = argparse.ArgumentParser(description="This script makes detections, using a previously trained model.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    main(args.config_file) 