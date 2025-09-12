#!/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import yaml

import geopandas as gpd
import numpy as np
import pandas as pd
import json

sys.path.insert(0, '../..')
import helpers.misc as misc
from helpers.functions_for_examples import get_categories, merge_adjacent_detections, read_dets_and_aoi

from loguru import logger
logger = misc.format_logger(logger)


if __name__ == "__main__":

    # Chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script assess the post-processed detections")
    parser.add_argument('config_file', type=str, help='input geojson path')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_directory']
    LABELS = misc.none_if_undefined(cfg, 'labels')
    DETECTION_FILES = cfg['detections']
    DISTANCE = cfg['distance']
    SCORE_THD = cfg['score_threshold']
    IOU_THD = misc.none_if_undefined(cfg, 'iou_threshold')
    AREA_THD = misc.none_if_undefined(cfg, 'area_threshold')

    os.chdir(WORKING_DIR)
    logger.info(f'Working directory set to {WORKING_DIR}')

    written_files = [] 

    tiles_gdf, detections_gdf = read_dets_and_aoi(DETECTION_FILES)

    # get class ids
    categories_info_df, _ = get_categories('category_ids.json')

    # Merge features
    logger.info(f"Merge adjacent polygons overlapping tiles with a buffer of {DISTANCE} m...")
    detections_all_years_gdf = gpd.GeoDataFrame()

    # Process detection by year
    for year in detections_gdf.year_det.unique():
        complete_merge_dets_gdf, detections_within_tiles_gdf = merge_adjacent_detections(detections_gdf, tiles_gdf, year, DISTANCE)
        detections_all_years_gdf = pd.concat([detections_all_years_gdf, complete_merge_dets_gdf, detections_within_tiles_gdf], ignore_index=True)

    detections_all_years_gdf['det_category'] = [
        categories_info_df.loc[categories_info_df.label_class==det_class+1, 'category'].iloc[0] 
        if not np.isnan(det_class) else None
        for det_class in detections_all_years_gdf.det_class.to_numpy()
    ] 

    # Remove duplicate detection for a given year
    detections_merge_gdf = detections_all_years_gdf.drop_duplicates(subset=['geometry', 'year_det'])
    
    nb_detections = len(detections_merge_gdf)
    logger.info(f"... {nb_detections} detections remaining after union of the shapes.")
    
    # Filter dataframe by score value
    detections_merge_gdf = detections_merge_gdf[detections_merge_gdf.score > SCORE_THD]
    nb_score = len(detections_merge_gdf)
    logger.info(f"{nb_detections - nb_score} detections were removed by score filtering (score threshold = {SCORE_THD})")

    # Save processed results
    feature = os.path.join(f'merged_detections_at_{SCORE_THD}_threshold.gpkg'.replace('0.', '0dot'))
    detections_merge_gdf = detections_merge_gdf.to_crs(2056)
    detections_merge_gdf[['geometry', 'score', 'det_class', 'det_category', 'year_det']]\
        .to_file(feature, driver='GPKG', index=False)
    written_files.append(feature)     

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    toc = time.time()
    logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()