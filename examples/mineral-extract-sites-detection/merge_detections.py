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
import helpers.metrics as metrics
import helpers.misc as misc
from helpers.constants import DONE_MSG

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

    logger.info("Loading split AoI tiles as a GeoPandas DataFrame...")
    tiles_gdf = gpd.read_file('split_aoi_tiles.geojson')
    tiles_gdf = tiles_gdf.to_crs(2056)
    if 'year_tile' in tiles_gdf.keys(): 
        tiles_gdf['year_tile'] = tiles_gdf.year_tile.astype(int)
    logger.success(f"{DONE_MSG} {len(tiles_gdf)} features were found.")

    logger.info("Loading detections as a GeoPandas DataFrame...")

    detections_gdf = gpd.GeoDataFrame()

    for dataset, dets_file in DETECTION_FILES.items():
        detections_ds_gdf = gpd.read_file(dets_file)    
        detections_ds_gdf[f'dataset'] = dataset
        detections_gdf = pd.concat([detections_gdf, detections_ds_gdf], axis=0, ignore_index=True)
    detections_gdf = detections_gdf.to_crs(2056)
    detections_gdf['area'] = detections_gdf.area 
    detections_gdf['det_id'] = detections_gdf.index
    if 'year_det' in detections_gdf:
        if not detections_gdf['year_det'].all(): 
            detections_gdf['year_det'] = detections_gdf.year_det.astype(int)
    logger.success(f"{DONE_MSG} {len(detections_gdf)} features were found.")

    # get classe ids
    filepath = open(os.path.join('category_ids.json'))
    categories_json = json.load(filepath)
    filepath.close()
    categories_info_df = pd.DataFrame()
    for key in categories_json.keys():
        categories_tmp = {sub_key: [value] for sub_key, value in categories_json[key].items()}
        categories_info_df = pd.concat([categories_info_df, pd.DataFrame(categories_tmp)], ignore_index=True)
    categories_info_df.sort_values(by=['id'], inplace=True, ignore_index=True)
    categories_info_df.drop(['supercategory'], axis=1, inplace=True)
    categories_info_df.rename(columns={'name':'CATEGORY', 'id': 'label_class'},inplace=True)
    id_classes = range(len(categories_json))

    # Merge features
    logger.info(f"Merge adjacent polygons overlapping tiles with a buffer of {DISTANCE} m...")
    detections_year = gpd.GeoDataFrame()

    # Process detection by year
    for year in detections_gdf.year_det.unique():
        detections_gdf = detections_gdf.copy()
        detections_by_year_gdf = detections_gdf[detections_gdf['year_det']==year]

        # Merge overlapping polygons
        detections_merge_overlap_poly_gdf = misc.merge_polygons(detections_by_year_gdf, id_name='det_id')

        # Saves the id of polygons contained entirely within the tile (no merging with adjacent tiles), to avoid merging them if they are at a distance of less than thd  
        detections_buffer_gdf = detections_merge_overlap_poly_gdf.copy()
        detections_buffer_gdf['geometry'] = detections_buffer_gdf.geometry.buffer(1, join_style='mitre')
        detections_tiles_join_gdf = gpd.sjoin(tiles_gdf, detections_buffer_gdf, how='left', predicate='contains')
        remove_det_list = detections_tiles_join_gdf.det_id.unique().tolist()
        
        detections_within_tiles_gdf = gpd.GeoDataFrame()
        detections_within_tiles_gdf = detections_buffer_gdf[detections_buffer_gdf.det_id.isin(remove_det_list)].drop_duplicates(subset=['det_id'], ignore_index=True)

        # Merge adjacent polygons between tiles
        detections_overlap_tiles_gdf = gpd.GeoDataFrame()
        detections_overlap_tiles_gdf = detections_buffer_gdf[~detections_buffer_gdf.det_id.isin(remove_det_list)].drop_duplicates(subset=['det_id'], ignore_index=True)
        detections_overlap_tiles_gdf = misc.merge_polygons(detections_overlap_tiles_gdf)
    
        # Concat polygons contained within a tile and the merged ones
        detections_merge_gdf = pd.concat([detections_overlap_tiles_gdf, detections_within_tiles_gdf], axis=0, ignore_index=True)
        detections_merge_gdf['geometry'] = detections_merge_gdf.geometry.buffer(-1, join_style='mitre')

        # Merge adjacent polygons within the provided thd distance
        detections_merge_gdf['geometry'] = detections_merge_gdf.geometry.buffer(DISTANCE, join_style='mitre')
        detections_merge_gdf = misc.merge_polygons(detections_merge_gdf)
        detections_merge_gdf['geometry'] = detections_merge_gdf.geometry.buffer(-DISTANCE, join_style='mitre')
        detections_merge_gdf = detections_merge_gdf.explode(ignore_index=True)
        detections_merge_gdf['id'] = detections_merge_gdf.index


        # Spatially join merged detection with raw ones to retrieve relevant information (score, area,...)
        # Select the class of the largest polygon -> To Do: compute a parameter dependant of the area and the score
        # Score averaged over all the detection polygon (even if the class is different from the selected one)
        detections_join_gdf = gpd.sjoin(detections_merge_gdf, detections_by_year_gdf, how='inner', predicate='intersects')

        det_class_all = []
        det_score_all = []

        for id in detections_merge_gdf.id.unique():
            detections_by_year_gdf = detections_join_gdf.copy()
            detections_by_year_gdf = detections_by_year_gdf[(detections_by_year_gdf['id']==id)]
            detections_by_year_gdf = detections_by_year_gdf.rename(columns={'score_left': 'score'})
            det_score_all.append(detections_by_year_gdf['score'].mean())
            detections_by_year_gdf = detections_by_year_gdf.dissolve(by='det_class', aggfunc='sum', as_index=False)
            if len(detections_by_year_gdf) > 0:
                detections_by_year_gdf['det_class'] = detections_by_year_gdf.loc[detections_by_year_gdf['area']==detections_by_year_gdf['area'].max(), 
                                                                'det_class'].iloc[0]    
                det_class = detections_by_year_gdf['det_class'].drop_duplicates().tolist()
            else:
                det_class = [0]
            det_class_all.append(det_class[0])

        detections_merge_gdf['det_class'] = det_class_all
        detections_merge_gdf['score'] = det_score_all

        detections_merge_gdf = pd.merge(detections_merge_gdf, detections_join_gdf[
            ['id', 'dataset', 'year_det']], 
            on='id')
        detections_year = pd.concat([detections_year, detections_merge_gdf])

    detections_year['det_category'] = [
        categories_info_df.loc[categories_info_df.label_class==det_class+1, 'CATEGORY'].iloc[0] 
        if not np.isnan(det_class) else None
        for det_class in detections_year.det_class.to_numpy()
    ] 

    # Remove duplicate detection for a given year
    detections_merge_gdf = detections_year.drop_duplicates(subset=['geometry', 'year_det'])
    
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