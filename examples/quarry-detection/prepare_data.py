#!/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import yaml
import re

import geopandas as gpd
import morecantile
import pandas as pd

sys.path.insert(0, '../..')
from helpers.misc import format_logger
from helpers.constants import DONE_MSG

from loguru import logger
logger = format_logger(logger)


def add_tile_id(row):

    re_search = re.search('(x=(?P<x>\d*), y=(?P<y>\d*), z=(?P<z>\d*))', row.title)
    row['id'] = f"({re_search.group('x')}, {re_search.group('y')}, {re_search.group('z')})"
    
    return row


if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script prepares the Mineral Extraction Sites dataset to be processed by the object-detector scripts")
    parser.add_argument('config_file', type=str, help="Framework configuration file")
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    OUTPUT_DIR = cfg['output_folder']
    SHPFILE = cfg['datasets']['shapefile']
    FP_SHPFILE = cfg['datasets']['FP'] if 'FP' in cfg['datasets'].keys() else None
    ZOOM_LEVEL = cfg['zoom_level']
    EMPTY_TILES = cfg['empty_tiles']['enable'] if 'empty_tiles' in cfg.keys() else None
    if EMPTY_TILES:
        NB_TILES_FRAC = cfg['empty_tiles']['tiles_frac']
        AOI = cfg['empty_tiles']['aoi']

    # Create an output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    written_files = []
    
    # Prepare the tiles

    # Convert datasets shapefiles into geojson format
    logger.info("Convert labels shapefile into GeoJSON format (EPSG:4326)...")

    labels = gpd.read_file(SHPFILE)
    labels_4326 = labels.to_crs(epsg=4326)
    labels_4326['CATEGORY'] = 'quarry'
    labels_4326['SUPERCATEGORY'] = 'land usage"'

    nb_labels = len(labels)
    logger.info(f"There are {nb_labels} polygons in {SHPFILE}")

    filename = 'labels.geojson'
    filepath = os.path.join(OUTPUT_DIR, filename)
    labels_4326.to_file(filepath, driver='GeoJSON')
    written_files.append(filepath)  
    logger.success(f"{DONE_MSG} A file was written: {filepath}")

    # Add FP labels if it exists
    if FP_SHPFILE:
        FP_labels = gpd.read_file(FP_SHPFILE)
        FP_labels_4326 = FP_labels.to_crs(epsg=4326)

        nb_FP_labels = len(FP_labels)
        logger.info(f"There are {nb_FP_labels} polygons in {FP_SHPFILE}")

        filename = 'FP.geojson'
        filepath = os.path.join(OUTPUT_DIR, filename)
        FP_labels_4326.to_file(filepath, driver='GeoJSON')
        written_files.append(filepath)  
        logger.success(f"{DONE_MSG} A file was written: {filepath}")

        labels_4326 = pd.concat([labels_4326, FP_labels_4326])
    
    logger.info("Creating tiles for the Area of Interest (AoI)...")   
    
    # Grid definition
    tms = morecantile.tms.get('WebMercatorQuad')    # epsg:3857

    # Keep only label boundary geometry info (minx, miny, maxx, maxy) 
    logger.info("- Get the label geometric boundaries")  
    label_boundaries_df = labels_4326.bounds

    # Find coordinates of tiles intersecting labels
    logger.info("- Compute tiles for each label geometry") 
    tiles_4326_all = [] 

    for label_boundary in label_boundaries_df.itertuples():
        coords = (label_boundary.minx, label_boundary.miny, label_boundary.maxx, label_boundary.maxy)   
        tiles_4326 = gpd.GeoDataFrame.from_features([tms.feature(x, projected=False) for x in tms.tiles(*coords, zooms=[ZOOM_LEVEL])])   
        tiles_4326.set_crs(epsg=4326, inplace=True)
        tiles_4326_all.append(tiles_4326)
    tiles_4326_aoi = gpd.GeoDataFrame(pd.concat(tiles_4326_all, ignore_index=True))

    # Delete duplicated tiles and reorganised the data set:
    logger.info("- Remove duplicated tiles and tiles that are not intersecting labels") 

    # Keep tiles that are intersecting labels
    labels_4326.rename(columns={'FID': 'id_aoi'}, inplace=True)
    tiles_4326 = gpd.sjoin(tiles_4326_aoi, labels_4326, how='inner')

    if nb_labels > 1:
        tiles_4326.drop_duplicates('title', inplace=True)
    nb_tiles = len(tiles_4326)
    logger.info(f"- Number of tiles = {nb_tiles}")

    # Add tiles not intersecting with labels to the dataset 
    if EMPTY_TILES:
        nb_empty_tiles = int(NB_TILES_FRAC * nb_tiles)
        logger.info(f"- Add {int(NB_TILES_FRAC * 100)}% of empty tiles = {nb_empty_tiles} empty tiles")

        aoi = gpd.read_file(AOI)
        aoi_4326 = aoi.to_crs(epsg=4326)
        
        logger.info("- Get AoI geometric boundaries")  
        aoi_boundaries_df = aoi_4326.bounds
 
        logger.info("- Select random empty tiles") 
        aoi_tiles_4326_all = [] 
        for aoi_boundary in aoi_boundaries_df.itertuples():
            coords = (aoi_boundary.minx, aoi_boundary.miny, aoi_boundary.maxx, aoi_boundary.maxy)      
            aoi_tiles_4326 = gpd.GeoDataFrame.from_features([tms.feature(x, projected=False) for x in tms.tiles(*coords, zooms=[ZOOM_LEVEL])]) 
            aoi_tiles_4326.set_crs(epsg=4326, inplace=True)
            aoi_tiles_4326_all.append(aoi_tiles_4326)
        empty_tiles_4326_aoi = gpd.GeoDataFrame(pd.concat(aoi_tiles_4326_all, ignore_index=True))

        # Filter tiles intersecting labels 
        empty_tiles_4326_aoi = empty_tiles_4326_aoi[~empty_tiles_4326_aoi['title'].isin(tiles_4326['title'])].sample(n=nb_empty_tiles, random_state=1)
        tiles_4326 = pd.concat([tiles_4326, empty_tiles_4326_aoi])

    # Add tile IDs and reorganise data set
    tiles_4326 = tiles_4326[['geometry', 'title']].copy()
    tiles_4326.reset_index(drop=True, inplace=True)
    tiles_4326 = tiles_4326.apply(add_tile_id, axis=1)
    
    nb_tiles = len(tiles_4326)
    logger.info(f"There were {nb_tiles} tiles created")

    # Save tile shapefile
    logger.info("Export tiles to GeoJSON (EPSG:4326)...")  
    tile_filename = 'tiles.geojson'
    tile_filepath = os.path.join(OUTPUT_DIR, tile_filename)
    tiles_4326.to_file(tile_filepath, driver='GeoJSON')
    written_files.append(tile_filepath)  
    logger.success(f"{DONE_MSG} A file was written: {tile_filepath}")

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")
    
    sys.stderr.flush()