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
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

sys.path.insert(0, '../..')
from helpers.functions_for_examples import assert_year, format_all_tiles
from helpers.misc import format_logger
from helpers.constants import DONE_MSG

from loguru import logger
logger = format_logger(logger)
 

def bbox(bounds):
    """Get a vector bounding box of a 2D shape

    Args:
        bounds (array): minx, miny, maxx, maxy of the bounding box

    Returns:
        geometry (Polygon): polygon geometry of the bounding box
    """

    minx = bounds[0]
    miny = bounds[1]
    maxx = bounds[2]
    maxy = bounds[3]

    return Polygon([[minx, miny],
                    [maxx, miny],
                    [maxx, maxy],
                    [minx, maxy]])


if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script prepares the ground truth dataset to be processed by the object-detector scripts")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    OUTPUT_DIR = cfg['output_folder']
    SHPFILE = cfg['datasets']['shapefile']
    CATEGORY = cfg['datasets']['category'] if 'category' in cfg['datasets'].keys() else False
    FP_SHPFILE = cfg['datasets']['fp_shapefile'] if 'fp_shapefile' in cfg['datasets'].keys() else None
    EPT_YEAR = cfg['datasets']['empty_tiles_year'] if 'empty_tiles_year' in cfg['datasets'].keys() else None
    if 'empty_tiles_aoi' in cfg['datasets'].keys() and 'empty_tiles_shp' in cfg['datasets'].keys():
        logger.error("Choose between supplying an AoI shapefile ('empty_tiles_aoi') in which empty tiles will be selected, or a shapefile with selected empty tiles ('empty_tiles_shp')")
        sys.exit(1)    
    elif 'empty_tiles_aoi' in cfg['datasets'].keys():
        EPT_SHPFILE = cfg['datasets']['empty_tiles_aoi']
        EPT = 'aoi'
    elif 'empty_tiles_shp' in cfg['datasets'].keys():
        EPT_SHPFILE = cfg['datasets']['empty_tiles_shp'] 
        EPT = 'shp'
    else:
        EPT_SHPFILE = None
        EPT = None
    CATEGORY = cfg['datasets']['category'] if 'category' in cfg['datasets'].keys() else False
    ZOOM_LEVEL = cfg['zoom_level']

    # Create an output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    written_files = []
    
    # Prepare the tiles

    ## Convert datasets shapefiles into geojson format
    logger.info('Convert labels shapefile into GeoJSON format (EPSG:4326)...')
    labels_gdf = gpd.read_file(SHPFILE)
    if 'year' in labels_gdf.keys():
        labels_gdf['year'] = labels_gdf.year.astype(int)
        labels_4326_gdf = labels_gdf.to_crs(epsg=4326).drop_duplicates(subset=['geometry', 'year'])
    else:
        labels_4326_gdf = labels_gdf.to_crs(epsg=4326).drop_duplicates(subset=['geometry'])
    nb_labels = len(labels_gdf)
    logger.info(f'There are {nb_labels} polygons in {SHPFILE}')

    labels_4326_gdf['CATEGORY'] = 'quarry'
    labels_4326_gdf['SUPERCATEGORY'] = 'land usage'
    
    gt_labels_4326_gdf = labels_4326_gdf.copy()

    label_filename = 'labels.geojson'
    label_filepath = os.path.join(OUTPUT_DIR, label_filename)
    labels_4326_gdf.to_file(label_filepath, driver='GeoJSON')
    written_files.append(label_filepath)  
    logger.success(f"{DONE_MSG} A file was written: {label_filepath}")

    tiles_4326_all_gdf, tmp_written_files = format_all_tiles(
        FP_SHPFILE, os.path.join(OUTPUT_DIR, 'FP.geojson'), EPT_SHPFILE, ept_data_type=EPT, labels_gdf=gt_labels_4326_gdf,
        category='quarry', supercategory='land usage', zoom_level=ZOOM_LEVEL
    )

    # Save tile shapefile
    logger.info("Export tiles to GeoJSON (EPSG:4326)...")  
    tile_filename = 'tiles.geojson'
    tile_filepath = os.path.join(OUTPUT_DIR, tile_filename)
    tiles_4326_all_gdf.to_file(tile_filepath, driver='GeoJSON')
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