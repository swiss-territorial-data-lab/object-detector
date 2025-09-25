#!/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import yaml

import geopandas as gpd

sys.path.insert(0, '../..')
from helpers.functions_for_examples import format_all_tiles, prepare_labels
from helpers.misc import format_logger
from helpers.constants import DONE_MSG

from loguru import logger
logger = format_logger(logger)

if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script prepares the dataset for the example about mineral extraction sites.")
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
    SUPERCATEGORY = 'land usage'
    ZOOM_LEVEL = cfg['zoom_level']

    # Create an output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    written_files = []
    
    gt_labels_4326_gdf = prepare_labels(SHPFILE, CATEGORY, supercategory=SUPERCATEGORY)

    label_filepath = os.path.join(OUTPUT_DIR, 'labels.geojson')
    gt_labels_4326_gdf.to_file(label_filepath, driver='GeoJSON')
    written_files.append(label_filepath)  
    logger.success(f"{DONE_MSG} A file was written: {label_filepath}")

    tiles_4326_all_gdf, tmp_written_files = format_all_tiles(
        FP_SHPFILE, os.path.join(OUTPUT_DIR, 'FP.geojson'), EPT_SHPFILE, ept_data_type=EPT, ept_year=EPT_YEAR, labels_4326_gdf=gt_labels_4326_gdf,
        category='quarry', supercategory=SUPERCATEGORY, zoom_level=ZOOM_LEVEL
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