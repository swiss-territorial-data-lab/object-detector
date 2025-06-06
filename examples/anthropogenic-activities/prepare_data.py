#!/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import yaml

import geopandas as gpd
import rasterio as rio

import helpers.functions_for_examples as ffe
import helpers.misc as misc

from loguru import logger
logger = misc.format_logger(logger)


if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script prepares the dataset for the example about anthropogenic activities.")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    DATASET_KEYS = cfg['datasets'].keys()
    CATEGORY = cfg['datasets']['category_field'] if 'category_field' in DATASET_KEYS else False
    OUTPUT_DIR = cfg['output_folder']
    SHPFILE = cfg['datasets']['shapefile']
    FP_SHPFILE = cfg['datasets']['fp_shapefile'] if 'fp_shapefile' in DATASET_KEYS else None
    EPT_YEAR = cfg['datasets']['empty_tiles_year'] if 'empty_tiles_year' in DATASET_KEYS else None
    DEM = cfg['dem'] if 'dem' in cfg.keys() else None

    EPT_SHPFILE = None
    EPT = None
    if 'empty_tiles_aoi' in DATASET_KEYS and 'empty_tiles_shp' in DATASET_KEYS:
        logger.error("Choose between supplying an AoI shapefile ('empty_tiles_aoi') in which empty tiles will be selected, or a shapefile with selected empty tiles ('empty_tiles_shp')")
        sys.exit(1)    
    elif 'empty_tiles_aoi' in DATASET_KEYS:
        EPT_SHPFILE = cfg['datasets']['empty_tiles_aoi']
        EPT = 'aoi'
    elif 'empty_tiles_shp' in DATASET_KEYS:
        EPT_SHPFILE = cfg['datasets']['empty_tiles_shp'] 
        EPT = 'shp'
    
    ZOOM_LEVEL = cfg['zoom_level']
    SUPERCATEGORY = 'anthropogenic soils'

    # Create an output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    written_files = []
    
    gt_labels_4326_gdf = ffe.preapre_labels(SHPFILE, CATEGORY, supercategory=SUPERCATEGORY)
    
    label_filepath = os.path.join(OUTPUT_DIR, 'labels.geojson')
    gt_labels_4326_gdf.to_file(label_filepath, driver='GeoJSON')
    written_files.append(label_filepath)  
    logger.success(f"Done! A file was written: {label_filepath}")

    tiles_4326_all_gdf, tmp_written_files = ffe.format_all_tiles(
        FP_SHPFILE, os.path.join(OUTPUT_DIR, 'FP.geojson'), EPT_SHPFILE, ept_data_type=EPT, labels_gdf=gt_labels_4326_gdf,
        category=CATEGORY, supercategory=SUPERCATEGORY, zoom_level=ZOOM_LEVEL
    )

    # Save tile shapefile
    tile_filepath = os.path.join(OUTPUT_DIR, 'tiles.geojson')
    if tiles_4326_all_gdf.empty:
        logger.warning('No tile generated for the designated area.')
        tile_filepath = os.path.join(OUTPUT_DIR, 'area_without_tiles.gpkg')
        gt_labels_4326_gdf.to_file(tile_filepath)
        written_files.append(tile_filepath)  
    else:
        logger.info("Export tiles to GeoJSON (EPSG:4326)...") 
        tiles_4326_all_gdf.to_file(tile_filepath, driver='GeoJSON')
        written_files.append(tile_filepath)  
        logger.success(f"Done! A file was written: {tile_filepath}")

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")
    
    sys.stderr.flush()