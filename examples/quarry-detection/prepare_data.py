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
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    OUTPUT_DIR = cfg['output_folder']
    SHPFILE = cfg['datasets']['shapefile']
    ZOOM_LEVEL = cfg['zoom_level']

    # Create an output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    written_files = []
    
    # Prepare the tiles

    ## Convert datasets shapefiles into geojson format
    logger.info('Convert labels shapefile into GeoJSON format (EPSG:4326)...')
    labels = gpd.read_file(SHPFILE)
    labels_4326 = labels.to_crs(epsg=4326)
    labels_4326['CATEGORY'] = "quarry"
    labels_4326['SUPERCATEGORY'] = "land usage"

    nb_labels = len(labels)
    logger.info('There is/are ' + str(nb_labels) + ' polygon(s) in ' + SHPFILE)

    label_filename = 'labels.geojson'
    label_filepath = os.path.join(OUTPUT_DIR, label_filename)
    labels_4326.to_file(label_filepath, driver='GeoJSON')
    written_files.append(label_filepath)  
    logger.success(f"{DONE_MSG} A file was written: {label_filepath}")

    logger.info('Creating tiles for the Area of Interest (AoI)...')   
    
    # Grid definition
    tms = morecantile.tms.get("WebMercatorQuad")    # epsg:3857

    # New gpd with only labels geometric info (minx, miny, maxx, maxy) 
    logger.info('- Get geometric boundaries of the label(s)')  
    label_boundaries_df = labels_4326.bounds

    # Iterate on geometric coordinates to defined tiles for a given label at a given zoom level
    # A gpd is created for each label and are then concatenate into a single gpd 
    logger.info('- Compute tiles for each label(s) geometry') 
    tiles_4326_all = [] 

    for label_boundary in label_boundaries_df.itertuples():
        coords = (label_boundary.minx, label_boundary.miny, label_boundary.maxx, label_boundary.maxy)   
        tiles_4326 = gpd.GeoDataFrame.from_features([tms.feature(x, projected=False) for x in tms.tiles(*coords, zooms=[ZOOM_LEVEL])])   
        tiles_4326.set_crs(epsg=4326, inplace=True)
        tiles_4326_all.append(tiles_4326)
    tiles_4326_aoi = gpd.GeoDataFrame(pd.concat(tiles_4326_all, ignore_index=True))

    # Remove unrelevant tiles and reorganised the data set:
    logger.info('- Remove duplicated tiles and tiles that are not intersecting labels') 

    # - Keep only tiles that are actually intersecting labels
    labels_4326.rename(columns={'FID': 'id_aoi'}, inplace=True)
    tiles_4326 = gpd.sjoin(tiles_4326_aoi, labels_4326, how='inner')

    # - Remove duplicated tiles
    if nb_labels > 1:
        tiles_4326.drop_duplicates('title', inplace=True)

    # - Remove useless columns, reset feature id and redefine it according to xyz format  
    logger.info('- Add tile IDs and reorganise data set')
    tiles_4326 = tiles_4326[['geometry', 'title']].copy()
    tiles_4326.reset_index(drop=True, inplace=True)

    # Add the ID column
    tiles_4326 = tiles_4326.apply(add_tile_id, axis=1)
    
    nb_tiles = len(tiles_4326)
    logger.info('There was/were ' + str(nb_tiles) + ' tiles(s) created')

    # Export tiles to GeoJSON
    logger.info('Export tiles to GeoJSON (EPSG:4326)...')  
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