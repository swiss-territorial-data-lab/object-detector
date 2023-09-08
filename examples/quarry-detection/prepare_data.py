#!/bin/python
# -*- coding: utf-8 -*-

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import time
import argparse
import yaml
import re
from tqdm import tqdm
from loguru import logger

import geopandas as gpd
import morecantile
import pandas as pd


# the following allows us to import modules from within this file's parent folder
sys.path.insert(0, '.')

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO")


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
    LABELS_SHPFILE = cfg['datasets']['labels_shapefile']
    ZOOM_LEVEL = cfg['zoom_level']

    # Create an output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Prepare the tiles
    written_files = []

    ## Convert datasets shapefiles into geojson format
    logger.info('Convert labels shapefile into GeoJSON format (EPSG:4326)...')
    labels = gpd.read_file(LABELS_SHPFILE)
    labels_4326 = labels.to_crs(epsg=4326)

    nb_labels = len(labels)
    logger.info('There is/are ' + str(nb_labels) + ' polygon(s) in ' + LABELS_SHPFILE)

    label_filename = 'labels.geojson'
    label_filepath = os.path.join(OUTPUT_DIR, label_filename)
    labels_4326.to_file(label_filepath, driver='GeoJSON')
    written_files.append(label_filepath)  
    logger.info(f"...done. A file was written: {label_filepath}")

    logger.info('Creating tiles for the Area of Interest (AOI)...')   
    
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
        tiles_4326 = gpd.GeoDataFrame.from_features([tms.feature(x, projected=False) for x in tqdm(tms.tiles(*coords, zooms=[ZOOM_LEVEL]))])   
        tiles_4326.set_crs(epsg=4326, inplace=True)
        tiles_4326_all.append(tiles_4326)
    tiles_4326_aoi = gpd.GeoDataFrame(pd.concat(tiles_4326_all, ignore_index=True))

    # Remove unrelevant tiles and reorganized the data set:
    logger.info('- Remove duplicated tiles and tiles that are not intersecting labels') 

    # - Keep only tiles that are actually intersecting labels
    labels_4326.rename(columns={'FID': 'id_aoi'}, inplace=True)
    tiles_4326 = gpd.sjoin(tiles_4326_aoi, labels_4326, how='inner')

    # - Remove duplicated tiles
    if nb_labels > 1:
        tiles_4326.drop_duplicates('title', inplace=True)

    # - Remove useless columns, reinitilize feature id and redifine it according to xyz format  
    logger.info('- Format feature id and reorganise data set') 
    tiles_4326.drop(tiles_4326.columns.difference(['geometry','id','title']), axis=1, inplace=True) 
    tiles_4326.reset_index(drop=True, inplace=True)

    # Format the xyz parameters and fill in the attributes columns
    xyz = []
    for idx in tiles_4326.index:
        xyz.append([re.sub('\D','',coor) for coor in tiles_4326.loc[idx,'title'].split(',')])
    tiles_4326['id'] = [f'({x}, {y}, {z})' for x, y, z in xyz]
    tiles_4326 = tiles_4326[['geometry', 'title', 'id']]

    nb_tiles = len(tiles_4326)
    logger.info('There was/were ' + str(nb_tiles) + ' tiles(s) created')

    # Convert datasets shapefiles into geojson format
    logger.info('Convert tiles shapefile into GeoJSON format (EPSG:4326)...')  
    tile_filename = 'tiles.geojson'
    tile_filepath = os.path.join(OUTPUT_DIR, tile_filename)
    tiles_4326.to_file(tile_filepath, driver='GeoJSON')
    written_files.append(tile_filepath)  
    logger.info(f"...done. A file was written: {tile_filepath}")

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")
    
    sys.stderr.flush()