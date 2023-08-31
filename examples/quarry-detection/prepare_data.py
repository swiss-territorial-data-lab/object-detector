#!/bin/python
# -*- coding: utf-8 -*-

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import argparse
import yaml
import os, sys
import geopandas as gpd
import pandas as pd
import morecantile
import re

from tqdm import tqdm
from loguru import logger

# the following allows us to import modules from within this file's parent folder
sys.path.insert(0, '.')

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO")

if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script prepares dataset to process the quarries detection project (STDL.proj-dqry)")
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

    feature = 'labels.geojson'
    feature_path = os.path.join(OUTPUT_DIR, feature)
    labels_4326.to_file(feature_path, driver='GeoJSON')
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")

    logger.info('Creating tiles for the Area of Interest (AOI)...')   
    
    # Grid definition
    tms = morecantile.tms.get("WebMercatorQuad")    # epsg:3857

    # New gpd with only labels geometric info (minx, miny, maxx, maxy) 
    logger.info('- Get geometric boundaries of the label(s)')  
    boundary = labels_4326.bounds

    # Iterate on geometric coordinates to defined tiles for a given label at a given zoom level
    # A gpd is created for each label and are then concatenate into a single gpd 
    logger.info('- Compute tiles for each label(s) geometry') 
    tiles_3857_all = [] 
    for row in range(len(boundary)):
        coords = (boundary.iloc[row,0],boundary.iloc[row,1],boundary.iloc[row,2],boundary.iloc[row,3])   
        tiles_3857 = gpd.GeoDataFrame.from_features([tms.feature(x, projected=True) for x in tqdm(tms.tiles(*coords, zooms=[ZOOM_LEVEL]))])   
        tiles_3857.set_crs(epsg=3857, inplace=True)
        tiles_3857_all.append(tiles_3857)
    tiles_3857_aoi = gpd.GeoDataFrame(pd.concat(tiles_3857_all, ignore_index=True) )

    # Remove unrelevant tiles and reorganized the data set:
    logger.info('- Remove duplicated tiles and tiles that are not intersecting labels') 

    # - Keep only tiles that are intersecting the label   
    labels_3857=labels_4326.to_crs(epsg=3857)
    labels_3857.rename(columns={'FID': 'id_aoi'},inplace=True)
    tiles_aoi=gpd.sjoin(tiles_3857_aoi, labels_3857, how='inner')

    # - Remove duplicated tiles
    if nb_labels > 1:
        tiles_aoi.drop_duplicates('title', inplace=True)

    # - Remove useless columns, reinitilize feature id and redifined it according to xyz format  
    logger.info('- Format feature id and reorganise data set') 
    tiles_aoi.drop(tiles_aoi.columns.difference(['geometry','id','title']), axis=1, inplace=True) 
    tiles_aoi.reset_index(drop=True, inplace=True)

    # Format the xyz parameters and filled in the attributes columns
    xyz = []
    for idx in tiles_aoi.index:
        xyz.append([re.sub('\D','',coor) for coor in tiles_aoi.loc[idx,'title'].split(',')])
    tiles_aoi['id'] = [f'({x}, {y}, {z})' for x, y, z in xyz]
    tiles_aoi = tiles_aoi[['geometry', 'title', 'id']]

    nb_tiles = len(tiles_aoi)
    logger.info('There was/were ' + str(nb_tiles) + ' tiles(s) created')

    # Convert datasets shapefiles into geojson format
    logger.info('Convert tiles shapefile into GeoJSON format (EPSG:4326)...')  
    feature = 'tiles.geojson'
    feature_path = os.path.join(OUTPUT_DIR, feature)
    tiles_4326=tiles_aoi.to_crs(epsg=4326)
    tiles_4326.to_file(feature_path, driver='GeoJSON')
    written_files.append(feature_path)  
    logger.info(f"...done. A file was written: {feature_path}")

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()