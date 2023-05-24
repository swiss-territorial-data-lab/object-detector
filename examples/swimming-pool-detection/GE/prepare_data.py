#!/bin/python
# -*- coding: utf-8 -*-

import logging
import logging.config
import time
import argparse
import yaml
import os, sys, inspect
import requests
import geopandas as gpd
import pandas as pd
import json

from tqdm import tqdm

# the following allows us to import modules from within this file's parent folder
sys.path.insert(0, '.')

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('root')

if __name__ == "__main__":

    tic = time.time()
    logger.info('Starting...')

    parser = argparse.ArgumentParser(description="This script prepares datasets for the Geneva's Swimming Pools detection task.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    OUTPUT_DIR = cfg['output_folder']
    LAKES_SHPFILE = cfg['datasets']['lakes_shapefile']
    PARCELS_SHPFILE = cfg['datasets']['parcels_shapefile']
    SWIMMINGPOOLS_SHPFILE = cfg['datasets']['swimmingpools_shapefile']
    OK_TILE_IDS_CSV = cfg['datasets']['OK_z18_tile_IDs_csv']
    ZOOM_LEVEL = 18 # this is hard-coded 'cause we only know "OK tile IDs" for this zoom level

    # let's make the output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    written_files = []

    # ------ (Down)loading datasets

    dataset_dict = {}

    for dataset in ['lakes', 'parcels', 'swimmingpools']:

        shpfile_name = eval(f'{dataset.upper()}_SHPFILE').split('/')[-1]
        shpfile_path = os.path.join(OUTPUT_DIR, shpfile_name)

        if eval(f'{dataset.upper()}_SHPFILE').startswith('http'):

            logger.info(f"Downloading the {dataset} dataset...")
            r = requests.get(eval(f'{dataset.upper()}_SHPFILE'), timeout=30)  
            with open(shpfile_path, 'wb') as f:
                f.write(r.content)

            written_files.append(shpfile_path)
            logger.info(f"...done. A file was written: {shpfile_path}")

        logger.info(f"Loading the {dataset} dataset as a GeoPandas DataFrame...")
        dataset_dict[dataset] = gpd.read_file(f'zip://{shpfile_path}')
        logger.info(f"...done. {len(dataset_dict[dataset])} records were found.")


    # ------ Computing the Area of Interest (AOI) = cadastral parcels - Léman lake

    logger.info("Computing the Area of Interest (AOI)...")

    # N.B.: 
    # it's faster to first compute Slippy Map Tiles (cf. https://developers.planet.com/tutorials/slippy-maps-101/), 
    # then suppress the tiles which "fall" within the Léman lake.
    # We rely on supermercado, mercantile and fiona for the computation of Slippy Map Tiles.

    # lake_gdf
    l_gdf = dataset_dict['lakes'].copy()
    # parcels_gdf
    p_gdf = dataset_dict['parcels'].copy()

    PARCELS_TILES_GEOJSON_FILE = os.path.join(OUTPUT_DIR, f"parcels_z{ZOOM_LEVEL}_tiles.geojson")

    if not os.path.isfile(PARCELS_TILES_GEOJSON_FILE):
        logger.info("Exporting the parcels dataset to a GeoJSON file...")
        PARCELS_GEOJSON_FILE = os.path.join(OUTPUT_DIR, 'parcels.geojson')
        p_gdf[['geometry']].to_crs(epsg=4326).to_file(PARCELS_GEOJSON_FILE, driver='GeoJSON')
        written_files.append(PARCELS_GEOJSON_FILE)
        logger.info(f"...done. The {PARCELS_GEOJSON_FILE} was written.")

        print()
        logger.warning(f"You should now open a Linux shell and run the following command from the working directory (./{OUTPUT_DIR}), then run this script again:")
        logger.warning(f"cat parcels.geojson | supermercado burn {ZOOM_LEVEL} | mercantile shapes | fio collect > parcels_z{ZOOM_LEVEL}_tiles.geojson")
        sys.exit(0) 
        
    else:
        parcels_tiles_gdf = gpd.read_file(PARCELS_TILES_GEOJSON_FILE)
        
    # parcels tiles falling within the lake
    tiles_to_remove_gdf = gpd.sjoin(parcels_tiles_gdf.to_crs(epsg=l_gdf.crs.to_epsg()), l_gdf[l_gdf.NOM == 'Léman'], how='right', predicate='within')

    aoi_tiles_gdf = parcels_tiles_gdf[ ~parcels_tiles_gdf.index.isin(tiles_to_remove_gdf.index_left) ]
    assert ( len(aoi_tiles_gdf.drop_duplicates(subset='id')) == len(aoi_tiles_gdf) ) # make sure there are no duplicates

    AOI_TILES_GEOJSON_FILE = os.path.join(OUTPUT_DIR, f'aoi_z{ZOOM_LEVEL}_tiles.geojson')
    aoi_tiles_gdf.to_crs(epsg=4326).to_file(AOI_TILES_GEOJSON_FILE, driver='GeoJSON')
    written_files.append(AOI_TILES_GEOJSON_FILE)


    # ------- Splitting labels into two groups: ground truth (those intersecting the "OK" tileset) and other

    # OK tiles: the subset of tiles containing neither false positives nor false negatives
    OK_ids = pd.read_csv(OK_TILE_IDS_CSV)
    OK_tiles_gdf = aoi_tiles_gdf[aoi_tiles_gdf.id.isin(OK_ids.id)]

    OK_TILES_GEOJSON = os.path.join(OUTPUT_DIR, 'OK_tiles.geojson')
    OK_tiles_gdf.to_crs(epsg=4326).to_file(OK_TILES_GEOJSON, driver='GeoJSON')
    written_files.append(OK_TILES_GEOJSON)

    labels_gdf = dataset_dict['swimmingpools'].copy()
    labels_gdf = labels_gdf.to_crs(epsg=4326)

    # Ground Truth Labels = Labels intersecting OK tiles
    try:
        assert( labels_gdf.crs == OK_tiles_gdf.crs ), f"CRS mismatching: labels' CRS = {labels_gdf.crs} != OK_tiles' CRS = {OK_tiles_gdf.crs}"
    except Exception as e:
        logger.critical(e)
        sys.exit(1)
    
    GT_labels_gdf = gpd.sjoin(labels_gdf, OK_tiles_gdf, how='inner', predicate='intersects')
    # the following two lines make sure that no swimming pool is counted more than once in case it intersects multiple tiles
    GT_labels_gdf = GT_labels_gdf[labels_gdf.columns]
    GT_labels_gdf.drop_duplicates(inplace=True)
    OTH_labels_gdf = labels_gdf[ ~labels_gdf.index.isin(GT_labels_gdf.index)]

    try:
        assert( len(labels_gdf) == len(GT_labels_gdf) + len(OTH_labels_gdf) ),\
            f"Something went wrong when splitting labels into Ground Truth Labels and Other Labels. Total no. of labels = {len(labels_gdf)}; no. of Ground Truth Labels = {len(GT_labels_gdf)}; no. of Other Labels = {len(OTH_labels_gdf)}"
    except Exception as e:
        logger.critical(e)
        sys.exit(1)

    GT_LABELS_GEOJSON = os.path.join(OUTPUT_DIR, 'ground_truth_labels.geojson')
    OTH_LABELS_GEOJSON = os.path.join(OUTPUT_DIR, 'other_labels.geojson')

    GT_labels_gdf.to_crs(epsg=4326).to_file(GT_LABELS_GEOJSON, driver='GeoJSON')
    written_files.append(GT_LABELS_GEOJSON)
    OTH_labels_gdf.to_crs(epsg=4326).to_file(OTH_LABELS_GEOJSON, driver='GeoJSON')
    written_files.append(OTH_LABELS_GEOJSON)

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()