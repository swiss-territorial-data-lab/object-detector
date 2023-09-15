#!/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import yaml
import geopandas as gpd
import pandas as pd

from loguru import logger


if __name__ == "__main__":


    tic = time.time()
    logger.info('Starting...')

    parser = argparse.ArgumentParser(description="This script prepares datasets for the Neuchâtel's Swimming Pools detection task.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    OUTPUT_DIR = cfg['output_folder']
    # sectors
    GROUND_TRUTH_SECTORS_SHPFILE = cfg['datasets']['ground_truth_sectors_shapefile']
    OTHER_SECTORS_SHPFILE = cfg['datasets']['other_sectors_shapefile']
    # swimming pools
    GROUND_TRUTH_SWIMMING_POOLS_SHPFILE = cfg['datasets']['ground_truth_swimming_pools_shapefile']
    OTHER_SWIMMING_POOLS_SHPFILE = cfg['datasets']['other_swimming_pools_shapefile']
    ZOOM_LEVEL = cfg['zoom_level']
    
    # let's make the output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    written_files = []


    # ------ Loading datasets

    dataset_dict = {}

    for dataset in [
        'ground_truth_sectors', 
        'other_sectors',
        'ground_truth_swimming_pools',
        'other_swimming_pools']:

        shpfile = eval(f'{dataset.upper()}_SHPFILE')#.split('/')[-1]

        logger.info(f"Loading the {dataset} dataset as a GeoPandas DataFrame...")
        dataset_dict[dataset] = gpd.read_file(f'{shpfile}')
        logger.success(f"...done. {len(dataset_dict[dataset])} records were found.")


    # ------ Computing the Area of Interest (AoI)

    aoi_gdf = pd.concat([
        dataset_dict['ground_truth_sectors'],
        dataset_dict['other_sectors']
    ])

    aoi_gdf.drop_duplicates(inplace=True)

    AOI_GEOJSON = os.path.join(OUTPUT_DIR, "aoi.geojson")
    try:
        aoi_gdf.to_crs(epsg=4326).to_file(AOI_GEOJSON, driver='GeoJSON', encoding='utf-8')
        written_files.append(AOI_GEOJSON)
    except Exception as e:
        logger.error(f"Could not write to file {AOI_GEOJSON}. Exception: {e}")    

    AOI_TILES_GEOJSON = os.path.join(OUTPUT_DIR, f"aoi_z{ZOOM_LEVEL}_tiles.geojson")
    
    if not os.path.isfile(AOI_TILES_GEOJSON):
        print()
        logger.warning(f"You should now open a Linux shell and run the following command from the working directory (./{OUTPUT_DIR}), then run this script again:")
        logger.warning(f"cat aoi.geojson | supermercado burn {ZOOM_LEVEL} | mercantile shapes | fio collect > aoi_z{ZOOM_LEVEL}_tiles.geojson")
        sys.exit(0) 
        
    else:
        logger.info("Loading AoI tiles as a GeoPandas DataFrame...")
        aoi_tiles_gdf = gpd.read_file(AOI_TILES_GEOJSON)
        logger.success(f"...done. {len(aoi_tiles_gdf)} records were found.")


    assert ( len(aoi_tiles_gdf.drop_duplicates(subset='id')) == len(aoi_tiles_gdf) ) # make sure there are no duplicates


    # ------ Exporting labels to GeoJSON

    GT_LABELS_GEOJSON = os.path.join(OUTPUT_DIR, 'ground_truth_labels.geojson')
    OTH_LABELS_GEOJSON = os.path.join(OUTPUT_DIR, 'other_labels.geojson')

    dataset_dict['ground_truth_swimming_pools'].to_crs(epsg=4326).to_file(GT_LABELS_GEOJSON, driver='GeoJSON')
    written_files.append(GT_LABELS_GEOJSON)
    dataset_dict['other_swimming_pools'].to_crs(epsg=4326).to_file(OTH_LABELS_GEOJSON, driver='GeoJSON')
    written_files.append(OTH_LABELS_GEOJSON)

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    toc = time.time()
    logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()



