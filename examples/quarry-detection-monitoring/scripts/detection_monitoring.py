#!/bin/python
# -*- coding: utf-8 -*-

#  Proj quarry detection and time machine
#
#      Nils Hamel - nils.hamel@alumni.epfl.ch
#      Huriel Reichel
#      Clemence Herny 
#      Shanci Li
#      Alessandro Cerioni 
#      Copyright (c) 2020 Republic and Canton of Geneva
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
# 
# 
################################################################
#  Script used to plot and visualize quarry data trough times 
#  Inputs are defined in config-dm.yaml


import logging
import logging.config
import time
import argparse
import yaml
import os, sys, inspect
import geopandas as gpd
import pandas as pd

# the following allows us to import modules from within this file's parent folder
sys.path.insert(0, '.')

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('root')

if __name__ == "__main__":

    # Start chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="Time machine script (STDL.proj-dqry)")
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    OUTPUT_DIR = cfg['output_folder']
    YEARS = sorted(cfg['years'])

    # Create an output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    written_files = []

    # Concatenate all the dataframe obtained for a given year to a single dataframe
    print()
    logger.info(f"Concaneting different years dataframes:")
    for YEAR in YEARS:      
        if YEAR == YEARS[0]: 
            DETECTION = cfg['datasets']['detection']
            DETECTION = DETECTION.replace('{year}', str(YEAR)) 
            gdf = gpd.read_file(DETECTION)
            gdf['year'] = YEAR 
            logger.info(f" Year: {YEAR} - {len(gdf)} detections")
        else:
            DETECTION = cfg['datasets']['detection']
            DETECTION = DETECTION.replace('{year}', str(YEAR)) 
            gdfn = gpd.read_file(DETECTION)
            gdfn['year'] = YEAR 
            logger.info(f" Year: {YEAR} - {len(gdfn)} detections")
            gdf = pd.concat([gdf, gdfn])

    # Create a dataframe with intersecting polygons merged together
    print()
    logger.info(f"Merging overlapping polygons")
    gdf_all = gdf.geometry.unary_union
    gdf_all = gpd.GeoDataFrame(geometry=[gdf_all], crs='EPSG:2056')  
    logger.info(f"Attribute unique object id")
    gdf_all = gdf_all.explode(index_parts = True).reset_index(drop=True)
    labels = gdf_all.index

    # Spatially compare the global dataframe with all prediction by year and the merged dataframe. Allow to attribute a unique ID to each detection
    print()
    logger.info(f"Compare single polygon dataframes detection to merge polygons dataframe")
    intersection = gpd.sjoin(gdf, gdf_all, how='inner')

    # Reorganized dataframe columns and save files 
    print()
    logger.info(f"Save files")
    intersection.rename(columns={'index_right': 'id_object'}, inplace=True)
    gdf_final = intersection[['id_object', 'id_feature', 'year', 'score', 'area', 'centroid_x', 'centroid_y', 'geometry']]
    feature_path = os.path.join(OUTPUT_DIR, 'quarry_times.geojson')
    gdf_final.to_file(feature_path, driver='GeoJSON') 
    written_files.append(feature_path) 
    feature_path = os.path.join(OUTPUT_DIR, 'quarry_times.csv')
    gdf_final.to_csv(feature_path, index=False)
    written_files.append(feature_path) 

    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()