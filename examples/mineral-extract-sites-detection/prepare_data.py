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
from helpers.functions_for_examples import format_all_tiles, prepare_labels
from helpers.misc import format_logger
from helpers.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


def add_tile_id(row):
    """Attribute tile id

    Args:
        row (DataFrame): row of a given df

    Returns:
        DataFrame: row with addition 'id' column
    """

    re_search = re.search('(x=(?P<x>\d*), y=(?P<y>\d*), z=(?P<z>\d*))', row.title)
    if 'year' in row.keys():
        row['id'] = f"({row.year}, {re_search.group('x')}, {re_search.group('y')}, {re_search.group('z')})"
    else:
        row['id'] = f"({re_search.group('x')}, {re_search.group('y')}, {re_search.group('z')})"
 
    return row


def aoi_tiling(gdf, tms='WebMercatorQuad'):
    """Tiling of an AoI

    Args:
        gdf (GeoDataFrame): gdf containing all the bbox boundary coordinates

    Returns:
        Geodataframe: gdf containing the tiles shape of the bbox of the AoI
    """

    # Grid definition
    tms = morecantile.tms.get(tms)    # epsg:3857

    tiles_all = [] 
    for boundary in gdf.itertuples():
        coords = (boundary.minx, boundary.miny, boundary.maxx, boundary.maxy)      
        tiles = gpd.GeoDataFrame.from_features([tms.feature(x, projected=False) for x in tms.tiles(*coords, zooms=[ZOOM_LEVEL])]) 
        tiles.set_crs(epsg=4326, inplace=True)
        tiles_all.append(tiles)
    tiles_all_gdf = gpd.GeoDataFrame(pd.concat(tiles_all, ignore_index=True))

    return tiles_all_gdf


def assert_year(gdf1, gdf2, ds, year=None):
    """Assert if the year of the dataset is well supported

    Args:
        gdf1 (GeoDataFrame): label geodataframe
        gdf2 (GeoDataFrame): other geodataframe to compare columns
        ds (string): dataset type (FP, empty tiles,...)
        year (string or numeric): attribution of year to empty tiles
    """

    labels_w_year = 'year' in gdf1.keys()
    oth_tiles_w_year = 'year' in gdf2.keys()

    if labels_w_year: 
        if not oth_tiles_w_year:
            if ds == 'empty_tiles' and year is None:
                logger.error("Year provided for labels, but not for empty tiles.")
                sys.exit(1)
            elif ds == 'FP':
                logger.error("Year provided for labels, but not for FP tiles.")
                sys.exit(1)
    else:
        if oth_tiles_w_year:
            logger.error(f"Year provided for the {ds.replace('_', ' ')} tiles, but not for the labels.")
            sys.exit(1)
        elif year != None: 
            logger.error(f"A year is provided as parameter, but no year available in the label attributes.")
            sys.exit(1)


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


def prepare_labels(shpfile, written_files, prefix=''):

    labels_gdf = gpd.read_file(shpfile)
    labels_gdf = misc.check_validity(labels_gdf, correct=True)
    if 'year' in labels_gdf.keys():
        labels_gdf['year'] = labels_gdf.year.astype(int)
        labels_4326_gdf = labels_gdf.to_crs(epsg=4326).drop_duplicates(subset=['geometry', 'year'])
    else:
        labels_4326_gdf = labels_gdf.to_crs(epsg=4326).drop_duplicates(subset=['geometry'])

    nb_labels = len(labels_4326_gdf)
    logger.info(f"There are {nb_labels} polygons in {os.path.basename(shpfile)}")

    labels_4326_gdf['CATEGORY'] = 'mineral extraction site'
    labels_4326_gdf['SUPERCATEGORY'] = 'land usage'

    labels_filepath = os.path.join(OUTPUT_DIR, f'{prefix}labels.geojson')
    labels_4326_gdf.to_file(labels_filepath, driver='GeoJSON')
    written_files.append(labels_filepath)  
    logger.success(f"{DONE_MSG} A file was written: {labels_filepath}")

    return labels_4326_gdf, written_files


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
    FP_SHPFILE = cfg['datasets']['fp_shapefile'] if 'fp_shapefile' in cfg['datasets'].keys() else None
    if 'empty_tiles' in cfg['datasets'].keys():
        EPT_TYPE = cfg['datasets']['empty_tiles']['type']
        EPT_SHPFILE = cfg['datasets']['empty_tiles']['shapefile']
        EPT_YEAR = cfg['datasets']['empty_tiles']['year'] if 'year' in cfg['datasets']['empty_tiles'].keys() else None
    else:
        EPT_SHPFILE = None
        EPT_TYPE = None
        EPT_YEAR = None
    CATEGORY = cfg['datasets']['category'] if 'category' in cfg['datasets'].keys() else None
    ZOOM_LEVEL = cfg['zoom_level']

    # Create an output directory in case it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    written_files = []
    
    gt_labels_4326_gdf = prepare_labels(SHPFILE, CATEGORY, supercategory=SUPERCATEGORY)

    # Get the number of tiles intersecting labels
    tiles_4326_gt_gdf = gpd.sjoin(tiles_4326_all_gdf, gt_labels_4326_gdf, how='inner', predicate='intersects')
    tiles_4326_gt_gdf.drop_duplicates(['id'], inplace=True)
    logger.info(f"- Number of tiles intersecting GT labels = {len(tiles_4326_gt_gdf)}")

    if FP_SHPFILE:
        tiles_4326_fp_gdf = gpd.sjoin(tiles_4326_all_gdf, fp_labels_4326_gdf, how='inner', predicate='intersects')
        tiles_4326_fp_gdf.drop_duplicates(['id'], inplace=True)
        logger.info(f"- Number of tiles intersecting FP labels = {len(tiles_4326_fp_gdf)}")

    # Save tile shapefile
    logger.info("Export tiles to GeoJSON (EPSG:4326)...")  
    tile_filepath = os.path.join(OUTPUT_DIR, 'tiles.geojson')
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