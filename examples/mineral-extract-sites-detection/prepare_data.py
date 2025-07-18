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
import helpers.misc as misc
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
    
    # Prepare the tiles

    ## Convert datasets shapefiles into geojson format
    logger.info('Convert the label shapefiles into GeoJSON format (EPSG:4326)...')
    labels_4326_gdf, written_files = prepare_labels(SHPFILE, written_files)
    gt_labels_4326_gdf = labels_4326_gdf[['geometry', 'CATEGORY', 'SUPERCATEGORY']].copy()

    # Add FP labels if it exists
    if FP_SHPFILE:
        logger.info('Convert the FP label shapefiles into GeoJSON format (EPSG:4326)...')
        fp_labels_4326_gdf, written_files = prepare_labels(FP_SHPFILE, written_files, prefix='FP_')
        labels_4326_gdf = pd.concat([labels_4326_gdf, fp_labels_4326_gdf], ignore_index=True)

    # Tiling of the AoI
    logger.info("- Get the label boundaries")  
    boundaries_df = labels_4326_gdf.bounds
    logger.info("- Tiling of the AoI")  
    tiles_4326_aoi_gdf = aoi_tiling(boundaries_df)
    tiles_4326_labels_gdf = gpd.sjoin(tiles_4326_aoi_gdf, labels_4326_gdf, how='inner', predicate='intersects')

    # Tiling of the AoI from which empty tiles will be selected
    if EPT_SHPFILE:
        EPT_aoi_gdf = gpd.read_file(EPT_SHPFILE)
        EPT_aoi_4326_gdf = EPT_aoi_gdf.to_crs(epsg=4326)
        assert_year(labels_4326_gdf, EPT_aoi_4326_gdf, 'empty_tiles', EPT_YEAR)
        
        if EPT_TYPE == 'aoi':
            logger.info("- Get AoI boundaries")  
            EPT_aoi_boundaries_df = EPT_aoi_4326_gdf.bounds

            # Get tile coordinates and shapes
            logger.info("- Tiling of the empty tiles AoI")  
            empty_tiles_4326_all_gdf = aoi_tiling(EPT_aoi_boundaries_df)
            # Delete tiles outside of the AoI limits 
            empty_tiles_4326_aoi_gdf = gpd.sjoin(empty_tiles_4326_all_gdf, EPT_aoi_4326_gdf, how='inner', lsuffix='ept_tiles', rsuffix='ept_aoi')
            # Attribute a year to empty tiles if necessary
            if 'year' in labels_4326_gdf.keys():
                if isinstance(EPT_YEAR, int):
                    empty_tiles_4326_aoi_gdf['year'] = int(EPT_YEAR)
                else:
                    empty_tiles_4326_aoi_gdf['year'] = np.random.randint(low=EPT_YEAR[0], high=EPT_YEAR[1], size=(len(empty_tiles_4326_aoi_gdf)))
        elif EPT_TYPE == 'shp':
            if EPT_YEAR:
                logger.warning("A shapefile of selected empty tiles are provided. The year set for the empty tiles in the configuration file will be ignored")
                EPT_YEAR = None
            empty_tiles_4326_aoi_gdf = EPT_aoi_4326_gdf.copy()

        # Get all the tiles in one gdf 
        logger.info("- Concatenate label tiles and empty AoI tiles") 
        tiles_4326_all_gdf = pd.concat([tiles_4326_labels_gdf, empty_tiles_4326_aoi_gdf])
    else: 
        tiles_4326_all_gdf = tiles_4326_labels_gdf.copy()

    # - Remove useless columns, reset feature id and redefine it according to xyz format  
    logger.info('- Add tile IDs and reorganise the data set')
    tiles_4326_all_gdf = tiles_4326_all_gdf[['geometry', 'title', 'year'] if 'year' in tiles_4326_all_gdf.keys() else ['geometry', 'title']].copy()
    tiles_4326_all_gdf.reset_index(drop=True, inplace=True)
    tiles_4326_all_gdf = tiles_4326_all_gdf.apply(add_tile_id, axis=1)

    # - Remove duplicated tiles
    tiles_4326_all_gdf.drop_duplicates(['id'], inplace=True)

    nb_tiles = len(tiles_4326_all_gdf)
    logger.info(f"There were {nb_tiles} tiles created")

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