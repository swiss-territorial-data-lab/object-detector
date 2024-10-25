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
from helpers.misc import format_logger
from helpers.constants import DONE_MSG

from loguru import logger
logger = format_logger(logger)


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


def assert_year(gdf1, gdf2, ds, year):
    """Assert if the year of the dataset is well supported

    Args:
        gdf1 (GeoDataFrame): label geodataframe
        gdf2 (GeoDataFrame): other geodataframe to compare columns
        ds (string): dataset type (FP, empty tiles,...)
        year (string or numeric): attribution of year to tiles
    """

    if ('year' not in gdf1.keys() and 'year' not in gdf2.keys()) or ('year' not in gdf1.keys() and year == None):
        pass
    elif ds == 'FP':
        if ('year' in gdf1.keys() and 'year' in gdf2.keys()):
            pass
        else:
            logger.error("One input label (GT or FP) shapefile contains a 'year' column while the other one no. Please, standardize the label shapefiles supplied as input data.")
            sys.exit(1)
    elif ds == 'empty_tiles':
        if ('year' in gdf1.keys() and 'year' in gdf2.keys()) or ('year' in gdf1.keys() and year != None):
            pass        
        elif 'year' in gdf1.keys() and 'year' not in gdf2.keys():
            logger.error("A 'year' column is provided in the GT shapefile but not for the empty tiles. Please, standardize the label shapefiles supplied as input data.")
            sys.exit(1)
        elif 'year' in gdf1.keys() and year == None:
            logger.error("A 'year' column is provided in the GT shapefile but no year info for the empty tiles. Please, provide a value to 'empty_tiles_year' in the configuration file.")
            sys.exit(1)
        elif ('year' not in gdf1.keys() and 'year' not in gdf2.keys()) and ('year' not in gdf1.keys() and year != None):
            logger.error("A year is provided for the empty tiles while no 'year' column is provided in the groud truth shapefile. Please, standardize the shapefiles or the year value in the configuration file.")
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
    CATEGORY = cfg['datasets']['category'] if 'category' in cfg['datasets'].keys() else False
    FP_SHPFILE = cfg['datasets']['FP_shapefile'] if 'FP_shapefile' in cfg['datasets'].keys() else None
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
    ZOOM_LEVEL = cfg['zoom_level']

    # Create an output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    written_files = []
    
    # Prepare the tiles

    ## Convert datasets shapefiles into geojson format
    logger.info('Convert labels shapefile into GeoJSON format (EPSG:4326)...')
    labels_gdf = gpd.read_file(SHPFILE)
    labels_4326_gdf = labels_gdf.to_crs(epsg=4326)
    labels_4326_gdf['CATEGORY'] = 'quarry'
    labels_4326_gdf['SUPERCATEGORY'] = 'land usage'
    gt_labels_4326_gdf = labels_4326_gdf.copy()

    nb_labels = len(labels_gdf)
    logger.info(f'There are {nb_labels} polygons in {SHPFILE}')

    label_filename = 'labels.geojson'
    label_filepath = os.path.join(OUTPUT_DIR, label_filename)
    labels_4326_gdf.to_file(label_filepath, driver='GeoJSON')
    written_files.append(label_filepath)  
    logger.success(f"{DONE_MSG} A file was written: {label_filepath}")

    # Add FP labels if it exists
    if FP_SHPFILE:
        fp_labels_gdf = gpd.read_file(FP_SHPFILE)
        assert_year(fp_labels_gdf, labels_gdf, 'FP', EPT_YEAR) 
        fp_labels_4326_gdf = fp_labels_gdf.to_crs(epsg=4326)
        fp_labels_4326_gdf['CATEGORY'] = 'quarry'
        fp_labels_4326_gdf['SUPERCATEGORY'] =  'land usage'

        nb_fp_labels = len(fp_labels_gdf)
        logger.info(f"There are {nb_fp_labels} polygons in {FP_SHPFILE}")

        filename = 'FP.geojson'
        filepath = os.path.join(OUTPUT_DIR, filename)
        fp_labels_4326_gdf.to_file(filepath, driver='GeoJSON')
        written_files.append(filepath)  
        logger.success(f"{DONE_MSG} A file was written: {filepath}")
        labels_4326_gdf = pd.concat([labels_4326_gdf, fp_labels_4326_gdf], ignore_index=True)

    # Get the label boundaries (minx, miny, maxx, maxy) 
    logger.info("- Get the label boundaries")  
    boundaries_df = labels_4326_gdf.bounds

    # Get the global boundaries for all the labels (minx, miny, maxx, maxy) 
    labels_bbox = bbox(labels_4326_gdf.iloc[0].geometry.bounds)

    # Get tiles for a given AoI from which empty tiles will be selected
    if EPT_SHPFILE:
        EPT_aoi_gdf = gpd.read_file(EPT_SHPFILE)
        EPT_aoi_4326_gdf = EPT_aoi_gdf.to_crs(epsg=4326)
        assert_year(labels_4326_gdf, EPT_aoi_4326_gdf, 'empty_tiles', EPT_YEAR)
        
        if EPT == 'aoi':
            logger.info("- Get AoI boundaries")  
            EPT_aoi_boundaries_df = EPT_aoi_4326_gdf.bounds

            # Get the boundaries for all the AoI (minx, miny, maxx, maxy) 
            aoi_bbox = bbox(EPT_aoi_4326_gdf.iloc[0].geometry.bounds)
            aoi_bbox_contains = aoi_bbox.contains(labels_bbox)

            if aoi_bbox_contains:
                logger.info("- The surface area occupied by the bbox of the AoI used to find empty tiles is bigger than the label's one. The AoI boundaries will be used for tiling") 
                boundaries_df = EPT_aoi_boundaries_df.copy()
            else:
                logger.info("- The surface area occupied by the bbox of the AoI used to find empty tiles is smaller than the label's one. Both the AoI and labels area will be used for tiling") 
                # Get tiles coordinates and shapes
                empty_tiles_4326_all_gdf = aoi_tiling(EPT_aoi_boundaries_df)
                # Delete tiles outside of the AoI limits 
                empty_tiles_4326_aoi_gdf = gpd.sjoin(empty_tiles_4326_all_gdf, EPT_aoi_4326_gdf, how='inner', lsuffix='ept_tiles', rsuffix='ept_aoi')
                # Attribute a year to empty tiles if necessary
                if 'year' in labels_4326_gdf.keys():
                    if isinstance(EPT_YEAR, int):
                        empty_tiles_4326_aoi_gdf['year'] = int(EPT_YEAR)
                    else:
                        empty_tiles_4326_aoi_gdf['year'] = np.random.randint(low=1945, high=2023, size=(len(empty_tiles_4326_aoi_gdf),))
        elif EPT == 'shp':
            if EPT_YEAR:
                logger.warning("A shapefile of selected empty tiles are provided. The year set for the empty tiles in the configuration file will be ignored")
                EPT_YEAR = None
            empty_tiles_4326_aoi_gdf = EPT_aoi_4326_gdf.copy()
            aoi_bbox = None
            aoi_bbox_contains = False

    logger.info('Creating tiles for the Area of Interest (AoI)...')   
    
    # Get tiles coordinates and shapes
    tiles_4326_aoi_gdf = aoi_tiling(boundaries_df)

    # Compute labels intersecting tiles 
    tiles_4326_lbl_gdf = gpd.sjoin(tiles_4326_aoi_gdf, gt_labels_4326_gdf, how='inner', predicate='intersects')
    tiles_4326_lbl_gdf.drop_duplicates('title', inplace=True)
    logger.info(f"- Number of tiles intersecting GT labels = {len(tiles_4326_lbl_gdf)}")
    
    if FP_SHPFILE:
        tiles_fp_4326_gdf = gpd.sjoin(tiles_4326_aoi_gdf, fp_labels_4326_gdf, how='inner', predicate='intersects')
        tiles_fp_4326_gdf.drop_duplicates('title', inplace=True)
        logger.info(f"- Number of tiles intersecting FP labels = {len(tiles_fp_4326_gdf)}")

    if not EPT_SHPFILE or EPT_SHPFILE and aoi_bbox_contains == False:
        # Keep only tiles intersecting labels 
        tiles_4326_aoi_gdf = tiles_4326_lbl_gdf.copy()

    # Get all the tiles in one gdf 
    if EPT_SHPFILE and aoi_bbox.contains(labels_bbox) == False:
        logger.info("- Add label tiles to empty AoI tiles") 
        tiles_4326_all_gdf = pd.concat([tiles_4326_aoi_gdf, empty_tiles_4326_aoi_gdf])
    else: 
        tiles_4326_all_gdf = tiles_4326_aoi_gdf.copy()
  
    # - Remove duplicated tiles
    if nb_labels > 1:
        tiles_4326_all_gdf.drop_duplicates(['title', 'year'] if 'year' in tiles_4326_all_gdf.keys() else 'title', inplace=True)

    # - Remove useless columns, reset feature id and redefine it according to xyz format  
    logger.info('- Add tile IDs and reorganise data set')
    tiles_4326_all_gdf = tiles_4326_all_gdf[['geometry', 'title', 'year'] if 'year' in tiles_4326_all_gdf.keys() else ['geometry', 'title']].copy()
    tiles_4326_all_gdf.reset_index(drop=True, inplace=True)
    tiles_4326_all_gdf = tiles_4326_all_gdf.apply(add_tile_id, axis=1)
    nb_tiles = len(tiles_4326_all_gdf)
    logger.info(f"There were {nb_tiles} tiles created")

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