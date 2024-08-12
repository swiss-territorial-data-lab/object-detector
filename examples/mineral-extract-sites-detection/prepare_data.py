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


def aoi_tiling(gdf):
    """Tiling of an AoI

    Args:
        gdf (GeoDataFrame): gdf containing all the bbox boundary coordinates

    Returns:
        Geodataframe: gdf containing the tiles shape of the bbox of the AoI
    """

    # Grid definition
    tms = morecantile.tms.get('WebMercatorQuad')    # epsg:3857

    tiles_all = [] 
    for boundary in gdf.itertuples():
        coords = (boundary.minx, boundary.miny, boundary.maxx, boundary.maxy)      
        tiles = gpd.GeoDataFrame.from_features([tms.feature(x, projected=False) for x in tms.tiles(*coords, zooms=[ZOOM_LEVEL])]) 
        tiles.set_crs(epsg=4326, inplace=True)
        tiles_all.append(tiles)
    tiles_all_gdf = gpd.GeoDataFrame(pd.concat(tiles_all, ignore_index=True))

    return tiles_all_gdf


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
    parser = argparse.ArgumentParser(description="The script prepares the Mineral Extraction Sites dataset to be processed by the object-detector scripts")
    parser.add_argument('config_file', type=str, help="Framework configuration file")
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")
 
    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    OUTPUT_DIR = cfg['output_folder']
    SHPFILE = cfg['datasets']['shapefile']
    FP_SHPFILE = cfg['datasets']['FP_shapefile'] if 'FP_shapefile' in cfg['datasets'].keys() else None
    if 'empty_tiles_aoi' in cfg['datasets'].keys() and 'empty_tiles_shp' in cfg['datasets'].keys():
        logger.error("Choose one option between providing an AoI shapefile ('empty_tiles_aoi') in which empty tiles will be selected and a shapefile with selected empty tiles ('empty_tiles_shp')")
        sys.exit(1)    
    if 'empty_tiles_aoi' in cfg['datasets'].keys():
        EPT_SHPFILE = cfg['datasets']['empty_tiles_aoi']
        EPT = 'aoi'
    elif 'empty_tiles_shp' in cfg['datasets'].keys():
        EPT_SHPFILE = cfg['datasets']['empty_tiles_shp'] 
        EPT = 'shp'
    else:
        EPT_SHPFILE = None
        EPT = None
    ZOOM_LEVEL = cfg['zoom_level']

    # Create an output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    written_files = []
    
    # Prepare the tiles

    # Convert datasets shapefiles into geojson format
    logger.info("Convert labels shapefile into GeoJSON format (EPSG:4326)...")
    labels = gpd.read_file(SHPFILE)
    labels_4326 = labels.to_crs(epsg=4326)
    labels_4326['CATEGORY'] = 'quarry'
    labels_4326['SUPERCATEGORY'] = 'land usage'

    nb_labels = len(labels)
    logger.info(f"There are {nb_labels} polygons in {SHPFILE}")

    filename = 'labels.geojson'
    filepath = os.path.join(OUTPUT_DIR, filename)
    labels_4326.to_file(filepath, driver='GeoJSON')
    written_files.append(filepath)  
    logger.success(f"{DONE_MSG} A file was written: {filepath}")

    # Add FP labels if it exists
    if FP_SHPFILE:
        fp_labels = gpd.read_file(FP_SHPFILE)
        fp_labels_4326 = fp_labels.to_crs(epsg=4326)
        fp_labels_4326['CATEGORY'] = 'quarry'
        fp_labels_4326['SUPERCATEGORY'] = 'land usage'

        nb_fp_labels = len(fp_labels)
        logger.info(f"There are {nb_fp_labels} polygons in {FP_SHPFILE}")

        filename = 'FP.geojson'
        filepath = os.path.join(OUTPUT_DIR, filename)
        fp_labels_4326.to_file(filepath, driver='GeoJSON')
        written_files.append(filepath)  
        logger.success(f"{DONE_MSG} A file was written: {filepath}")

        labels_4326 = pd.concat([labels_4326, fp_labels_4326], ignore_index=True)
    else:
        labels_4326 = labels_4326

    # Keep only label boundary geometry info (minx, miny, maxx, maxy) 
    logger.info("- Get the label boundaries")  
    boundaries_df = labels_4326.bounds

    # Get the boundaries for all the labels (minx, miny, maxx, maxy) 
    global_boundaries_gdf = labels_4326.dissolve() if len(labels_4326) > 0 else labels_4326
    labels_bbox = bbox(global_boundaries_gdf.iloc[0].geometry.bounds)

    # Get tiles for a given AoI from which empty tiles will be selected when the images are retrieved
    # Get tiles for a given AoI from which empty tiles will be selected when the images are retrieved
    if EPT_SHPFILE:
        EPT_aoi = gpd.read_file(EPT_SHPFILE)
        EPT_aoi_4326 = EPT_aoi.to_crs(epsg=4326)
        
        if EPT == 'aoi':
            logger.info("- Get AoI boundaries")  
            EPT_aoi_boundaries_df = EPT_aoi_4326.bounds

            # Get the boundaries for all the AoI (minx, miny, maxx, maxy) 
            EPT_aoi_boundaries_gdf = EPT_aoi_4326.dissolve() if len(EPT_aoi_4326) > 0 else EPT_aoi_4326
            aoi_bbox = bbox(EPT_aoi_boundaries_gdf.iloc[0].geometry.bounds)
            aoi_bbox_contains = aoi_bbox.contains(labels_bbox)

            if aoi_bbox_contains:
                logger.info("- The surface area occupied by the bbox of the AoI used to find empty tiles is bigger than the label's one. The AoI boundaries will be used for tiling") 
                boundaries_df = EPT_aoi_boundaries_df
            else:
                logger.info("- The surface area occupied by the bbox of the AoI used to find empty tiles is smaller than the label's one. Both the AoI and labels area will be used for tiling") 
                # Get tiles coordinates and shapes
                empty_tiles_4326_all = aoi_tiling(EPT_aoi_boundaries_df)
                # Delete tiles outside of the AoI limits 
                empty_tiles_4326_aoi = gpd.sjoin(empty_tiles_4326_all, EPT_aoi_4326, how='inner', lsuffix='ept_tiles', rsuffix='ept_aoi')
        elif EPT == 'shp':
            empty_tiles_4326_aoi = EPT_aoi_4326
            aoi_bbox = None
            aoi_bbox_contains = False

    logger.info("Creating tiles for the Area of Interest (AoI)...")   

    # Get tiles coordinates and shapes
    tiles_4326_aoi = aoi_tiling(boundaries_df)

    if EPT_SHPFILE and aoi_bbox_contains:
        # Delete tiles outside of the AoI limits 
        tiles_4326_aoi = gpd.sjoin(tiles_4326_aoi, EPT_aoi_4326, how='inner', lsuffix='ept_tiles', rsuffix='ept_aoi', predicate='intersects')

    # Compute labels intersecting tiles 
    tiles_gt_4326 = gpd.sjoin(tiles_4326_aoi, labels_4326, how='inner', predicate='intersects')
    tiles_gt_4326.drop_duplicates('title', inplace=True)
    logger.info(f"- Number of tiles intersecting GT labels = {len(tiles_gt_4326)}")
    if FP_SHPFILE:
        tiles_fp_4326 = gpd.sjoin(tiles_4326_aoi, fp_labels_4326, how='inner', predicate='intersects')
        tiles_fp_4326.drop_duplicates('title', inplace=True)
        logger.info(f"- Number of tiles intersecting FP labels = {len(tiles_fp_4326)}")

    if not EPT_SHPFILE or EPT_SHPFILE and aoi_bbox_contains == False:
        # Keep only tiles intersecting labels 
        tiles_4326_aoi = gpd.sjoin(tiles_4326_aoi, labels_4326, how='inner', predicate='intersects')
        tiles_4326_aoi.drop_duplicates('title', inplace=True)
 
    # Get all the tiles in one gdf 
    if EPT_SHPFILE and aoi_bbox_contains == False:
        logger.info("- Add label tiles to empty AoI tiles") 
        tiles_4326_all = pd.concat([tiles_4326_aoi, empty_tiles_4326_aoi])
    else: 
        tiles_4326_all = tiles_4326_aoi
    
    tiles_4326_all.drop_duplicates('title', inplace=True)

    # Add tile IDs and reorganise data set
    tiles_4326_all = tiles_4326_all[['geometry', 'title']].copy()
    tiles_4326_all.reset_index(drop=True, inplace=True)
    tiles_4326_all = tiles_4326_all.apply(add_tile_id, axis=1)
    
    nb_tiles = len(tiles_4326_all)
    logger.info(f"There were {nb_tiles} tiles created")

    # Save tile shapefile
    logger.info("Export tiles to GeoJSON (EPSG:4326)...")  
    tile_filename = 'tiles.geojson'
    tile_filepath = os.path.join(OUTPUT_DIR, tile_filename)
    tiles_4326_all.to_file(tile_filepath, driver='GeoJSON')
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