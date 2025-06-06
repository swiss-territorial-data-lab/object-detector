#!/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import yaml
import re

import geopandas as gpd
import rasterio as rio
from shapely.geometry import Polygon

import helpers.functions_for_examples as ffe
import helpers.misc as misc

from loguru import logger
logger = misc.format_logger(logger)


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
    DATASET_KEYS = cfg['datasets'].keys()
    CANTON = cfg['canton'] if 'canton' in cfg.keys() else None
    CATEGORY = cfg['datasets']['category_field'] if 'category_field' in DATASET_KEYS else False
    OUTPUT_DIR = cfg['output_folder']
    SHPFILE = cfg['datasets']['shapefile']
    FP_SHPFILE = cfg['datasets']['fp_shapefile'] if 'fp_shapefile' in DATASET_KEYS else None
    EPT_YEAR = cfg['datasets']['empty_tiles_year'] if 'empty_tiles_year' in DATASET_KEYS else None
    DEM = cfg['dem'] if 'dem' in cfg.keys() else None

    EPT_SHPFILE = None
    EPT = None
    if 'empty_tiles_aoi' in DATASET_KEYS and 'empty_tiles_shp' in DATASET_KEYS:
        logger.error("Choose between supplying an AoI shapefile ('empty_tiles_aoi') in which empty tiles will be selected, or a shapefile with selected empty tiles ('empty_tiles_shp')")
        sys.exit(1)    
    elif 'empty_tiles_aoi' in DATASET_KEYS:
        EPT_SHPFILE = cfg['datasets']['empty_tiles_aoi']
        EPT = 'aoi'
    elif 'empty_tiles_shp' in DATASET_KEYS:
        EPT_SHPFILE = cfg['datasets']['empty_tiles_shp'] 
        EPT = 'shp'

    WATERS = None
    ELEVATION_THD = None
    if CANTON == 'vaud':
        WATERS = 'data/layers/vaud/lakes_VD.gpkg'
    elif CANTON == 'ticino':
        WATERS = 'data/layers/ticino/MU_Acque_TI_dissolved.gpkg'
        ELEVATION_THD = 1000
    elif CANTON:
        logger.critical(f'Unknown canton: {CANTON}')
        sys.exit(1)
    logger.info(f'Using cantonal parameters:')
    logger.info(f'    - water bodies: {WATERS}')
    logger.info(f'    - elevation threshold: {ELEVATION_THD}')
    
    ZOOM_LEVEL = cfg['zoom_level']
    OVERWRITE = cfg['overwrite']

    # Create an output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    written_files = []

    label_filepath = os.path.join(OUTPUT_DIR, 'labels.geojson')
    tile_filepath = os.path.join(OUTPUT_DIR, 'tiles.geojson')
    fp_filepath = os.path.join(OUTPUT_DIR, 'FP.geojson')
    if os.path.exists(tile_filepath) and (not FP_SHPFILE or os.path.exists(fp_filepath)) and not OVERWRITE:           
        logger.success(f"Done! All files already exist in folder {OUTPUT_DIR}. Exiting.")
        sys.exit(0)
    
    # Prepare the tiles

    ## Convert datasets shapefiles into geojson format
    logger.info('Convert labels shapefile into GeoJSON format (EPSG:4326)...')
    labels_gdf = gpd.read_file(SHPFILE)
    if 'year' in labels_gdf.keys():
        labels_gdf['year'] = labels_gdf.year.astype(int)
        labels_4326_gdf = labels_gdf.to_crs(epsg=4326).drop_duplicates(subset=['geometry', 'year'])
    else:
        labels_4326_gdf = labels_gdf.to_crs(epsg=4326).drop_duplicates(subset=['geometry'])
    nb_labels = len(labels_4326_gdf)
    logger.info(f'There are {nb_labels} polygons in {SHPFILE}')

    topic = 'anthropogenic soils'
    if CATEGORY and CATEGORY in labels_4326_gdf.keys():
        labels_4326_gdf['CATEGORY'] = labels_4326_gdf[CATEGORY]
        category = labels_4326_gdf['CATEGORY'].unique()
        logger.info(f'Working with {len(category)} class.es: {category}')
        labels_4326_gdf['SUPERCATEGORY'] = topic
    else:
        logger.warning(f'No category column in {SHPFILE}. A unique category will be assigned')
        labels_4326_gdf['CATEGORY'] = topic
        labels_4326_gdf['SUPERCATEGORY'] = topic

    gt_labels_4326_gdf = labels_4326_gdf.copy()
    
    gt_labels_4326_gdf.to_file(label_filepath, driver='GeoJSON')
    written_files.append(label_filepath)  
    logger.success(f"Done! A file was written: {label_filepath}")

    tiles_4326_all_gdf, tmp_written_files = ffe.format_all_tiles(
        FP_SHPFILE, fp_filepath, EPT_SHPFILE, ept_data_type=EPT, labels_gdf=gt_labels_4326_gdf, category=CATEGORY, supercategory=topic, zoom_level=ZOOM_LEVEL
    )
    written_files.extend(tmp_written_files)

    if WATERS:
        logger.info('Remove tiles in water...')
        waters_gdf = gpd.read_file(WATERS).to_crs(2056)
        waters_gdf.loc[:, 'geometry'] = waters_gdf.buffer(-420)     # Remove borders for the size of a tile
        waters_poly = waters_gdf[~waters_gdf.geometry.is_empty].unary_union

        tiles_2056_all_gdf = tiles_4326_all_gdf.to_crs(epsg=2056)
        tiles_2056_all_gdf.loc[:, 'geometry'] = tiles_2056_all_gdf.centroid
        tiles_in_waters = tiles_2056_all_gdf.loc[tiles_2056_all_gdf.intersects(waters_poly), 'id'].tolist()
        tiles_4326_all_gdf = tiles_4326_all_gdf[~tiles_4326_all_gdf.id.isin(tiles_in_waters)]

    if ELEVATION_THD:
        logger.info("Control altitude...")
        dem = rio.open(DEM)
        tiles_4326_all_gdf = misc.check_validity(tiles_4326_all_gdf, correct=True)
        tiles_2056_all_gdf = tiles_4326_all_gdf.to_crs(epsg=2056)

        row, col = dem.index(tiles_2056_all_gdf.centroid.x, tiles_2056_all_gdf.centroid.y)
        elevation = dem.read(1)[row, col]
        tiles_4326_all_gdf['elevation'] = elevation
        tiles_4326_all_gdf = tiles_4326_all_gdf[tiles_4326_all_gdf.elevation < ELEVATION_THD]

    # Save tile shapefile
    if tiles_4326_all_gdf.empty:
        logger.warning('No tile generated for the designated area.')
        tile_filepath = os.path.join(OUTPUT_DIR, 'area_without_tiles.gpkg')
        labels_gdf.to_file(tile_filepath)
        written_files.append(tile_filepath)  
    else:
        logger.info("Export tiles to GeoJSON (EPSG:4326)...") 
        tiles_4326_all_gdf.to_file(tile_filepath, driver='GeoJSON')
        written_files.append(tile_filepath)  
        logger.success(f"Done! A file was written: {tile_filepath}")

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")
    
    sys.stderr.flush()