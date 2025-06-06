#!/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import yaml
import re
from tqdm import tqdm

import geopandas as gpd
import morecantile
import numpy as np
import pandas as pd
import rasterio as rio
from shapely.geometry import Polygon

sys.path.insert(0, '.')
import misc as misc

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
    for boundary in tqdm(gdf.itertuples(), desc='Tiling AOI parts', total=len(gdf)):
        coords = (boundary.minx, boundary.miny, boundary.maxx, boundary.maxy)      
        tiles = gpd.GeoDataFrame.from_features([tms.feature(x, projected=False) for x in tms.tiles(*coords, zooms=[ZOOM_LEVEL])]) 
        tiles.set_crs(epsg=4326, inplace=True)
        tiles_all.append(tiles)
    tiles_all_gdf = gpd.GeoDataFrame(pd.concat(tiles_all, ignore_index=True)).drop_duplicates(subset=['title'], keep='first')

    return tiles_all_gdf


def assert_year(gdf1, gdf2, ds, year):
    """Assert if the year of the dataset is well supported

    Args:
        gdf1 (GeoDataFrame): label geodataframe
        gdf2 (GeoDataFrame): other geodataframe to compare columns
        ds (string): dataset type (FP, empty tiles,...)
        year (string or numeric): attribution of year to tiles
    """

    gdf1_has_year = 'year' in gdf1.keys()
    gdf2_has_year = 'year' in gdf2.keys()
    param_gives_year = year != None

    if gdf1_has_year or (gdf2_has_year and param_gives_year):   # year for label or double year info oth
        if ds == 'FP':
            if not (gdf1_has_year and gdf2_has_year):   
                logger.error("One input label (GT or FP) shapefile contains a 'year' column while the other one does not. Please, standardize the label shapefiles supplied as input data.")
                sys.exit(1)
        elif ds == 'empty_tiles':
            if gdf1_has_year:
                if not gdf2_has_year:
                    logger.error("A 'year' column is provided in the GT shapefile but not for the empty tiles. Please, standardize the label shapefiles supplied as input data.")
                    sys.exit(1)
                elif  not param_gives_year:
                    logger.error("A 'year' column is provided in the GT shapefile but no year info for the empty tiles. Please, provide a value to 'empty_tiles_year' in the configuration file.")
                    sys.exit(1)
            elif gdf2_has_year or param_gives_year: # "not gdf1_has_year" is implied by elif-statement
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

    # Add FP label dataset if it exists
    if FP_SHPFILE:
        fp_labels_gdf = gpd.read_file(FP_SHPFILE)
        assert_year(fp_labels_gdf, labels_gdf, 'FP', EPT_YEAR) 
        if 'year' in fp_labels_gdf.keys():
            fp_labels_gdf['year'] = fp_labels_gdf.year.astype(int)
            fp_labels_4326_gdf = fp_labels_gdf.to_crs(epsg=4326).drop_duplicates(subset=['geometry', 'year'])
        else:
            fp_labels_4326_gdf = fp_labels_gdf.to_crs(epsg=4326).drop_duplicates(subset=['geometry'])
        if CATEGORY:
            fp_labels_4326_gdf['CATEGORY'] = fp_labels_4326_gdf[CATEGORY]
            fp_labels_4326_gdf['SUPERCATEGORY'] = topic

        nb_fp_labels = len(fp_labels_4326_gdf)
        logger.info(f"There are {nb_fp_labels} polygons in {FP_SHPFILE}")

        fp_labels_4326_gdf.to_file(fp_filepath, driver='GeoJSON')
        written_files.append(fp_filepath)  
        logger.success(f"Done! A file was written: {fp_filepath}")
        labels_4326_gdf = pd.concat([labels_4326_gdf, fp_labels_4326_gdf], ignore_index=True)

    # Tiling of the AoI
    logger.info("- Get the label boundaries")  
    boundaries_df = labels_4326_gdf.bounds
    logger.info("- Tiling of the AoI")  
    PRE_EXISTING_TILING = f'data/layers/{CANTON}/tiles_z{ZOOM_LEVEL}_w_dets.geojson'
    if os.path.exists(PRE_EXISTING_TILING):
        logger.info(f'Using existing tiling: {PRE_EXISTING_TILING}')
        tiles_4326_aoi_gdf = gpd.read_file(PRE_EXISTING_TILING)
    else:
        tiles_4326_aoi_gdf = aoi_tiling(boundaries_df)
    tiles_4326_labels_gdf = gpd.sjoin(tiles_4326_aoi_gdf, labels_4326_gdf, how='inner', predicate='intersects')

    # Tiling of the AoI from which empty tiles will be selected
    if EPT_SHPFILE:
        EPT_aoi_gdf = gpd.read_file(EPT_SHPFILE)
        EPT_aoi_4326_gdf = EPT_aoi_gdf.to_crs(epsg=4326)
        assert_year(labels_4326_gdf, EPT_aoi_4326_gdf, 'empty_tiles', EPT_YEAR)
        
        if EPT == 'aoi':
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
            elif EPT_SHPFILE and EPT_YEAR: 
                logger.warning("No year column in the label shapefile. The provided empty tile year will be ignored.")
        elif EPT == 'shp':
            if EPT_YEAR:
                logger.warning("A shapefile of selected empty tiles are provided. The year set for the empty tiles in the configuration file will be ignored")
                EPT_YEAR = None
            empty_tiles_4326_aoi_gdf = EPT_aoi_4326_gdf.copy()

    # Get all the tiles in one gdf 
    if EPT_SHPFILE:
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
    if nb_labels > 1:
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