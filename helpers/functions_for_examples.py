import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import geopandas as gpd
import json
import morecantile
import numpy as np
import pandas as pd
from loguru import logger
from re import search
from tqdm import tqdm

try:
    from constants import DONE_MSG
    from misc import check_validity
except:
    from helpers.constants import DONE_MSG
    from helpers.misc import check_validity

def add_tile_id(row):
    """Attribute tile id

    Args:
        row (DataFrame): row of a given df

    Returns:
        DataFrame: row with addition 'id' column
    """

    re_search = search('(x=(?P<x>\d*), y=(?P<y>\d*), z=(?P<z>\d*))', row.title)
    if 'year' in row.keys():
        row['id'] = f"({row.year}, {re_search.group('x')}, {re_search.group('y')}, {re_search.group('z')})"
    else:
        row['id'] = f"({re_search.group('x')}, {re_search.group('y')}, {re_search.group('z')})"
 
    return row


def aoi_tiling(gdf, zoom_level, tms='WebMercatorQuad'):
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
        tiles = gpd.GeoDataFrame.from_features([tms.feature(x, projected=False) for x in tms.tiles(*coords, zooms=[zoom_level])]) 
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

    gdf1_has_year = 'year' in gdf1.columns
    gdf2_has_year = 'year' in gdf2.columns
    param_gives_year = year != None

    if gdf1_has_year or gdf2_has_year or param_gives_year:   # if any info about year exists, control
        if ds == 'FP' and gdf1_has_year != gdf2_has_year:   
            logger.error("One input label (GT or FP) shapefile contains a 'year' column while the other one does not. Please, standardize the label shapefiles supplied as input data.")
            sys.exit(1)
        elif ds == 'empty_tiles':
            if gdf1_has_year and not (gdf2_has_year or param_gives_year):
                logger.error("A 'year' column is provided in the GT shapefile but not for the empty tiles. Please, standardize the shapefiles or provide a value to 'empty_tiles_year' in the configuration file.")
                sys.exit(1)
            elif not gdf1_has_year and (gdf2_has_year or param_gives_year):
                logger.error("A year is provided for the empty tiles while no 'year' column is provided in the groud truth shapefile. Please, standardize the shapefiles or the year value in the configuration file.")
                sys.exit(1)


def format_all_tiles(fp_labels_shp,  ept_labels_shp, ept_data_type, ept_year, labels_4326_gdf, category, supercategory, zoom_level, output_dir='outputs'):
    """
    Format all tiles of a given area from a geodataframe.

    Args:
        fp_labels_shp (str): path to the file containing the false positive labels
        ept_labels_shp (str): path to the file containing the empty tiles labels
        ept_data_type (str): type of empty tiles ('aoi' or 'shp')
        ept_year (str or numeric): year to attribute to empty tiles
        labels_4326_gdf (GeoDataFrame): gdf containing the formatted ground truth labels
        category (str): category of the dataset
        supercategory (str): supercategory of the dataset
        zoom_level (int): zoom level of the tiles
        output_dir (str): directory where the output files will be saved

    Returns:
        GeoDataFrame: gdf containing all the tiles of the given area
        list: list of files written
    """
    written_files = []

    # Add FP labels if it exists
    if fp_labels_shp:
        logger.info("- Get FP labels")
        fp_labels_4326_gdf, _ = prepare_labels(fp_labels_shp, category=category, supercategory=supercategory, prefix='FP', output_dir=output_dir)
        labels_4326_gdf = pd.concat([labels_4326_gdf, fp_labels_4326_gdf], ignore_index=True)

    # Tiling of the AoI
    logger.info("- Get the label boundaries")  
    boundaries_df = labels_4326_gdf.bounds
    logger.info("- Tiling of the AoI")  
    tiles_4326_aoi_gdf = aoi_tiling(boundaries_df, zoom_level)
    tiles_4326_labels_gdf = gpd.sjoin(tiles_4326_aoi_gdf, labels_4326_gdf, how='inner', predicate='intersects')

    # Tiling of the AoI from which empty tiles will be selected
    if ept_labels_shp:
        ept_aoi_gdf = gpd.read_file(ept_labels_shp)
        ept_aoi_4326_gdf = ept_aoi_gdf.to_crs(epsg=4326)
        assert_year(labels_4326_gdf, ept_aoi_4326_gdf, 'empty_tiles', ept_year)
        
        if ept_data_type == 'aoi':
            logger.info("- Get AoI boundaries")  
            ept_aoi_boundaries_df = ept_aoi_4326_gdf.bounds

            # Get tile coordinates and shapes
            logger.info("- Tiling of the empty tiles AoI")  
            empty_tiles_4326_all_gdf = aoi_tiling(ept_aoi_boundaries_df, zoom_level)
            # Delete tiles outside of the AoI limits 
            empty_tiles_4326_aoi_gdf = gpd.sjoin(empty_tiles_4326_all_gdf, ept_aoi_4326_gdf, how='inner', lsuffix='ept_tiles', rsuffix='ept_aoi')
            # Attribute a year to empty tiles if necessary
            if 'year' in labels_4326_gdf.columns:
                if 'year' not in empty_tiles_4326_aoi_gdf.columns:
                    empty_tiles_4326_aoi_gdf['year'] = int(ept_year)
                else:
                    empty_tiles_4326_aoi_gdf['year'] = empty_tiles_4326_aoi_gdf.year.astype(int)

        elif ept_data_type == 'shp':
            if ept_year:
                logger.warning("A shapefile of selected empty tiles are provided. The year set for the empty tiles in the configuration file will be ignored")
                ept_year = None
            empty_tiles_4326_aoi_gdf = ept_aoi_4326_gdf.copy()
            empty_tiles_4326_aoi_gdf['year'] = empty_tiles_4326_aoi_gdf.year.astype(int)

        # Get all the tiles in one gdf 
        logger.info("- Concatenate label tiles and empty AoI tiles") 
        tiles_4326_all_gdf = pd.concat([tiles_4326_labels_gdf, empty_tiles_4326_aoi_gdf])

    else: 
        tiles_4326_all_gdf = tiles_4326_labels_gdf.copy()

    # - Remove useless columns, reset feature id and redefine it according to xyz format  
    logger.info('- Add tile IDs and reorganise the data set')
    tiles_4326_all_gdf = tiles_4326_all_gdf[['geometry', 'title'] + (['year'] if 'year' in tiles_4326_all_gdf.columns else [])].copy()
    tiles_4326_all_gdf.reset_index(drop=True, inplace=True)
    tiles_4326_all_gdf = tiles_4326_all_gdf.apply(add_tile_id, axis=1)
    tiles_4326_all_gdf.drop_duplicates(['id'], inplace=True)

    nb_tiles = len(tiles_4326_all_gdf)
    logger.info(f"There were {nb_tiles} tiles created")

    # Get the number of tiles intersecting labels
    tiles_4326_gt_gdf = gpd.sjoin(tiles_4326_all_gdf, labels_4326_gdf[['geometry', 'CATEGORY', 'SUPERCATEGORY']], how='inner', predicate='intersects')
    tiles_4326_gt_gdf.drop_duplicates(['id'], inplace=True)
    logger.info(f"- Number of tiles intersecting GT labels = {len(tiles_4326_gt_gdf)}")

    if fp_labels_shp:
        tiles_4326_fp_gdf = gpd.sjoin(tiles_4326_all_gdf, fp_labels_4326_gdf, how='inner', predicate='intersects')
        tiles_4326_fp_gdf.drop_duplicates(['id'], inplace=True)
        logger.info(f"- Number of tiles intersecting FP labels = {len(tiles_4326_fp_gdf)}")

    # Save tile shapefile
    tile_filepath = os.path.join(output_dir, 'tiles.gpkg')
    if tiles_4326_all_gdf.empty:
        logger.warning('No tile generated for the designated area.')
        tile_filepath = os.path.join(output_dir, 'area_without_tiles.gpkg')
        labels_4326_gdf.to_file(tile_filepath)
        written_files.append(tile_filepath)  
    else:
        logger.info("Export tiles to geopackage (EPSG:4326)...") 
        tiles_4326_all_gdf.to_file(tile_filepath)
        written_files.append(tile_filepath)  
        logger.success(f"Done! A file was written: {tile_filepath}")

    return tiles_4326_all_gdf, written_files


def get_bbox_origin(bbox_geom):
    """Get the lower xy coorodinates of a bounding box.

    Args:
        bbox_geom (geometry): bounding box

    Returns:
        tuple: lower xy coordinates of the passed geometry
    """

    coords = bbox_geom.exterior.coords.xy
    min_x = min(coords[0])
    min_y = min(coords[1])

    return (min_x, min_y)


def get_categories(filepath):
    """
    Get the categories from a JSON file.

    Args:
        filepath (str): path to the JSON file

    Returns:
        tuple: a DataFrame containing the categories and their label classes, and a list of the label classes

    The JSON file should have the following structure:
    {
        "category": {
            "id": int,
            "name": str,
            "supercategory": str
        }
    }
    """
    category_file = open(filepath)
    categories_json = json.load(category_file)
    category_file.close()
    categories_info_df = pd.DataFrame()
    for key in categories_json.keys():
        categories_tmp = {sub_key: [value] for sub_key, value in categories_json[key].items()}
        categories_info_df = pd.concat([categories_info_df, pd.DataFrame(categories_tmp)], ignore_index=True)
    categories_info_df.sort_values(by=['id'], inplace=True, ignore_index=True)
    categories_info_df.drop(['supercategory'], axis=1, inplace=True)
    categories_info_df.rename(columns={'name':'category', 'id': 'label_class'},inplace=True)
    id_classes = range(len(categories_json))

    return categories_info_df, id_classes


def get_tile_name(path, geom):
    # Determine the name of the new tile for the example of border points

    (min_x, min_y) = get_bbox_origin(geom)
    tile_nbr = int(os.path.basename(path).split("_")[0])
    new_name = f"{tile_nbr}_{round(min_x)}_{round(min_y)}.tif"

    return new_name


def merge_polygons(gdf, id_name='id'):
    '''
    Merge overlapping polygons in a GeoDataFrame.

    - gdf: GeoDataFrame with polygon geometries
    - id_name (string): name of the index column

    return: a GeoDataFrame with polygons
    '''

    merge_gdf = gpd.GeoDataFrame(geometry=[gdf.geometry.unary_union], crs=gdf.crs) 
    merge_gdf = merge_gdf.explode(ignore_index=True)
    merge_gdf[id_name] = merge_gdf.index 

    return merge_gdf


def merge_adjacent_detections(detections_gdf, tiles_gdf, year=None, buffer_distance=1):
    """
    Merge adjacent detections and tiles. The function takes a GeoDataFrame of detections and tiles, and a year.
    It will merge overlapping polygons within the tile. It will also merge adjacent polygons between tiles with a buffer. 
    The function returns two GeoDataFrames: one for the merged polygons within the tile and one for the merged polygons between tiles.

    Parameters:
        detections_gdf (GeoDataFrame): GeoDataFrame of detections
        tiles_gdf (GeoDataFrame): GeoDataFrame of tiles
        year (int): year of the detections
        buffer_distance (int): distance in meters applied to the detections when testing adjacent tiles

    Returns:
        complete_merge_dets_gdf (GeoDataFrame): GeoDataFrame of merged polygons between tiles
        detections_within_tiles_gdf (GeoDataFrame): GeoDataFrame of merged polygons within the tile
    """
    if year:
        _detections_gdf = detections_gdf[detections_gdf['year_det']==year].copy()
    else:
        _detections_gdf = detections_gdf.copy()

    # Merge overlapping polygons
    detections_merge_overlap_poly_gdf = merge_polygons(_detections_gdf, id_name='det_id')

    # Saves the ids of polygons contained entirely within the tile (no merging with adjacent tiles), to avoid merging them if they are at a distance of less than thd  
    detections_buffer_gdf = detections_merge_overlap_poly_gdf.copy()
    detections_buffer_gdf['geometry'] = detections_buffer_gdf.geometry.buffer(buffer_distance, join_style='mitre')
    detections_tiles_join_gdf = gpd.sjoin(tiles_gdf, detections_buffer_gdf, how='left', predicate='contains')
    remove_det_list = detections_tiles_join_gdf.det_id.unique().tolist()
    
    detections_within_tiles_gdf = detections_merge_overlap_poly_gdf[
        detections_merge_overlap_poly_gdf.det_id.isin(remove_det_list)
    ].drop_duplicates(subset=['det_id'], ignore_index=True)

    # Merge adjacent polygons between tiles
    detections_overlap_tiles_gdf = detections_buffer_gdf[~detections_buffer_gdf.det_id.isin(remove_det_list)].drop_duplicates(subset=['det_id'], ignore_index=True)
    detections_merge_gdf = merge_polygons(detections_overlap_tiles_gdf)
    detections_merge_gdf['geometry'] = detections_merge_gdf.geometry.buffer(-buffer_distance, join_style='mitre')
    detections_merge_gdf = detections_merge_gdf.explode(ignore_index=True)
    detections_merge_gdf['id'] = detections_merge_gdf.index

    # Spatially join merged detection with raw ones to retrieve relevant information (score, area,...)
    # Select the class of the largest polygon -> To Do: compute a parameter dependant of the area and the score
    # Score averaged over all the detection polygon (even if the class is different from the selected one)
    detections_join_gdf = gpd.sjoin(detections_merge_gdf, _detections_gdf, how='inner', predicate='intersects')

    det_class_all = []
    det_score_all = []

    for id in detections_merge_gdf.id.unique():
        _detections_gdf = detections_join_gdf[(detections_join_gdf['id']==id)].rename(columns={'score_left': 'score'})
        det_score_all.append(_detections_gdf['score'].mean())
        _detections_gdf = _detections_gdf.dissolve(by='det_class', aggfunc='sum', as_index=False)
        if len(_detections_gdf) > 0:
            _detections_gdf['det_class'] = _detections_gdf.loc[
                _detections_gdf['area'] == _detections_gdf['area'].max(), 'det_class'
            ].iloc[0]    
            det_class = _detections_gdf['det_class'].drop_duplicates().tolist()
        else:
            det_class = [0]
        det_class_all.append(det_class[0])

    detections_merge_gdf['det_class'] = det_class_all
    detections_merge_gdf['score'] = det_score_all
    
    complete_merge_dets_gdf = pd.merge(detections_merge_gdf, detections_join_gdf[
        ['id', 'year_det'] + ([] if 'dataset' in detections_merge_gdf.columns else ['dataset'])
    ], on='id')
    
    return complete_merge_dets_gdf, detections_within_tiles_gdf

def prepare_labels(labels_shp, category, supercategory, prefix='', output_dir='outputs'):
    """
    Prepare a shapefile of labels into a formatted GeoPandas DataFrame.

    Args:
        labels_shp (string): path to the shapefile of labels
        category (string): column name of the category
        supercategory (string): column name of the supercategory
        prefix (string): prefix for the output filename
        output_dir (string): output directory for the GeoPackage

    Returns:
        labels_4326_gdf (GeoDataFrame): formatted GeoPandas DataFrame of labels
        written_files (list): list of written files
    """
    logger.info('Convert labels shapefile into formatted geopackage (EPSG:4326)...')
    labels_gdf = gpd.read_file(labels_shp)
    labels_gdf = check_validity(labels_gdf, correct=True)
    if 'year' in labels_gdf.columns:
        labels_gdf['year'] = labels_gdf.year.astype(int)
        labels_4326_gdf = labels_gdf.to_crs(epsg=4326).drop_duplicates(subset=['geometry', 'year'])
    else:
        labels_4326_gdf = labels_gdf.to_crs(epsg=4326).drop_duplicates(subset=['geometry'])
    nb_labels = len(labels_4326_gdf)
    logger.info(f'There are {nb_labels} polygons in {labels_shp}')

    if category and category in labels_4326_gdf.columns:
        labels_4326_gdf['CATEGORY'] = labels_4326_gdf[category]
        category = labels_4326_gdf['CATEGORY'].unique()
        logger.info(f'Working with {len(category)} class.es: {category}')
        labels_4326_gdf['SUPERCATEGORY'] = supercategory
    else:
        logger.warning(f'No category column in {labels_shp}. A unique category "{category}" will be assigned')
        labels_4326_gdf['CATEGORY'] = category
        labels_4326_gdf['SUPERCATEGORY'] = supercategory

    label_filepath = os.path.join(output_dir, f'{prefix if prefix.endswith("_") or prefix==""else prefix + "_"}labels.gpkg')
    labels_4326_gdf.to_file(label_filepath)
    written_files = [label_filepath]
    logger.success(f"Done! A file was written: {label_filepath}")

    return labels_4326_gdf, written_files


def read_dets_and_aoi(detection_files_dict):
    """
    Load split AoI tiles and detections as GeoPandas DataFrames.

    Args:
    - detection_files_dict (dict): a dictionary with keys as dataset names and values as file paths to GeoJSON files containing detections.

    Returns:
    - tiles_gdf (GeoPandas GeoDataFrame): a GeoDataFrame with split AoI tiles. Columns are 'id', 'geometry', and 'year_tile' (if 'year_tile' is present in the input file).
    - detections_gdf (GeoPandas GeoDataFrame): a GeoDataFrame with detections. Columns are 'det_id', 'geometry', 'dataset', 'area', and 'year_det' (if 'year_det' is present in the input file).
    """

    logger.info("Loading split AoI tiles as a GeoPandas DataFrame...")
    tiles_gdf = gpd.read_file('split_aoi_tiles.geojson')
    tiles_gdf = tiles_gdf.to_crs(2056)
    if 'year_tile' in tiles_gdf.keys(): 
        tiles_gdf['year_tile'] = tiles_gdf.year_tile.astype(int)
    logger.success(f"{DONE_MSG} {len(tiles_gdf)} features were found.")

    logger.info("Loading detections as a GeoPandas DataFrame...")

    detections_gdf = gpd.GeoDataFrame()

    for dataset, dets_file in detection_files_dict.items():
        detections_ds_gdf = gpd.read_file(dets_file)    
        detections_ds_gdf[f'dataset'] = dataset
        detections_gdf = pd.concat([detections_gdf, detections_ds_gdf], axis=0, ignore_index=True)
    detections_gdf = detections_gdf.to_crs(2056)
    detections_gdf['area'] = detections_gdf.area 
    detections_gdf['det_id'] = detections_gdf.index
    if 'year_det' in detections_gdf.keys():
        if not detections_gdf['year_det'].all(): 
            detections_gdf['year_det'] = detections_gdf.year_det.astype(int)
    logger.success(f"{DONE_MSG} {len(detections_gdf)} features were found.")

    return tiles_gdf, detections_gdf


def save_name_correspondence(features_list, output_dir, initial_name_column, new_name_column):
    """
    Create a file to keep track of the tile names through the transformations
    If a file of name correspondences already exists in the output folder, the names for the converted tiles will be added. 

    Args:
        features_list (list): A list of features containing the initial name and new name.
        output_dir (str): The directory where the name correspondence file will be saved.
        initial_name_column (str): The name of the column containing the initial name.
        new_name_column (str): The name of the column containing the new name.

    Returns:
        None
    """

    name_correspondence_df = pd.DataFrame.from_records(features_list, columns=[initial_name_column, new_name_column])
    filepath = os.path.join(output_dir, 'name_correspondence.csv')

    if os.path.isfile(filepath):
        logger.warning("A file of name correspondences already exists in the output folder. The names of the converted tiles will be added.")
        existing_df = pd.read_csv(filepath)

        if len(existing_df.columns) > 2:
            existing_df = existing_df[['original_name', 'rgb_name']].drop_duplicates(['original_name', 'rgb_name'])

        if new_name_column in existing_df.columns:
            # Check that the table in not a duplicate due to OVERWRITE = True
            if name_correspondence_df[new_name_column].isin(existing_df[new_name_column]).all():
                return
            elif initial_name_column in existing_df.columns:
                name_correspondence_df = pd.concat([
                    existing_df, 
                    name_correspondence_df[~name_correspondence_df[new_name_column].isin(existing_df[new_name_column])]
                ], ignore_index=True)
        else:
            name_correspondence_df = pd.merge(existing_df, name_correspondence_df, on=initial_name_column, how='left')

    name_correspondence_df.to_csv(filepath, index=False)
    logger.success(f'The correspondence of tile names between the tranformations was saved in {filepath}.')