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

    gdf1_has_year = 'year' in gdf1.keys()
    gdf2_has_year = 'year' in gdf2.keys()
    param_gives_year = year != None

    if gdf1_has_year or (gdf2_has_year and param_gives_year):   # year for label or double year info oth
        if ds == 'FP' and not (gdf1_has_year and gdf2_has_year):   
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


def format_all_tiles(fp_labels_shp, fp_filepath, ept_labels_shp, ept_data_type, ept_year, labels_gdf, category, supercategory, zoom_level):
    written_files = []

    # Add FP labels if it exists
    if fp_labels_shp:
        fp_labels_gdf = gpd.read_file(fp_labels_shp)
        assert_year(fp_labels_gdf, labels_gdf, 'FP', ept_year) 
        if 'year' in fp_labels_gdf.keys():
            fp_labels_gdf['year'] = fp_labels_gdf.year.astype(int)
            fp_labels_4326_gdf = fp_labels_gdf.to_crs(epsg=4326).drop_duplicates(subset=['geometry', 'year'])
        else:
            fp_labels_4326_gdf = fp_labels_gdf.to_crs(epsg=4326).drop_duplicates(subset=['geometry'])
        
        fp_labels_4326_gdf['CATEGORY'] = fp_labels_4326_gdf[category] if category in fp_labels_4326_gdf.columns() else category
        fp_labels_4326_gdf['SUPERSUPERCATOGRY'] = supercategory

        nb_fp_labels = len(fp_labels_4326_gdf)
        logger.info(f"There are {nb_fp_labels} polygons in {fp_labels_shp}")

        fp_labels_4326_gdf.to_file(fp_filepath, driver='GeoJSON')
        written_files.append(fp_filepath)  
        logger.success(f"Done! A file was written: {fp_filepath}")
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
            if 'year' in labels_4326_gdf.keys():
                if isinstance(ept_year, int):
                    empty_tiles_4326_aoi_gdf['year'] = int(ept_year)
                else:
                    empty_tiles_4326_aoi_gdf['year'] = np.random.randint(low=ept_year[0], high=ept_year[1], size=(len(empty_tiles_4326_aoi_gdf)))
            elif ept_labels_shp and ept_year: 
                logger.warning("No year column in the label shapefile. The provided empty tile year will be ignored.")
        elif ept_data_type == 'shp':
            if ept_year:
                logger.warning("A shapefile of selected empty tiles are provided. The year set for the empty tiles in the configuration file will be ignored")
                ept_year = None
            empty_tiles_4326_aoi_gdf = ept_aoi_4326_gdf.copy()

    # Get all the tiles in one gdf 
    if ept_labels_shp:
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
    if len(labels_4326_gdf) > 1:
        tiles_4326_all_gdf.drop_duplicates(['id'], inplace=True)

    nb_tiles = len(tiles_4326_all_gdf)
    logger.info(f"There were {nb_tiles} tiles created")

    # Get the number of tiles intersecting labels
    tiles_4326_gt_gdf = gpd.sjoin(tiles_4326_all_gdf, labels_gdf[['geometry', 'CATEGORY', 'SUPERCATEGORY']], how='inner', predicate='intersects')
    tiles_4326_gt_gdf.drop_duplicates(['id'], inplace=True)
    logger.info(f"- Number of tiles intersecting GT labels = {len(tiles_4326_gt_gdf)}")

    if fp_labels_shp:
        tiles_4326_fp_gdf = gpd.sjoin(tiles_4326_all_gdf, fp_labels_4326_gdf, how='inner', predicate='intersects')
        tiles_4326_fp_gdf.drop_duplicates(['id'], inplace=True)
        logger.info(f"- Number of tiles intersecting FP labels = {len(tiles_4326_fp_gdf)}")

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
    tile_nbr = int(os.path.basename(path).split('_')[0])
    new_name = f"{tile_nbr}_{round(min_x)}_{round(min_y)}.tif"

    return new_name


def preapre_labels(labels_shp, category, supercategory):

    ## Convert datasets shapefiles into geojson format
    logger.info('Convert labels shapefile into GeoJSON format (EPSG:4326)...')
    labels_gdf = gpd.read_file(labels_shp)
    if 'year' in labels_gdf.keys():
        labels_gdf['year'] = labels_gdf.year.astype(int)
        labels_4326_gdf = labels_gdf.to_crs(epsg=4326).drop_duplicates(subset=['geometry', 'year'])
    else:
        labels_4326_gdf = labels_gdf.to_crs(epsg=4326).drop_duplicates(subset=['geometry'])
    nb_labels = len(labels_4326_gdf)
    logger.info(f'There are {nb_labels} polygons in {labels_shp}')

    if category and category in labels_4326_gdf.keys():
        labels_4326_gdf['CATEGORY'] = labels_4326_gdf[category]
        category = labels_4326_gdf['CATEGORY'].unique()
        logger.info(f'Working with {len(category)} class.es: {category}')
        labels_4326_gdf['SUPERCATEGORY'] = supercategory
    else:
        logger.warning(f'No category column in {labels_shp}. A unique category will be assigned')
        labels_4326_gdf['CATEGORY'] = category
        labels_4326_gdf['SUPERCATEGORY'] = supercategory

    gt_labels_4326_gdf = labels_4326_gdf.copy()

    return gt_labels_4326_gdf


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