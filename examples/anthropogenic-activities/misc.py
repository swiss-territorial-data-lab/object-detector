import os
import sys
from loguru import logger
from shapely.validation import make_valid

import json
import pandas as pd
from geopandas import sjoin
from shapely.affinity import scale
        

def check_validity(poly_gdf, correct=False):
    '''
    Test if all the geometry of a dataset are valid. When it is not the case, correct the geometries with a buffer of 0 m
    if correct != False and stop with an error otherwise.

    - poly_gdf: dataframe of geometries to check
    - correct: boolean indicating if the invalid geometries should be corrected with a buffer of 0 m

    return: a dataframe with valid geometries.
    '''

    invalid_condition = ~poly_gdf.is_valid

    if poly_gdf[invalid_condition].shape[0]!=0:
        logger.warning(f"{poly_gdf[invalid_condition].shape[0]} geometries are invalid on {poly_gdf.shape[0]} detections.")
        if correct:
            logger.info("Correction of the invalid geometries with the shapely function 'make_valid'...")
            
            invalid_poly = poly_gdf.loc[invalid_condition, 'geometry']
            try:
                poly_gdf.loc[invalid_condition, 'geometry'] = [
                    make_valid(poly) for poly in invalid_poly
                    ]
     
            except ValueError:
                logger.info('Failed to fix geometries with "make_valid", try with a buffer of 0.')
                poly_gdf.loc[invalid_condition, 'geometry'] = [poly.buffer(0) for poly in invalid_poly] 
        else:
            sys.exit(1)

    return poly_gdf


def clip_objects(objects_gdf, tiles_gdf, fact=0.99):
    """
    Clips the objects in the `objects_gdf` GeoDataFrame to the tiles in the `tiles_gdf` GeoDataFrame.
    
    Args:
        objects_gdf (geopandas.GeoDataFrame): The GeoDataFrame containing the objects to be clipped.
        tiles_gdf (geopandas.GeoDataFrame): The GeoDataFrame containing the tiles to clip the objects to.
        fact (float, optional): The scaling factor to apply to the tiles when clipping the objects. Defaults to 0.99.
    
    Returns:
        geopandas.GeoDataFrame: The clipped objects GeoDataFrame.
        
    Raises:
        AssertionError: If the CRS of `objects_gdf` is not equal to the CRS of `tiles_gdf`.
    """

    tiles_gdf['tile_geometry'] = tiles_gdf['geometry']
        
    assert(objects_gdf.crs == tiles_gdf.crs)
    
    objects_tiles_sjoined_gdf = sjoin(objects_gdf, tiles_gdf, how='inner', predicate='intersects')

    if 'year_label' in objects_gdf.keys():
        objects_tiles_sjoined_gdf = objects_tiles_sjoined_gdf[objects_tiles_sjoined_gdf.year_label == objects_tiles_sjoined_gdf.year_tile]

    if 'year_det' in objects_gdf.keys():
        objects_tiles_sjoined_gdf = objects_tiles_sjoined_gdf[objects_tiles_sjoined_gdf.year_det == objects_tiles_sjoined_gdf.year_tile]
    
    def clip_row(row, fact=fact):
        
        old_geo = row.geometry
        scaled_tile_geo = scale(row.tile_geometry, xfact=fact, yfact=fact)
        new_geo = old_geo.intersection(scaled_tile_geo)
        row['geometry'] = new_geo

        return row

    clipped_objects_gdf = objects_tiles_sjoined_gdf.apply(lambda row: clip_row(row, fact), axis=1)
    clipped_objects_gdf.crs = objects_gdf.crs

    clipped_objects_gdf.drop(columns=['tile_geometry', 'index_right'], inplace=True)
    clipped_objects_gdf.rename(columns={'id': 'tile_id'}, inplace=True)

    return clipped_objects_gdf


def convert_crs(gdf, epsg=2056):
    """Convert crs of a vector layer to a defined one

    Args:
        gdf (GeoDataFrame): input geodataframe
        epsg (int): projected coordinate system

    Returns:
        GeoDataFrame: output geodataframe with the desired epsg
    """

    if gdf.crs == None:
        gdf = gdf.set_crs(epsg)
        logger.info(f"Set crs to epsg:{epsg}.")
    elif gdf.crs != epsg:
        gdf = gdf.to_crs(epsg)
        logger.info(f"Convert crs to epsg:{epsg}.")

    return gdf
        

def format_logger(logger):
    """Format the logger from loguru

    Args:
        logger: logger object from loguru

    Returns:
        logger: formatted logger object
    """

    logger.remove()
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
            level="INFO", filter=lambda record: record["level"].no < 25)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <green>{level}</green> - {message}",
            level="SUCCESS", filter=lambda record: record["level"].no < 30)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <yellow>{level}</yellow> - {message}",
            level="WARNING", filter=lambda record: record["level"].no < 40)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <red>{level}</red> - <level>{message}</level>",
            level="ERROR")

    return logger


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
    categories_info_df.rename(columns={'name':'CATEGORY', 'id': 'label_class'},inplace=True)
    id_classes = range(len(categories_json))

    return categories_info_df, id_classes


def overlap(polygon1_shape, polygon2_shape):
    """Determine the overlap area of one polygon with another one

    Args:
        polygon1_shape (geometry): first polygon
        polygon2_shape (geometry): second polygon

    Returns:
        float: ratio of overlapped area
    """

    # Calculate intersection area
    
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_area = polygon1_shape.area
    
    return polygon_intersection / polygon_area