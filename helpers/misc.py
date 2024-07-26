#!/bin/python
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import geopandas as gpd
import json
import pygeohash as pgh
import networkx as nx
from loguru import logger

from shapely.affinity import scale
from shapely.validation import make_valid
from rasterio.transform import from_bounds


class BadFileExtensionException(Exception):
    "Raised when the file extension is different from the expected one"
    pass


def add_geohash(gdf, prefix=None, suffix=None):
    """Add geohash column to a geodaframe.

    Args:
        gdf: geodaframe
        prefix (string): custom geohash string with a chosen prefix 
        suffix (string): custom geohash string with a chosen suffix

    Returns:
        out (gdf): geodataframe with geohash column
    """

    out_gdf = gdf.copy()
    out_gdf['geohash'] = gdf.to_crs(epsg=4326).apply(geohash, axis=1)

    if prefix is not None:
        out_gdf['geohash'] = prefix + out_gdf['geohash'].astype(str)

    if suffix is not None:
        out_gdf['geohash'] = out_gdf['geohash'].astype(str) + suffix

    return out_gdf


def assign_groups(row, groups):
    """Assign a group number to GT and detection of a geodataframe

    Args:
        row (row): geodataframe row

    Returns:
        row (row): row with a new 'group_id' column
    """

    group_index = {node: i for i, group in enumerate(groups) for node in group}

    try:
        row['group_id'] = group_index[row['geohash_left']]
    except: 
        row['group_id'] = None
    
    return row
    

def bounds_to_bbox(bounds):
    
    xmin = bounds[0]
    ymin = bounds[1]
    xmax = bounds[2]
    ymax = bounds[3]
    
    bbox = f"{xmin},{ymin},{xmax},{ymax}"
    
    return bbox


def check_validity(poly_gdf, correct=False):
    '''
    Test if all the geometry of a dataset are valid. When it is not the case, correct the geometries with a buffer of 0 m
    if correct != False and stop with an error otherwise.

    - poly_gdf: dataframe of geometries to check
    - correct: boolean indicating if the invalid geometries should be corrected with a buffer of 0 m

    return: a dataframe with valid geometries.
    '''

    invalid_condition = ~poly_gdf.is_valid

    try:
        assert(poly_gdf[invalid_condition].shape[0]==0), \
            f"{poly_gdf[invalid_condition].shape[0]} geometries are invalid on" + \
                    f" {poly_gdf.shape[0]} detections."
    except Exception as e:
        logger.warning(e)
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


def clip_labels(labels_gdf, tiles_gdf, fact=0.99):

    tiles_gdf['tile_geometry'] = tiles_gdf['geometry']
        
    assert(labels_gdf.crs == tiles_gdf.crs)
    
    labels_tiles_sjoined_gdf = gpd.sjoin(labels_gdf, tiles_gdf, how='inner', predicate='intersects')

    if 'year_label' in labels_gdf.keys():
        labels_tiles_sjoined_gdf = labels_tiles_sjoined_gdf[labels_tiles_sjoined_gdf.year_label == labels_tiles_sjoined_gdf.year_tile]  

    def clip_row(row, fact=fact):
        old_geo = row.geometry
        scaled_tile_geo = scale(row.tile_geometry, xfact=fact, yfact=fact)
        new_geo = old_geo.intersection(scaled_tile_geo)
        row['geometry'] = new_geo

        return row

    clipped_labels_gdf = labels_tiles_sjoined_gdf.apply(lambda row: clip_row(row, fact), axis=1)
    clipped_labels_gdf.crs = labels_gdf.crs

    clipped_labels_gdf.drop(columns=['tile_geometry', 'index_right'], inplace=True)
    clipped_labels_gdf.rename(columns={'id': 'tile_id'}, inplace=True)
    tiles_gdf.drop('tile_geometry', inplace=True, axis=1)

    return clipped_labels_gdf


def intersection_over_union(polygon1_shape, polygon2_shape):
    """Determine the intersection area over union area (IoU) of two polygons

    Args:
        polygon1_shape (geometry): first polygon
        polygon2_shape (geometry): second polygon

    Returns:
        int: Unrounded ratio between the intersection and union area
    """

    # Calculate intersection and union, and the IoU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection

    if polygon_union != 0:
        iou = polygon_intersection / polygon_union
    else:
        iou = 0

    return iou


def format_logger(logger):

    logger.remove()
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
            level="INFO", filter=lambda record: record["level"].no < 25)
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - <green>{level}</green> - {message}",
            level="SUCCESS", filter=lambda record: record["level"].no < 30)
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - <yellow>{level}</yellow> - {message}",
            level="WARNING", filter=lambda record: record["level"].no < 40)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <red>{level}</red> - <level>{message}</level>",
            level="ERROR")
    
    return logger


def find_category(df):

    if 'category' in df.columns:
        df.rename(columns={'category': 'CATEGORY'}, inplace = True)
    elif 'CATEGORY' not in df.columns:
        logger.critical('The GT labels have no category. Please produce a CATEGORY column when preparing the data.')
        sys.exit(1)

    if 'supercategory' in df.columns:
        df.rename(columns={'supercategory': 'SUPERCATEGORY'}, inplace = True)
    elif 'SUPERCATEGORY' not in df.columns:
        logger.critical('The GT labels have no supercategory. Please produce a SUPERCATEGORY column when preparing the data.')
        sys.exit(1)
    
    return df


def geohash(row):
    """Geohash encoding (https://en.wikipedia.org/wiki/Geohash) of a location (point).
    If geometry type is a point then (x, y) coordinates of the point are considered. 
    If geometry type is a polygon then (x, y) coordinates of the polygon centroid are considered. 
    Other geometries are not handled at the moment    

    Args:
        row: geodaframe row

    Raises:
        Error: geometry error

    Returns:
        out (str): geohash code for a given geometry
    """
    
    if row.geometry.geom_type == 'Point':
        out = pgh.encode(latitude=row.geometry.y, longitude=row.geometry.x, precision=16)
    elif row.geometry.geom_type == 'Polygon':
        out = pgh.encode(latitude=row.geometry.centroid.y, longitude=row.geometry.centroid.x, precision=16)
    else:
        logger.error(f"{row.geometry.geom_type} type is not handled (only Point or Polygon geometry type)")
        sys.exit()

    return out


def get_number_of_classes(coco_files_dict):

    # get the number of classes
    classes = {"file":[coco_files_dict['trn'], coco_files_dict['tst'], coco_files_dict['val']], "num_classes":[]}

    for filepath in classes["file"]:
        file_content = open(filepath)
        coco_json = json.load(file_content)
        classes["num_classes"].append(len(coco_json["categories"]))
        file_content.close()

    # test if it is the same number of classes in all datasets
    try:
        assert classes["num_classes"][0]==classes["num_classes"][1] and classes["num_classes"][0]==classes["num_classes"][2]
    except AssertionError:
        logger.critical(f"The number of classes is not equal in the training ({classes['num_classes'][0]}), testing ({classes['num_classes'][1]}), ",
                    f"and validation ({classes['num_classes'][2]}) datasets.")
        sys.exit(1)

   # set the number of classes to detect 
    num_classes = classes["num_classes"][0]
    logger.info(f"Working with {num_classes} class{'es' if num_classes > 1 else ''}.")

    return num_classes


def image_metadata_to_affine_transform(image_metadata):
    """
    This uses rasterio.
    cf. https://gdal.org/drivers/raster/wld.html#wld-esri-world-file
    """
    
    xmin = image_metadata['extent']['xmin']
    xmax = image_metadata['extent']['xmax']
    ymin = image_metadata['extent']['ymin']
    ymax = image_metadata['extent']['ymax']
    width  = image_metadata['width']
    height = image_metadata['height']
    
    affine = from_bounds(xmin, ymin, xmax, ymax, width, height)

    return affine


def image_metadata_to_world_file(image_metadata):
    """
    This uses rasterio.
    cf. https://www.perrygeo.com/python-affine-transforms.html
    """
    
    xmin = image_metadata['extent']['xmin']
    xmax = image_metadata['extent']['xmax']
    ymin = image_metadata['extent']['ymin']
    ymax = image_metadata['extent']['ymax']
    width  = image_metadata['width']
    height = image_metadata['height']
    
    affine = from_bounds(xmin, ymin, xmax, ymax, width, height)

    a = affine.a
    b = affine.b
    c = affine.c
    d = affine.d
    e = affine.e
    f = affine.f
    
    c += a/2.0 # <- IMPORTANT
    f += e/2.0 # <- IMPORTANT

    return "\n".join([str(a), str(d), str(b), str(e), str(c), str(f)+"\n"])


def img_md_record_to_tile_id(img_md_record):
    
        filename = os.path.split(img_md_record.img_file)[-1]

        id = filename.split('.')[0]
        
        if len(id.split('_')) == 3:
            z, x, y = id.split('_')

            return f'({x}, {y}, {z})'
    
        elif len(id.split('_')) == 4:
            t, z, x, y = id.split('_')

            return f'({t}, {x}, {y}, {z})'


def make_groups(gdf):
    """Identify groups based on pairing nodes with NetworkX. The Graph is a collection of nodes.
    Nodes are hashable objects (geohash (str)).

    Returns:
        groups (list): list of connected geohash groups
    """

    g = nx.Graph()
    for row in gdf[gdf.geohash_left.notnull()].itertuples():
        g.add_edge(row.geohash_left, row.geohash_right)

    groups = list(nx.connected_components(g))

    return groups


def make_hard_link(img_file, new_img_file):

    if not os.path.isfile(img_file):
        raise FileNotFoundError(img_file)

    src_file = img_file
    dst_file = new_img_file

    if os.path.exists(dst_file):
        os.remove(dst_file)

    os.link(src_file, dst_file)

    return None


def my_unpack(list_of_tuples):
    # cf. https://www.geeksforgeeks.org/python-convert-list-of-tuples-into-list/
    
    return [item for t in list_of_tuples for item in t]


def remove_overlap_poly(gdf_temp, id_to_keep):

    gdf_temp = gpd.sjoin(gdf_temp, gdf_temp,
                        how="inner",
                        predicate="intersects",
                        lsuffix="left",
                        rsuffix="right")
                
    # Remove geometries that intersect themselves
    gdf_temp = gdf_temp[gdf_temp.index != gdf_temp.index_right]

    # Select polygons that overlap
    geom1 = gdf_temp.geom_left.values.tolist()
    geom2 = gdf_temp.geom_right.values.tolist()
    iou = []
    for (i, ii) in zip(geom1, geom2):
        iou.append(intersection_over_union(i, ii))
    gdf_temp['IoU'] = iou
    gdf_temp = gdf_temp[gdf_temp['IoU']>=0.5] 
    gdf_temp['index_left'] = gdf_temp.index

    # Group overlapping polygons
    if len(gdf_temp) > 0:
        groups = make_groups(gdf_temp) 
        gdf_temp = gdf_temp.apply(lambda row: assign_groups(row, groups), axis=1)
        # Find the polygon in the group with the highest detection score
        for id in gdf_temp.group_id.unique():
            gdf_temp2 = gdf_temp[gdf_temp['group_id']==id]
            geohash_max = gdf_temp2[gdf_temp2['score_left']==gdf_temp2['score_left'].max()]['geohash_left'].values[0] 
            id_to_keep.append(geohash_max)

    return id_to_keep
        

def scale_point(x, y, xmin, ymin, xmax, ymax, width, height):

    return (x-xmin)/(xmax-xmin)*(width), (ymax-y)/(ymax-ymin)*(height)


def scale_polygon(shapely_polygon, xmin, ymin, xmax, ymax, width, height):
    
    xx, yy = shapely_polygon.exterior.coords.xy

    scaled_polygon = [scale_point(x, y, xmin, ymin, xmax, ymax, width, height) for x, y in zip(xx, yy)]
    
    return scaled_polygon


logger = format_logger(logger)