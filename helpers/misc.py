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

try:
    try:
        from helpers.metrics import intersection_over_union
    except ModuleNotFoundError:
        from metrics import intersection_over_union
except Exception as e:
    logger.error(f"Could not import some dependencies. Exception: {e}")
    sys.exit(1)


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


def assign_groups(row, group_index):
    """Assign a group number to GT and detection of a geodataframe

    Args:
        row (row): geodataframe row

    Returns:
        row (row): row with a new 'group_id' column
    """

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

    return clipped_labels_gdf


def format_logger(logger):
    """
    Configures the logger to format log messages with specific styles and colors based on their severity level.

    Args:
        logger (loguru.logger): The logger instance to be formatted.

    Returns:
        loguru.logger: The configured logger instance with custom formatting.

    """

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
    """
    Ensures that the CATEGORY and SUPERCATEGORY columns are present in the input DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the GT labels.

    Returns:
        pandas.DataFrame: The input DataFrame with the CATEGORY and SUPERCATEGORY columns properly renamed.
    """

    if 'category' in df.columns:
        df.rename(columns={'category': 'CATEGORY'}, inplace = True)
    elif 'CATEGORY' not in df.columns:
        logger.critical('The labels have no category. Please produce a CATEGORY column when preparing the data.')
        sys.exit(1)

    if 'supercategory' in df.columns:
        df.rename(columns={'supercategory': 'SUPERCATEGORY'}, inplace = True)
    elif 'SUPERCATEGORY' not in df.columns:
        logger.warning('The labels have no supercategory. A standard "foo" supercategory will be assigned.')
        df['SUPERCATEGORY'] = 'foo'
    
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
        out = pgh.encode(latitude=row.geometry.y, longitude=row.geometry.x, precision=12)
    elif row.geometry.geom_type == 'Polygon':
        out = pgh.encode(latitude=row.geometry.centroid.y, longitude=row.geometry.centroid.x, precision=12)
    else:
        logger.error(f"{row.geometry.geom_type} type is not handled (only Point or Polygon geometry type)")
        sys.exit()

    return out


def get_number_of_classes(coco_files_dict):
    """Read the number of classes from the tileset COCO file.

    Args:
        coco_files_dict (dict): COCO file of the tileset

    Returns:
        num_classes (int): number of classes in the dataset
    """

    file_content = open(next(iter(coco_files_dict.values())))
    coco_json = json.load(file_content)
    num_classes = len(coco_json["categories"])
    file_content.close()
    if num_classes == 0:
        logger.critical('No defined class in the 1st COCO file.')
        sys.exit(0)

    logger.info(f"Working with {num_classes} class{'es' if num_classes > 1 else ''}.")

    return num_classes


def intersect_labels_with_aoi(aoi_tiles_gdf, labels_gdf):
    """Check the crs of the two GDF and perform an inner sjoin.

    Args:
        aoi_tiles_gdf (GeoDataFrame): tiles of the area of interest
        labels_gdf (GeoDataFrame): labels

    Returns:
        tuple: 
            - aoi_tiles_intersecting_labels (GeoDataFrame): tiles of the area of interest intersecting the labels
            - id_list_tiles (list): id of the tiles intersecting a label
    """

    assert( aoi_tiles_gdf.crs == labels_gdf.crs )
    _aoi_tiles_gdf = aoi_tiles_gdf.copy()
    _labels_gdf = labels_gdf.copy()
    aoi_tiles_intersecting_labels = gpd.sjoin(_aoi_tiles_gdf, _labels_gdf, how='inner', predicate='intersects')
    aoi_tiles_intersecting_labels = aoi_tiles_intersecting_labels[_aoi_tiles_gdf.columns]
    aoi_tiles_intersecting_labels.drop_duplicates(inplace=True)
    id_list_tiles = aoi_tiles_intersecting_labels.id.to_numpy().tolist()

    return aoi_tiles_intersecting_labels, id_list_tiles


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


def none_if_undefined(cfg, key):
    
    return cfg[key] if key in cfg.keys() else None


def remove_overlap_poly(gdf_temp, id_to_keep):

    gdf_temp = gpd.sjoin(gdf_temp, gdf_temp,
                        how="inner",
                        predicate="intersects",
                        lsuffix="left",
                        rsuffix="right")
                
    # Remove geometries that intersect themselves
    gdf_temp = gdf_temp[gdf_temp.index != gdf_temp.index_right].copy()

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
        group_index = {node: i for i, group in enumerate(groups) for node in group}
        gdf_temp = gdf_temp.apply(lambda row: assign_groups(row, group_index), axis=1)
        # Find the polygon in the group with the highest detection score
        for id in gdf_temp.group_id.unique():
            gdf_temp2 = gdf_temp[gdf_temp['group_id']==id].copy()
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