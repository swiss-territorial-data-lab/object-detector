#!/bin/python
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import geopandas as gpd
import json
import pandas as pd
from loguru import logger

from shapely.affinity import scale
from rasterio.transform import from_bounds


class BadFileExtensionException(Exception):
    "Raised when the file extension is different from the expected one"
    pass


def bounds_to_bbox(bounds):
    
    xmin = bounds[0]
    ymin = bounds[1]
    xmax = bounds[2]
    ymax = bounds[3]
    
    bbox = f"{xmin},{ymin},{xmax},{ymax}"
    
    return bbox


def clip_labels(labels_gdf, tiles_gdf, fact=0.99):

    tiles_gdf['tile_geometry'] = tiles_gdf['geometry']
        
    assert(labels_gdf.crs == tiles_gdf.crs)
    
    labels_tiles_sjoined_gdf = gpd.sjoin(labels_gdf, tiles_gdf, how='inner', predicate='intersects')
    
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
        
        z_x_y = filename.split('.')[0]
        z, x, y = z_x_y.split('_')
        
        return f'({x}, {y}, {z})'


def make_hard_link(row):

        if not os.path.isfile(row.img_file):
            raise FileNotFoundError(row.img_file)

        src_file = row.img_file
        dst_file = src_file.replace('all', row.dataset)

        dirname = os.path.dirname(dst_file)

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        if os.path.exists(dst_file):
            os.remove(dst_file)

        os.link(src_file, dst_file)

        return None


def my_unpack(list_of_tuples):
    # cf. https://www.geeksforgeeks.org/python-convert-list-of-tuples-into-list/
    
    return [item for t in list_of_tuples for item in t]


def scale_point(x, y, xmin, ymin, xmax, ymax, width, height):

    return (x-xmin)/(xmax-xmin)*(width), (ymax-y)/(ymax-ymin)*(height)


def scale_polygon(shapely_polygon, xmin, ymin, xmax, ymax, width, height):
    
    xx, yy = shapely_polygon.exterior.coords.xy

    scaled_polygon = [scale_point(x, y, xmin, ymin, xmax, ymax, width, height) for x, y in zip(xx, yy)]
    
    return scaled_polygon


logger = format_logger(logger)