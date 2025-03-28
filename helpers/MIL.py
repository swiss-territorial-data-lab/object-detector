#!/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import json
import requests

from osgeo import gdal
from tqdm import tqdm
from loguru import logger

try:
    try:
        from helpers.misc import image_metadata_to_world_file, bounds_to_bbox, format_logger, BadFileExtensionException
    except ModuleNotFoundError:
        from misc import image_metadata_to_world_file, bounds_to_bbox, format_logger, BadFileExtensionException
except Exception as e:
    logger.error(f"Could not import some dependencies. Exception: {e}")
    sys.exit(1)


logger = format_logger(logger)


def get_geotiff(mil_url, bbox, width, height, filename, image_sr="3857", bbox_sr="3857", save_metadata=False, overwrite=True):
    """
        by default, bbox must be in EPSG:3857
    """

    if not filename.endswith('.tif'):
        raise BadFileExtensionException("Filename must end with .tif")

    png_filename = filename.replace('.tif', '_.png')
    pgw_filename  = filename.replace('.tif', '_.pgw')
    md_filename   = filename.replace('.tif', '.json')
    geotiff_filename = f"{filename}"

    if save_metadata:
        if not overwrite and os.path.isfile(geotiff_filename) and os.path.isfile(geotiff_filename.replace('.tif', '.json')):
            return None
    else:
        if not overwrite and os.path.isfile(geotiff_filename):
            return None

    params = dict(
        bbox=bbox, 
        format='png',
        size=f'{width},{height}',
        f='image',
        imageSR=image_sr,
        bboxSR=bbox_sr,
        transparent=False
    )

    xmin, ymin, xmax, ymax = [float(x) for x in bbox.split(',')]

    image_metadata = {
        "width": width, 
        "height": height, 
        "extent": {
            "xmin": xmin, 
            "ymin": ymin, 
            "xmax": xmax, 
            "ymax": ymax,
            'spatialReference': {
                'latestWkid': bbox_sr
            }
        }
    }
    
    r = requests.post(mil_url + '/export', data=params, timeout=30)

    if r.status_code == 200:

        with open(png_filename, 'wb') as fp:
            fp.write(r.content)

        pgw = image_metadata_to_world_file(image_metadata)

        with open(pgw_filename, 'w') as fp:
            fp.write(pgw)

        if save_metadata:
            with open(md_filename, 'w') as fp:
                json.dump(image_metadata, fp)

        try:
            src_ds = gdal.Open(png_filename)
            gdal.Translate(geotiff_filename, src_ds, options=f'-of GTiff -a_srs EPSG:{image_sr}')
            src_ds = None
        except Exception as e:
            logger.warning(f"Exception in the 'get_geotiff' function: {e}")

        os.remove(png_filename)
        os.remove(pgw_filename)

        return {geotiff_filename: image_metadata}

    else:
       
        return {}


def get_job_dict(tiles_gdf, mil_url, width, height, img_path, image_sr, save_metadata=False, overwrite=True):

    job_dict = {}

    for tile in tqdm(tiles_gdf.itertuples(), total=len(tiles_gdf)):

        img_filename = os.path.join(img_path, f'{tile.z}_{tile.x}_{tile.y}.tif')
        bbox = bounds_to_bbox(tile.geometry.bounds)

        job_dict[img_filename] = {
            'mil_url': mil_url, 
            'bbox': bbox, 
            'width': width, 
            'height': height, 
            'filename': img_filename, 
            'image_sr': image_sr, 
            'bbox_sr': tiles_gdf.crs.to_epsg(),
            'save_metadata': save_metadata,
            'overwrite': overwrite
        }

    return job_dict
    

if __name__ == '__main__':

    print('Doing nothing.')