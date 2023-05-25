#!/bin/python
# -*- coding: utf-8 -*-

import os, sys
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


def get_geotiff(wms_url, layers, bbox, width, height, filename, srs="EPSG:3857", save_metadata=False, overwrite=True):
    """
        ...
    """

    if not filename.endswith('.tif'):
        raise BadFileExtensionException("Filename must end with .tif")

    png_filename = filename.replace('.tif', '_.png')
    pgw_filename = filename.replace('.tif', '_.pgw')
    md_filename  = filename.replace('.tif', '.json')
    geotiff_filename = filename
    
    if save_metadata:
        if not overwrite and os.path.isfile(geotiff_filename) and os.path.isfile(geotiff_filename.replace('.tif', '.json')):
            return None
    else:
        if not overwrite and os.path.isfile(geotiff_filename):
            return None

    params = dict(
        service="WMS",
        version="1.1.1",
        request="GetMap",
        layers=layers,
        format="image/png",
        srs=srs,
        transparent=True,
        styles="",
        bbox=bbox,
        width=width,
        height=height
    )

    xmin, ymin, xmax, ymax = [float(x) for x in bbox.split(',')]

    # we can mimick ESRI MapImageLayer's metadata, 
    # at least the section that we need
    image_metadata = {
        "width": width, 
        "height": height, 
        "extent": {
            "xmin": xmin, 
            "ymin": ymin, 
            "xmax": xmax, 
            "ymax": ymax,
            'spatialReference': {
                'latestWkid': srs.split(':')[1]
            }
        }
    }

    r = requests.get(wms_url, params=params, allow_redirects=True)

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
            gdal.Translate(geotiff_filename, src_ds, options=f'-of GTiff -a_srs {srs}')
            src_ds = None
        except Exception as e:
            logger.warning(f"Exception in the 'get_geotiff' function: {e}")

        os.remove(png_filename)
        os.remove(pgw_filename)

        return {geotiff_filename: image_metadata}
        
    else:
        logger.warning(f"Failed to get image from WMS: HTTP Status Code = {r.status_code}, received text = '{r.text}'")
        return {}


def get_job_dict(tiles_gdf, wms_url, layers, width, height, img_path, srs, save_metadata=False, overwrite=True):

    job_dict = {}

    for tile in tqdm(tiles_gdf.itertuples(), total=len(tiles_gdf)):

        img_filename = os.path.join(img_path, f'{tile.z}_{tile.x}_{tile.y}.tif')
        bbox = bounds_to_bbox(tile.geometry.bounds)

        job_dict[img_filename] = {
            'wms_url': wms_url,
            'layers': layers, 
            'bbox': bbox,
            'width': width, 
            'height': height, 
            'filename': img_filename, 
            'srs': srs,
            'save_metadata': save_metadata,
            'overwrite': overwrite
        }

    return job_dict
    

if __name__ == '__main__':

    print("Testing using Neuchâtel Canton's WMS...")

    ROOT_URL = "https://sitn.ne.ch/mapproxy95/service"
    BBOX = "763453.0385123404,5969120.412845984,763605.9125689107,5969273.286902554"
    WIDTH=256
    HEIGHT=256
    LAYERS = "ortho2019"
    SRS="EPSG:900913"
    OUTPUT_IMG = 'test.tif'
    OUTPUT_DIR = 'test_output'
    # let's make the output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    BBOX = "763453.0385123404,5969120.412845984,763605.9125689107,5969273.286902554"

    out_filename = os.path.join(OUTPUT_DIR, OUTPUT_IMG)

    outcome = get_geotiff(
       ROOT_URL,
       LAYERS,
       bbox=BBOX,
       width=WIDTH,
       height=HEIGHT,
       filename=out_filename,
       srs=SRS,
       save_metadata=True
    )

    if outcome != {}:
        print(f'...done. An image was generated: {out_filename}')