#!/bin/python
# -*- coding: utf-8 -*-

import os, sys
import json
import requests
import pyproj
import logging
import logging.config

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('MIL')

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

from rasterio.transform import from_bounds
from rasterio import rasterio, features
from osgeo import gdal
from shapely.geometry import box
from shapely.affinity import affine_transform
from tqdm import tqdm

try:
    try:
        from helpers.misc import reformat_xyz, image_metadata_to_world_file, bounds_to_bbox
    except:
        from misc import reformat_xyz, image_metadata_to_world_file, bounds_to_bbox
except Exception as e:
    logger.error(f"Could not import some dependencies. Exception: {e}")
    sys.exit(1)


def get_geotiff(MIL_url, bbox, width, height, filename, imageSR="2056", bboxSR="2056", save_metadata=False, overwrite=True):
    """
        by default, bbox must be in EPSG:2056
    """

    if not filename.endswith('.tif'):
        raise Exception("Filename must end with .tif")

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
        imageSR=imageSR,
        bboxSR=bboxSR,
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
                'latestWkid': bboxSR
            }
        }
    }

    #params = {'bbox': bbox, 'format': 'tif', 'size': f'{width},{height}', 'f': 'pjson', 'imageSR': imageSR, 'bboxSR': bboxSR}
    
    r = requests.post(MIL_url + '/export', data=params, verify=False, timeout=30)

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
            gdal.Translate(geotiff_filename, src_ds, options=f'-of GTiff -a_srs EPSG:{imageSR}')
            src_ds = None
        except Exception as e:
            logger.warning(f"Exception in the 'get_geotiff' function: {e}")

        os.remove(png_filename)
        os.remove(pgw_filename)

        return {geotiff_filename: image_metadata}

    else:
       
        return {}


def get_job_dict(tiles_gdf, MIL_url, width, height, img_path, imageSR, save_metadata=False, overwrite=True):

    job_dict = {}

    #print('Computing xyz...')
    gdf = tiles_gdf.apply(reformat_xyz, axis=1)
    gdf.crs = tiles_gdf.crs
    #print('...done.')

    for tile in tqdm(gdf.itertuples(), total=len(gdf)):

        x, y, z = tile.xyz

        img_filename = os.path.join(img_path, f'{z}_{x}_{y}.tif')
        bbox = bounds_to_bbox(tile.geometry.bounds)

        job_dict[img_filename] = {'MIL_url': MIL_url, 
                                  'bbox': bbox, 
                                  'width': width, 
                                  'height': height, 
                                  'filename': img_filename, 
                                  'imageSR': imageSR, 
                                  'bboxSR': gdf.crs.to_epsg(),
                                  'save_metadata': save_metadata,
                                  'overwrite': overwrite
        }

    return job_dict
    

if __name__ == '__main__':

    print('Doing nothing.')