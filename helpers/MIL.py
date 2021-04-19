#!/bin/python
# -*- coding: utf-8 -*-

import os, sys
import json
import requests
import pyproj
import logging
import logging.config

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

from rasterio.transform import from_bounds
from rasterio import rasterio, features
from osgeo import gdal
from shapely.geometry import box
from shapely.affinity import affine_transform

from helpers.misc import reformat_xyz


logging.config.fileConfig('logging.conf')
logger = logging.getLogger('MIL')


def bounds_to_bbox(bounds):
    
    xmin = bounds[0]
    ymin = bounds[1]
    xmax = bounds[2]
    ymax = bounds[3]
    
    bbox = f"{xmin},{ymin},{xmax},{ymax}"
    
    return bbox


def image_metadata_to_tfw(image_metadata):
    """
    This uses rasterio.
    cf. https://gdal.org/drivers/raster/wld.html#wld-esri-world-file
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


def get_geotiff(mil_url, bbox, width, height, filename, imageSR="2056", bboxSR="2056", save_metadata=False, overwrite=True):
    """
        by default, bbox must be in EPSG:2056
    """

    if not filename.endswith('.tif'):
        raise Exception("Filename must end with .tif")

    tiff_filename = filename.replace('.tif', '_.tif')
    tfw_filename  = filename.replace('.tif', '_.tfw')
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
    
    r = requests.post(mil_url + '/export', data=params, verify=False, timeout=30)

    if r.status_code == 200:
        with open(tiff_filename, 'wb') as fp:
            fp.write(r.content)

        tfw = image_metadata_to_tfw(image_metadata)

        with open(tfw_filename, 'w') as fp:
            fp.write(tfw)

        if save_metadata:
            with open(md_filename, 'w') as fp:
                json.dump(image_metadata, fp)

        try:
            src_ds = gdal.Open(tiff_filename)
            gdal.Translate(geotiff_filename, src_ds, options=f'-of GTiff -a_srs EPSG:{imageSR}')
            src_ds = None
        except Exception as e:
            logger.warning(f"Exception in the 'get_geotiff' function: {e}")

        os.remove(tiff_filename)
        os.remove(tfw_filename)

        return {geotiff_filename: image_metadata}

    else:
       
        return {}

def burn_mask(src_img_filename, dst_img_filename, polys):

    with rasterio.open(src_img_filename) as src:

        src_img = src.read(1)

        if polys == []:
            # TODO: check whether we should replace the following with mask = None
            mask = src_img != -1 # -> everywhere
        else:

            mask = features.geometry_mask(polys, 
                                          out_shape=src.shape, 
                                          transform=src.transform,
                                          all_touched=True)

        shapes = features.shapes(src_img, 
                                 mask=mask, 
                                 transform=src.transform)
        
        profile = src.profile
        profile.update(dtype=rasterio.uint8, count=1)
        
        image = features.rasterize(((g, 255) for g, v in shapes), 
                                   out_shape=src.shape, 
                                   transform=src.transform)
    
    with rasterio.open(dst_img_filename, 'w', **profile) as dst:
        dst.write(image, indexes=1)
    
    return


def get_job_dict(tiles_gdf, mil_url, width, height, img_path, imageSR, save_metadata=False, overwrite=True):

    job_dict = {}

    #print('Computing xyz...')
    gdf = tiles_gdf.apply(reformat_xyz, axis=1)
    gdf.crs = tiles_gdf.crs
    #print('...done.')

    for tile in gdf.itertuples():

        x, y, z = tile.xyz

        img_filename = os.path.join(img_path, f'{z}_{x}_{y}.tif')
        bbox = bounds_to_bbox(tile.geometry.bounds)

        job_dict[img_filename] = {'mil_url': mil_url, 
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