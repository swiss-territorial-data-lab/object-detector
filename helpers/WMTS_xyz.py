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

from tqdm import tqdm


from helpers.misc import reformat_xyz

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('WMTS_XYZ')


def bounds_to_bbox(bounds):
    
    xmin = bounds[0]
    ymin = bounds[1]
    xmax = bounds[2]
    ymax = bounds[3]
    
    bbox = f"{xmin},{ymin},{xmax},{ymax}"
    
    return bbox


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


def image_metadata_to_affine_transform(image_metadata):
    """
    This uses rasterio.
    """
    
    xmin = image_metadata['extent']['xmin']
    xmax = image_metadata['extent']['xmax']
    ymin = image_metadata['extent']['ymin']
    ymax = image_metadata['extent']['ymax']
    width  = image_metadata['width']
    height = image_metadata['height']
    
    affine = from_bounds(xmin, ymin, xmax, ymax, width, height)

    return affine


def get_geotiff(WMTS_xyz_url, bbox, xyz, width, height, filename, param, srs="EPSG:3857", save_metadata=False, overwrite=True):
    """
        ...
    """

    if not filename.endswith('.tif'):
        raise Exception("Filename must end with .tif")

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

    x, y, z = xyz
    WMTS_xyz_url_completed=WMTS_xyz_url + '/' + str(z) + '/' + str(x) + '/' + str(y)+ '.tif'


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

    r = requests.get(WMTS_xyz_url_completed, params=param, allow_redirects=True, verify=False)

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
            # print(f'{e}')
            logger.warning(f"Exception in the 'get_geotiff' function: {e}")

        os.remove(png_filename)
        os.remove(pgw_filename)

        return {geotiff_filename: image_metadata}
        
    else:logging.config.fileConfig('logging.conf')
logger = logging.getLogger('WMTS_XYZ')


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


def get_job_dict(tiles_gdf, WMTS_xyz_url, width, height, img_path, param, srs="EPSG:3857", save_metadata=False, overwrite=True):

    job_dict = {}

    #print('Computing xyz...')
    gdf = tiles_gdf.apply(reformat_xyz, axis=1)
    gdf.crs = tiles_gdf.crs
    #print('...done.')

    for tile in tqdm(gdf.itertuples(), total=len(gdf)):

        x, y, z = tile.xyz      # xyz: name of the filed created in the "reformat_xyz" function 

        img_filename = os.path.join(img_path, f'{z}_{x}_{y}.tif')
        bbox = bounds_to_bbox(tile.geometry.bounds)

        job_dict[img_filename] = {
            'WMTS_xyz_url': WMTS_xyz_url, 
            'bbox':bbox,
            'xyz': tile.xyz,
            'width': width, 
            'height': height, 
            'filename': img_filename,
            'param': param,
            'srs': srs,
            'save_metadata': save_metadata,
            'overwrite': overwrite
        }

    return job_dict
    

if __name__ == '__main__':

    print("Testing using Titiler's WMTS...")

    ROOT_URL = "https://titiler.vm-gpu-01.stdl.ch/mosaicjson/tiles"
    PARAMETERS= dict(
        url="/data/mosaic.json",
        no_data=0,
        return_mask="false",
        pixel_selection="lowest"
    )
    BBOX = "860986.68660422,5925092.68455372,861139.56066079,5925245.55861029"
    xyz= [136704, 92313, 18]
    WIDTH=256
    HEIGHT=256
    SRS="EPSG:3857"
    OUTPUT_IMG = 'test.tif'
    OUTPUT_DIR = 'test_output'
    # let's make the output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    out_filename = os.path.join(OUTPUT_DIR, OUTPUT_IMG)

    outcome = get_geotiff(
       ROOT_URL,
       bbox=BBOX,
       xyz=xyz,
       width=WIDTH,
       height=HEIGHT,
       filename=out_filename,
       parameters_dict=PARAMETERS,
       srs=SRS,
       save_metadata=True
    )

    if outcome != {}:
        print(f'...done. An image was generated: {out_filename}')