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


class UnsupportedImageFormatException(Exception):
    "Raised when the detected image format is not supported"
    pass


def detect_img_format(url):

    lower_url = url.lower()

    if '.png' in lower_url:
        return 'png'
    elif any(x in lower_url for x in ['.jpg', '.jpeg']):
        return 'jpg'
    elif any(x in lower_url for x in ['.tif', '.tiff']):
        return 'tif'
    else:
        return None


def get_geotiff(xyz_url, bbox, year, xyz, filename, save_metadata=False, overwrite=True):
    """ Download tile url formatting and addition of image metadata

    Args:
        xyz_url (path): path to the original image tile
        bbox (tuple): coordinates of the bounding box
        year (int): year of the image tile
        xyz (tuple): x, y, z coordinates of the tile
        save_metadata (bool, optional): Whether to save the metadata in a json file. Defaults to False.
        overwrite (bool, optional): Whether to overwrite the files already existing in the target folder or to skip them. Defaults to True.

    Returns:
        dictionnary: 
            - key: name of the geotiff file
            - value: image metadata
    """

    if not filename.endswith('.tif'):
        raise BadFileExtensionException("Filename must end with .tif")

    img_format = detect_img_format(xyz_url)
    
    if not img_format:
        raise UnsupportedImageFormatException("Unsupported image format")
    
    img_filename = filename.replace('.tif', f'_.{img_format}')
    wld_filename = filename.replace('.tif', '_.wld') # world file
    md_filename  = filename.replace('.tif', '.json')
    geotiff_filename = filename
    
    if save_metadata:
        if not overwrite and os.path.isfile(geotiff_filename) and os.path.isfile(md_filename):
            return None
    else:
        if not overwrite and os.path.isfile(geotiff_filename):
            return None

    x, y, z = xyz

    xyz_url_completed = xyz_url.replace('{year}', str(year)).replace('{z}', str(z)).replace('{x}', str(x)).replace('{y}', str(y))

    xmin, ymin, xmax, ymax = [float(x) for x in bbox.split(',')]

    r = requests.get(xyz_url_completed, allow_redirects=True)

    if r.status_code == 200:
        
        with open(img_filename, 'wb') as fp:
            fp.write(r.content)

        src_ds = gdal.Open(img_filename)
        width, height = src_ds.RasterXSize, src_ds.RasterYSize
        src_ds = None

        # we can mimick ESRI MapImageLayer's metadata, 
        # at least the section that we need
        image_metadata = {
            **({'year': year} if year else {}),
            "width": width, 
            "height": height, 
            "extent": {
                "xmin": xmin, 
                "ymin": ymin, 
                "xmax": xmax, 
                "ymax": ymax,
                'spatialReference': {
                    'latestWkid': "3857" # <- NOTE: hard-coded
                }
            }
        }

        wld = image_metadata_to_world_file(image_metadata)

        with open(wld_filename, 'w') as fp:
            fp.write(wld)

        if save_metadata:
            with open(md_filename, 'w') as fp:
                json.dump(image_metadata, fp)

        try:
            src_ds = gdal.Open(img_filename)
            # NOTE: EPSG:3857 is hard-coded
            gdal.Translate(geotiff_filename, src_ds, options='-of GTiff -a_srs EPSG:3857')
            src_ds = None
        except Exception as e:
            logger.warning(f"Exception in the 'get_geotiff' function: {e}")

        os.remove(img_filename)
        os.remove(wld_filename)

        return {geotiff_filename: image_metadata}

    else:

        return {}


def get_job_dict(tiles_gdf, xyz_url, img_path, year='None', save_metadata=False, overwrite=True):

    job_dict = {}

    for tile in tqdm(tiles_gdf.itertuples(), total=len(tiles_gdf)):

        if year == 'multi-year': 
            img_filename = os.path.join(img_path, f'{tile.year_tile}_{tile.z}_{tile.x}_{tile.y}.tif')  
        else:
            img_filename = os.path.join(img_path, f'{tile.z}_{tile.x}_{tile.y}.tif')

        bbox = bounds_to_bbox(tile.geometry.bounds)

        job_dict[img_filename] = {
            'xyz_url': xyz_url, 
            'bbox': bbox,
            'year': tile.year_tile if 'year_tile' in tiles_gdf.keys() and str(year).isnumeric()==False else year,
            'xyz': (tile.x, tile.y, tile.z),
            'filename': img_filename,
            'save_metadata': save_metadata,
            'overwrite': overwrite
        }

    return job_dict
    

if __name__ == '__main__':

    print("Testing using TiTiler's XYZ...")

    QUERY_STR = "url=/data/mosaic.json&bidx=2&bidx=3&bidx=4&bidx=1&no_data=0&return_mask=false&pixel_selection=lowest"

    #ROOT_URL = f"https://titiler.vm-gpu-01.stdl.ch/mosaicjson/tiles/{{z}}/{{x}}/{{y}}.jpg?{QUERY_STR}"
    ROOT_URL = f"https://titiler.vm-gpu-01.stdl.ch/mosaicjson/tiles/{{z}}/{{x}}/{{y}}.png?{QUERY_STR}"
    ROOT_URL = f"https://titiler.vm-gpu-01.stdl.ch/mosaicjson/tiles/{{z}}/{{x}}/{{y}}.tif?{QUERY_STR}"
    BBOX = "860986.68660422,5925092.68455372,861139.56066079,5925245.55861029"
    xyz= (136704, 92313, 18)
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
       filename=out_filename,
       save_metadata=True
    )

    if outcome != {}:
        print(f'...done. An image was generated: {out_filename}')
    else:
        print("An error occurred.")