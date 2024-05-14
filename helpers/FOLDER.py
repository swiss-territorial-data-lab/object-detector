#!/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import json
import rasterio as rio
from tqdm import tqdm
from loguru import logger

try:
    try:
        from helpers.misc import image_metadata_to_world_file, bounds_to_bbox, format_logger, make_hard_link, BadFileExtensionException
    except ModuleNotFoundError:
        from misc import image_metadata_to_world_file, bounds_to_bbox, format_logger, make_hard_link, BadFileExtensionException
except Exception as e:
    logger.error(f"Could not import some dependencies. Exception: {e}")
    sys.exit(1)


logger = format_logger(logger)



def get_job_dict(tiles_gdf, base_path, end_path='all-images', save_metadata=False, overwrite=True):
    """Make a dictonnary of the necessary parameters to get the tiles from a base folder and place them in the right folder.

    Args:
        tiles_gdf (GeoDataFrame): tiles with the x, y, and z columns deduced from their id
        base_path (path): path to the original folder with the tiles
        end_path (path): path to the target folder used by the object detector. Defaults to 'all-images'.
        save_metadata (bool, optional): Whether to save the metadata in a json file. Defaults to False.
        overwrite (bool, optional): Whether to overwrite files already existing in the target folder or skip them. Defaults to True.

    Returns:
        dictionnary: parameters for the function 'get_image_to_folder' for each image file with the final image path as key.
    """

    job_dict = {}

    for tile in tqdm(tiles_gdf.itertuples(), total=len(tiles_gdf)):

        image_path = os.path.join(end_path, f'{tile.z}_{tile.x}_{tile.y}.tif')
        bbox = bounds_to_bbox(tile.geometry.bounds)

        job_dict[image_path] = {
            'basepath': base_path,
            'filename': image_path,
            'bbox': bbox,
            'save_metadata': save_metadata,
            'overwrite': overwrite
        }

    return job_dict


def get_image_to_folder(basepath, filename, bbox, save_metadata=False, overwrite=True):
    """Copy the image from the original folder to the folder used by object detector.

    Args:
        basepath (path): path to the original image tile
        filename (path): path to the image tile for the object detector
        bbox (tuple): coordinates of the bounding box
        save_metadata (bool, optional): Whether to save the metadata in a json file. Defaults to False.
        overwrite (bool, optional): Whether to overwrite the files already existing in the target folder or to skip them. Defaults to True.

    Raises:
        BadFileExtensionException: The file must be GeoTIFF.

    Returns:
        dictionnary: 
            - key: name of the geotiff file
            - value: image metadata
    """

    if not filename.endswith('.tif'):
        raise BadFileExtensionException("Filename must end with .tif")
    
    basefile = os.path.join(basepath, os.path.basename(filename))
    wld_filename = filename.replace('.tif', '_.wld')    # world file
    md_filename  = filename.replace('.tif', '.json')
    geotiff_filename = filename
    
    dont_overwrite_geotiff = (not overwrite) and os.path.isfile(geotiff_filename)
    if dont_overwrite_geotiff and ((save_metadata and os.path.isfile(md_filename)) or (not save_metadata)):
            return None

    xmin, ymin, xmax, ymax = [float(x) for x in bbox.split(',')]

    with rio.open(basefile) as src:
        image_meta = src.meta.copy()
        width = image_meta['width']
        height = image_meta['height']
        crs = image_meta['crs']

    make_hard_link(basefile, filename)

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
                'latestWkid': str(crs.to_epsg())
            }
        }
    }

    wld = image_metadata_to_world_file(image_metadata)

    with open(wld_filename, 'w') as fp:
        fp.write(wld)

    if save_metadata:
        with open(md_filename, 'w') as fp:
            json.dump(image_metadata, fp)

    os.remove(wld_filename)

    return {geotiff_filename: image_metadata}