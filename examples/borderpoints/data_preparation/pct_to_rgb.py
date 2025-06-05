import os
import sys
from loguru import logger
from time import time
from tqdm import tqdm

import numpy as np
import rasterio as rio
from glob import glob
from rasterio.crs import CRS
from rasterio.warp import reproject

sys.path.insert(1, 'scripts')
from fct_misc import format_logger, get_config, save_name_correspondence

logger = format_logger(logger)

# Functions ------------------------------------------


def main(input_dir, output_dir='outputs/rgb_images', nodata_key=255, tile_suffix='.tif', overwrite=False):
    """Convert images with a color palette to RGB images.
    Reproject the images to EPSG:2056 if the tranform or the CRS is not already corresponding to EPSG:2056.

    Args:
        input_dir (str): path to the directory containing the tiles
        output_dir (str, optional): path to the output directory. Defaults to 'outputs/rgb_images'.
        nodata_key (int, optional): number in the color palette correponding to nodata, i.e. to the color definied as (0,0,0,0). Defaults to 255.
        tile_suffix (str, optional): suffix of the filename, which is the part coming after the tile number or id. Defaults to '.tif'.
    """
    
    os.makedirs(output_dir, exist_ok=True)

    tiles_list = glob(os.path.join(input_dir, '**/*.tif'), recursive=True)
    if len(tiles_list) == 0:
        logger.critical('No tile found in the input folder. Please control the path.')
        sys.exit(1)

    name_correspondence_list = []
    tile_nbr = 0
    existing_tiles = glob(os.path.join(output_dir, '*.tif'))
    for tile_path in tqdm(tiles_list, desc='Convert images from colormap to RGB'):
        if tile_suffix in tile_path:
            tile_name = os.path.basename(tile_path).rstrip(tile_suffix)
        else:
            tile_name = os.path.basename(tile_path).rstrip('.tif')

        end_out_path = f"{tile_name[:6]}_{tile_name[6:]}.tif"
        if not overwrite and any(end_out_path in outpath for outpath in existing_tiles):
            continue
        
        while any(os.path.basename(name).startswith(str(tile_nbr) + '_') for name in existing_tiles) and not overwrite:
            tile_nbr += 1
        out_path = os.path.join(output_dir, str(tile_nbr) + '_' + end_out_path)
        
        name_correspondence_list.append((tile_name, (str(tile_nbr) + '_' + end_out_path).rstrip('.tif')))

        with rio.open(tile_path) as src:
            image = src.read()
            meta = src.meta
            colormap = src.colormap(1)

        nodata_value = colormap[nodata_key][0]
        if nodata_value != 0:
            print()
            logger.warning(f'The nodata value for the plan {tile_name} is {nodata_value} and not 0.')
            logger.warning(f'Setting the nodata value to 0.')
            colormap[nodata_key] = (0, 0, 0, 0)
            nodata_value = 0
        converted_image = np.empty((3, meta['height'], meta['width']))
        # Efficient mapping: https://stackoverflow.com/questions/55949809/efficiently-replace-elements-in-array-based-on-dictionary-numpy-python
        mapping_key = np.array(list(colormap.keys()))
        for band_nbr in range(3):
            # Get colormap corresponding to band 
            mapping_values = np.array([mapping_list[band_nbr] for mapping_list in colormap.values()])
            mapping_array = np.zeros(mapping_key.max()+1, dtype=mapping_values.dtype)
            mapping_array[mapping_key] = mapping_values
            
            # Translate colormap into corresponding band
            new_band = mapping_array[image]
            converted_image[band_nbr, :, :] = new_band

        if not meta['crs']:
            print()
            logger.warning(f'No crs for the tile {tile_name}. Setting it to EPSG:2056.')
            meta.update(crs=CRS.from_epsg(2056))
        elif meta['crs'] != CRS.from_epsg(2056):
            print()
            logger.warning(f'Wrong crs for the tile {tile_name}: {meta["crs"]}, it will be reprojected.')

        if (meta['crs'] != CRS.from_epsg(2056)) or (meta['transform'][1] != 0):
            converted_image, new_transform = reproject(
                converted_image, src_transform=meta['transform'],
                src_crs=meta['crs'], dst_crs=CRS.from_epsg(2056)
            )
            meta.update(transform=new_transform, height=converted_image.shape[1], width=converted_image.shape[2])

        meta.update(count=3, nodata=nodata_value, compression='lzw')
        with rio.open(out_path, 'w', **meta) as dst:
            dst.write(converted_image)
        
        tile_nbr += 1

    if len(name_correspondence_list) > 0:
        save_name_correspondence(name_correspondence_list, output_dir, 'original_name', 'rgb_name')
        logger.success(f"Done converting color map images to RGB! The files were written in the folder {output_dir}. Let's check them out!")
    else:
        logger.success(f"All files were already present in folder {output_dir}. Nothing to do.")


# ------------------------------------------

if __name__ == "__main__":

    # Start chronometer
    tic = time()
    logger.info('Starting...')

    cfg = get_config('prepare_data.py', 'The script converts the images from colormap to RGB.')

    WORKING_DIR = cfg['working_dir']
    INPUT_DIR = cfg['initial_image_dir']
    OUTPUT_DIR = cfg['tile_dir']

    PLAN_SCALES = cfg['plan_scales'] if 'plan_scales' in cfg.keys() else None
    NODATA_KEY = cfg['nodata_key'] if 'nodata_key' in cfg.keys() else 255

    TILE_SUFFIX = cfg['tile_suffix'] if 'tile_suffix' in cfg.keys() else '.tif'

    os.chdir(WORKING_DIR)

    main(INPUT_DIR, OUTPUT_DIR, PLAN_SCALES, NODATA_KEY, TILE_SUFFIX)