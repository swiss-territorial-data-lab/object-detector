import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

import geopandas as gpd

sys.path.insert(1,'scripts')
from data_preparation import format_surveying_data, get_delimitation_tiles, pct_to_rgb, tiles_to_box
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Processing ---------------------------------------

# Start chronometer
tic = time()
logger.info('Starting...')

cfg = misc.get_config(os.path.basename(__file__), desc="The script prepares the initial files for the use of the OD in the detection of border points.")

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR_VECT= cfg['output_dir']['vectors']

INITIAL_IMAGE_DIR = cfg['initial_image_dir']
TILE_DIR = cfg['tile_dir']

CADASTRAL_SURVEYING = cfg['cadastral_surveying']
OVERLAP_INFO = cfg['overlap_info'] if 'overlap_info' in cfg.keys() else None
TILE_SUFFIX  = cfg['tile_suffix'] if 'tile_suffix' in cfg.keys() else '.tif'

TILE_SUFFIX = cfg['tile_suffix'] if 'tile_suffix' in cfg.keys() else '.tif'
CONVERT_IMAGES = cfg['convert_images']

os.chdir(WORKING_DIR)

if CONVERT_IMAGES:
    pct_to_rgb.main(INITIAL_IMAGE_DIR, TILE_DIR, tile_suffix=TILE_SUFFIX)

tiles_gdf, _, subtiles_gdf, written_files = get_delimitation_tiles.main(TILE_DIR, 
                                                                                       overlap_info=OVERLAP_INFO, output_dir=OUTPUT_DIR_VECT, subtiles=True)

logger.info('Format cadastral surveying data...')
cs_points_gdf, tmp_written_files = format_surveying_data.main(CADASTRAL_SURVEYING, subtiles_gdf, output_dir=OUTPUT_DIR_VECT)
written_files.extend(tmp_written_files)

logger.info('Limit subtiles to the area with cadastral survey data and overwrite the initial file...')
subtiles_gdf = gpd.sjoin(subtiles_gdf, cs_points_gdf[['pt_id', 'geometry']])
subtiles_gdf.drop(columns='pt_id', inplace=True)
subtiles_gdf.drop_duplicates(subset='id', inplace=True, ignore_index=True)
subtiles_gdf.to_file(os.path.join(OUTPUT_DIR_VECT, 'subtiles.gpkg'))

# Clip images to subtiles
SUBTILE_DIR = os.path.join(TILE_DIR, 'subtiles')
os.makedirs(SUBTILE_DIR, exist_ok=True)
tiles_to_box.main(TILE_DIR, subtiles_gdf, SUBTILE_DIR)

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)

# Stop chronometer
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()