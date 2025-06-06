import os
import sys
from loguru import logger
from time import time

import pandas as pd

sys.path.insert(1,'scripts')
from data_preparation import format_labels,  get_delimitation_tiles, pct_to_rgb, tiles_to_box
import fct_misc as misc

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

CONVERT_IMAGES = cfg['convert_images']
TILE_SUFFIX = cfg['tile_suffix']  if 'tile_suffix' in cfg.keys() else '.tif'

OVERWRITE = False
if 'initial_files' in cfg.keys():
    TRAINING = True
    OUTPUT_DIR_CLIPPED_TILES = cfg['output_dir']['clipped_tiles']
    BORDER_POINTS = cfg['initial_files']['border_points']
    BBOX = cfg['initial_files']['bbox']
    PLAN_SCALES = cfg['initial_files']['plan_scales']
else:
    TRAINING = False
    written_files = []
    OUTPUT_DIR_CLIPPED_TILES = TILE_DIR

os.chdir(WORKING_DIR)

if CONVERT_IMAGES:
    pct_to_rgb.main(INITIAL_IMAGE_DIR, TILE_DIR, tile_suffix=TILE_SUFFIX, overwrite=OVERWRITE)

if TRAINING:
    pts_gdf, written_files = format_labels.main(BORDER_POINTS, OUTPUT_DIR_VECT)

    logger.info('Clip tiles to the digitization bounding boxes...')
    tiles_to_box.main(TILE_DIR, BBOX, OUTPUT_DIR_CLIPPED_TILES)

tiles_gdf, _, subtiles_gdf, tmp_written_files = get_delimitation_tiles.main(
    tile_dir=OUTPUT_DIR_CLIPPED_TILES, output_dir=OUTPUT_DIR_VECT, subtiles=True, overwrite=OVERWRITE
)
written_files.extend(tmp_written_files)

if TRAINING:
    output_path_tiles = os.path.join(OUTPUT_DIR_VECT, 'tiles.gpkg')
    if len(tiles_gdf['scale'].unique()) == 1:
        logger.info('Correct scale info on tiles...')
        tile_columns = tiles_gdf.columns
        tiles_gdf.drop(columns='scale', inplace=True)

        name_correspondence_df = pd.read_csv(os.path.join(TILE_DIR, 'name_correspondence.csv'))
        name_correspondence_df.drop_duplicates(subset=name_correspondence_df.columns, inplace=True)
        scales_df = pd.read_excel(PLAN_SCALES)
        scales_df.loc[:, 'Num_plan'] = scales_df.Num_plan.astype(str)
        tmp_gdf = pd.merge(tiles_gdf, name_correspondence_df, left_on='name', right_on='bbox_name')
        tmp_gdf = pd.merge(tmp_gdf, scales_df, left_on='original_name', right_on='Num_plan').rename(columns={'Echelle': 'scale'})
        assert len(tmp_gdf) == len(tiles_gdf), "The number of rows changed when determining the bbox scales."
        tiles_gdf = tmp_gdf[tile_columns].copy()

        tiles_gdf.to_file(os.path.join(OUTPUT_DIR_VECT, 'tiles.gpkg'))

logger.info('Clip images to subtiles...')
SUBTILE_DIR = os.path.join(OUTPUT_DIR_CLIPPED_TILES, 'subtiles')
os.makedirs(SUBTILE_DIR, exist_ok=True)
tiles_to_box.main(OUTPUT_DIR_CLIPPED_TILES, subtiles_gdf, SUBTILE_DIR, overwrite=OVERWRITE)

print()
logger.success("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.success(written_file)

# Stop chronometer
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()