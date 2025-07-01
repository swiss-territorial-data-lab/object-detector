import os
import sys
from argparse import ArgumentParser
from tqdm import tqdm
from time import time
from yaml import FullLoader, load

import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio

from helpers.misc import format_logger
from helpers.constants import DONE_MSG

from loguru import logger
logger = format_logger(logger)

tic = time()
logger.info('Starting...')

def row_to_filepath(row):
    x, y, z = row.id.lstrip('(').rstrip(')').split(', ')
    filename = z + '_' + x + '_' + y + '.tif'
    filepath = os.path.join(row.dataset+'-images', filename)

    return filepath

parser = ArgumentParser(description="This script gets the statistics of the bands for all images based on their id indicated in the tile gdf.")
parser.add_argument('config_file', type=str, help='a YAML config file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")
with open(args.config_file) as fp:
        cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

# Define constants -----------------------------------

WORKING_DIRECTORY = cfg['working_directory']
TILES = cfg['tiles']

os.chdir(WORKING_DIRECTORY)
logger.info(f'Working directory: {WORKING_DIRECTORY}')

# Import data -----------------------------------
logger.info('Importing data...')

tiles_gdf = gpd.read_file(TILES)

# Get number of bands
first_tile = tiles_gdf.loc[0, :]
filepath = row_to_filepath(first_tile)
with rasterio.open(filepath, 'r') as src:
    num_bands = src.meta["count"]

# Initialize dict and lists
tile_stats = {}
for band in range(1, num_bands+1):
    tile_stats['mean_'+str(band)] = []
    tile_stats['std_'+str(band)] = []

for tile_row in tqdm(tiles_gdf.itertuples(), desc='Calculating the stats', total=tiles_gdf.shape[0]):
    # Get the tile filepath
    filepath = row_to_filepath(tile_row)

    # Get the tile
    with rasterio.open(filepath, "r") as src:
        tile_img = src.read()

    im_num_bands=tile_img.shape[0]
    assert im_num_bands == num_bands, f"The tile {filepath} does not have the expected number of bands ({im_num_bands} instead of {num_bands})."

    for band in range(1, im_num_bands+1):
        tile_stats['mean_'+str(band)].append(round(np.mean(tile_img[band-1]),3))
        tile_stats['std_'+str(band)].append(round(np.std(tile_img[band-1]),3))

filename='tiles_stats.csv'
tile_stats_df = pd.DataFrame(tile_stats)
tile_stats_df.to_csv('tiles_stats.csv', index=False)

for band in range(1, im_num_bands+1):
    print(f"For band {band}, the median of the means is {round(tile_stats_df['mean_'+str(band)].median(),2)}",
        f"and the median of the standard deviations is {round(tile_stats_df['std_'+str(band)].median(),2)}.")  

logger.info(f"Written file: {filename}")

toc = time()
logger.info(f"{DONE_MSG} Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()