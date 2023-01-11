import os, sys
import logging, logging.config
import yaml, argparse
import time
from tqdm import tqdm

import pandas as pd
import geopandas as gpd
import numpy as np

import rasterio


logging.config.fileConfig('logging.conf')
logger = logging.getLogger('root')

tic = time.time()
logger.info('Starting...')

parser = argparse.ArgumentParser(description="This script trains a predictive models.")
parser.add_argument('config_file', type=str, help='a YAML config file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")
with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

# Define constants -----------------------------------

WORKING_DIRECTORY=cfg['working_folder']
TILES=cfg['tiles']

os.chdir(WORKING_DIRECTORY)

# Import data -----------------------------------
logger.info('Importing data...')

tiles=gpd.read_file(TILES)


for tile_row in tqdm(tiles.itertuples(), desc='Producing the masks', total=tiles.shape[0]):
    tile_id=tile_row.id

    # Get the tile filepath
    x, y, z = tile_id.lstrip('(').rstrip(')').split(', ')
    filename=z+'_'+x+'_'+y+'.tif'
    filepath=os.path.join(tile_row.dataset+'-images', filename)

    # Get the tile
    with rasterio.open(os.path.join(filepath), "r") as src:
        tile_img = src.read()

    im_num_bands=tile_img.shape[0]

    if tile_row.Index==0:
        tile_stats={}
        for band in range(1, im_num_bands+1):
            tile_stats['mean_'+str(band)]=[]
            tile_stats['std_'+str(band)]=[]

    for band in range(1, im_num_bands+1):
        tile_stats['mean_'+str(band)].append(round(np.mean(tile_img[band-1]),3))
        tile_stats['std_'+str(band)].append(round(np.std(tile_img[band-1]),3))

tiles_stats_df=pd.DataFrame(tile_stats)
tiles_stats_df.to_csv('tiles_stats.csv', index=False)

for band in range(1, im_num_bands+1):
    print(f"For band {band}, the median of the means is {round(tiles_stats_df['mean_'+str(band)].median(),2)}",
        f"and the median of the standard deviations is {round(tiles_stats_df['std_'+str(band)].median(),2)}.")    