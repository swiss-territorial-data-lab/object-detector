import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd

import json
import networkx as nx

sys.path.insert(1, '../..')
from helpers.misc import format_logger
from helpers.metrics import intersection_over_union

logger = format_logger(logger)

# Processing ---------------------------------------

# Start chronometer
tic = time()
logger.info('Starting...')

# Argument and parameter specification
parser = ArgumentParser(description="Post-process detections from the object detector.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

# Load input parameters
WORKING_DIR = cfg['working_dir']
INPUT_DIR = cfg['input_dir']
OUTPUT_DIR = cfg['output_dir']

DETECTIONS = cfg['detections']
SUBTILES = cfg['subtiles']
TILES = cfg['tiles']
CATEGORY_IDS_JSON = cfg['category_ids_json']
SCORE = cfg['score']
KEEP_DATASETS = cfg['keep_datasets']

AREA_THRESHOLDS = {500: (0.1, 4.5), 1000: (0.2, 17), 2000: (1.3, 53), 4000: (5, 190)}

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Processing  ---------------------------------------

logger.info('Read files...')

detections_gdf = gpd.GeoDataFrame()
for dataset_acronym in DETECTIONS.keys():
    dataset = gpd.read_file(os.path.join(INPUT_DIR, DETECTIONS[dataset_acronym]))
    dataset['dataset'] = dataset_acronym
    detections_gdf = pd.concat([detections_gdf, dataset], ignore_index=True)

detections_gdf['det_id'] = detections_gdf.index
det_columns = detections_gdf.columns.tolist()

subtiles_gdf = gpd.read_file(SUBTILES)
tiles_gdf = gpd.read_file(TILES)

filepath = open(os.path.join(INPUT_DIR, CATEGORY_IDS_JSON))
categories_json = json.load(filepath)
filepath.close()

logger.info('Filter dataframe by score and area...')
detections_gdf = detections_gdf[detections_gdf['score'] > SCORE].copy()

# Get scale from subtile info
initial_det_nbr = detections_gdf.shape[0]
subtiles_gdf['xyz'] = [id.strip('()').split(', ') for id in subtiles_gdf.id]
subtiles_gdf['tilename'] = [f"{z}_{x}_{y}.tif" for x, y, z in subtiles_gdf.xyz]
detections_gdf = pd.merge(detections_gdf, subtiles_gdf[['tilename', 'initial_tile']], left_on='image', right_on='tilename')
detections_gdf = pd.merge(detections_gdf, tiles_gdf[['name', 'scale']], left_on='initial_tile', right_on='name')
assert initial_det_nbr == detections_gdf.shape[0], "Some detections disappreard in the join with subtiles and tiles!"

for scale in AREA_THRESHOLDS.keys():
    min_area, max_area = AREA_THRESHOLDS[scale]
    detections_gdf = detections_gdf[~((detections_gdf['scale'] == scale) & ((detections_gdf.area < min_area) | (detections_gdf.area > max_area)))].copy()

logger.info('Find pairs of matching detections across subtiles...')
detections_gdf['original_geom'] = detections_gdf.geometry
detections_gdf['geometry'] = detections_gdf.buffer(0.1)
joined_detections_gdf = gpd.sjoin(detections_gdf, detections_gdf)
# Remove duplicates of the same tuple and self-intersections
joined_detections_gdf = joined_detections_gdf[joined_detections_gdf.det_id_left > joined_detections_gdf.det_id_right].copy()
# Keep pairs about the same object (category) on different images
dets_one_obj_gdf = joined_detections_gdf[
    (joined_detections_gdf.image_right != joined_detections_gdf.image_left)
    & (joined_detections_gdf.det_class_right == joined_detections_gdf.det_class_left)
].copy()

logger.info('Assign a cluster id to each group of detetections...')
# Make a graph of the overlapping detections
graph_dets = nx.Graph()
for det_pair in dets_one_obj_gdf.itertuples():
    graph_dets.add_edge(det_pair.det_id_left, det_pair.det_id_right)

# Attribute a cluster id to each component of the graph
detections_gdf['cluster_id'] = None
cluster_id = 1
for comp in nx.connected_components(graph_dets):
    detections_gdf.loc[detections_gdf.det_id.isin(comp), 'cluster_id'] = cluster_id
    cluster_id += 1

clustered_dets = list(graph_dets)

logger.info('Remove det pair with IoU > 0.75 and not in cluster...')
intersect_detections_gdf  = joined_detections_gdf[joined_detections_gdf.image_right == joined_detections_gdf.image_left].copy()
intersect_detections_gdf['iou'] = [
    intersection_over_union(geom1, geom2) for geom1, geom2 in zip(intersect_detections_gdf.geometry, intersect_detections_gdf.original_geom_right)
]
duplicated_detections_gdf = intersect_detections_gdf[intersect_detections_gdf.iou > 0.75].copy()
duplicated_det_ids = []
if not duplicated_detections_gdf.empty:
    for duplicate in duplicated_detections_gdf.itertuples():
        id_lowest_score = duplicate.det_id_right if duplicate.score_right < duplicate.score_left else duplicate.det_id_left
        if (duplicate.det_id_right in clustered_dets) & (duplicate.det_id_left in clustered_dets):
            det_to_remove = id_lowest_score
        elif duplicate.det_id_right in clustered_dets:
            det_to_remove = duplicate.det_id_left
        elif duplicate.det_id_left in clustered_dets:
            det_to_remove = duplicate.det_id_right
        else:
            det_to_remove = id_lowest_score
        duplicated_det_ids.append(det_to_remove)

logger.info(f'{len(duplicated_det_ids)} detections will be removed because they are duplicates of a same object.')
detections_gdf = detections_gdf[~detections_gdf.det_id.isin(duplicated_det_ids)].copy()

del dets_one_obj_gdf, duplicated_detections_gdf, intersect_detections_gdf, joined_detections_gdf

logger.info('Dissolve dets in clusters....')
clustered_dets_gdf = detections_gdf[~detections_gdf.cluster_id.isnull()].copy()
clustered_dets_gdf.loc[:, 'geometry'] = clustered_dets_gdf.buffer(0.1)
if KEEP_DATASETS:
    dissolved_dets_gdf = clustered_dets_gdf[['dataset', 'score', 'det_id', 'cluster_id', 'det_class', 'initial_tile', 'scale', 'geometry']].dissolve(
        ['cluster_id', 'dataset'], {'score': 'median', 'det_class': 'first', 'initial_tile': 'first', 'scale': 'max', 'det_id': 'first'}, as_index=False
    )
    dissolved_dets_gdf = dissolved_dets_gdf.assign(geometry=dissolved_dets_gdf.buffer(-0.1))
else:
    dissolved_dets_gdf = clustered_dets_gdf[['score', 'cluster_id', 'det_class', 'initial_tile', 'scale', 'geometry']].dissolve(
        'cluster_id', {'score': 'median', 'det_class': 'first', 'initial_tile': 'first', 'scale': 'max'}, as_index=False
    )
    dissolved_dets_gdf = dissolved_dets_gdf.assign(det_id=dissolved_dets_gdf.cluster_id + detections_gdf.det_id.max(), geometry=dissolved_dets_gdf.buffer(-0.1))

detections_gdf = pd.concat(
    [detections_gdf[detections_gdf.cluster_id.isnull()], dissolved_dets_gdf],
    ignore_index=True
)

detections_gdf = detections_gdf[det_columns + ['scale', 'cluster_id', 'initial_tile']].copy()
detections_gdf.drop(columns='image', inplace=True)

logger.info('Get the category name of each detection...')
# get corresponding class ids
categories_info_df = pd.DataFrame()

for key in categories_json.keys():
    categories_tmp={sub_key: [value] for sub_key, value in categories_json[key].items()}
    categories_info_df = pd.concat([categories_info_df, pd.DataFrame(categories_tmp)], ignore_index=True)

# Attribute a category to each row depending on the class id
detections_gdf['det_category'] = [
    categories_info_df.loc[categories_info_df['id']==det_class+1, 'name'].iloc[0]
    for det_class in detections_gdf.det_class.to_numpy()
] 


logger.info('Save file...')
if KEEP_DATASETS:
    filepath = os.path.join(OUTPUT_DIR, 'dst_detected_points.gpkg')
    detections_gdf.to_file(filepath)
else:
    filepath = os.path.join(OUTPUT_DIR, 'detected_points.gpkg')
    detections_gdf.to_file(filepath)

logger.success(f'Done! One file was written: {filepath}.')

# Stop chronometer
toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()