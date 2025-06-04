import os
import sys
from argparse import ArgumentParser
from loguru import logger
from time import time
from yaml import load, FullLoader

import geopandas as gpd
import pandas as pd

import json
from sklearn.metrics import confusion_matrix

sys.path.insert(1, 'scripts')
import functions.fct_metrics as metrics
import functions.fct_misc as misc

logger = misc.format_logger(logger)

# Parameter definition ---------------------------------------

# Start chronometer
tic = time()
logger.info('Starting...')

# Argument and parameter specification
parser = ArgumentParser(description="The script prepares the initial files for the use of the OD in the detection of border points.")
parser.add_argument('config_file', type=str, help='Framework configuration file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")

with open(args.config_file) as fp:
    cfg = load(fp, Loader=FullLoader)[os.path.basename(__file__)]

# Load input parameters
WORKING_DIR = cfg['working_dir']
OUTPUT_DIR = cfg['output_dir']

SUBTILES = cfg['subtiles']
DETECTIONS = cfg['detections']
GROUND_TRUTH = cfg['ground_truth']
CATEGORY_IDS_JSON = cfg['category_ids_json']
NAME_CORRESPONDENCE = cfg['name_correspondence']

KEEP_DATASETS = cfg['keep_datasets']
IOU_THRESHOLD = cfg['iou_threshold'] if 'iou_threshold' in cfg.keys() else 0.25

os.chdir(WORKING_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

written_files = []


# Processing  ---------------------------------------

logger.info('Read files...')
subtiles_gdf = gpd.read_file(SUBTILES)

detections_gdf = gpd.read_file(DETECTIONS)

labels_gdf = gpd.read_file(GROUND_TRUTH)

filepath = open(CATEGORY_IDS_JSON)
categories_json = json.load(filepath)
filepath.close()

name_correspondence_df = pd.read_csv(NAME_CORRESPONDENCE)

logger.info('Format files...')

# Merge tiles according to parameters and put dets in a dict
dets_gdf_dict = {}
if KEEP_DATASETS:
    tiles_gdf = subtiles_gdf.dissolve(['dataset', 'initial_tile'], as_index=False)

    if ('dst' not in DETECTIONS) and ('dataset' not in DETECTIONS):
        logger.warning("There is no indication in the name of the dataset for the detections that the partition was retained.")
        logger.warning("Ensure that the proper file is used.")

    datasets_list = detections_gdf.dataset.unique()
    for dataset in datasets_list:
        dets_gdf_dict[dataset] = detections_gdf[detections_gdf.dataset == dataset].copy()

else:
    tiles_gdf = subtiles_gdf.dissolve(['initial_tile'], as_index=False)

    if ('dst' in DETECTIONS) or ('dataset' in DETECTIONS):
        logger.warning("There is an indication in the name of the dataset for the detections that the partition was retained.")
        logger.warning("Ensure that the proper file is used.")

    datasets_list = ['all']
    dets_gdf_dict['all'] = detections_gdf.copy()
    tiles_gdf.loc[:, 'dataset'] = 'all'

assert(labels_gdf.crs == tiles_gdf.crs)

# Clip the labels to the tiles
tiles_gdf = tiles_gdf.merge(name_correspondence_df, left_on="initial_tile", right_on="bbox_name")

clipped_labels_gdf = gpd.GeoDataFrame()
for tile_name in tiles_gdf.original_name.unique():
    labels_on_tile_gdf = labels_gdf[labels_gdf.Num_plan == tile_name].copy()
    selected_tiles_gdf = tiles_gdf[tiles_gdf.original_name == tile_name].copy()

    tmp_gdf = misc.clip_labels(labels_on_tile_gdf, selected_tiles_gdf, fact=0.999)
    clipped_labels_gdf = pd.concat([clipped_labels_gdf, tmp_gdf], ignore_index=True)

# get classe ids
id_classes = range(len(categories_json))

# append class ids to labels
categories_info_df = pd.DataFrame()

for key in categories_json.keys():

    categories_tmp={sub_key: [value] for sub_key, value in categories_json[key].items()}
    
    categories_info_df = pd.concat([categories_info_df, pd.DataFrame(categories_tmp)], ignore_index=True)

categories_info_df.sort_values(by=['id'], inplace=True, ignore_index=True)
categories_info_df.drop(['supercategory'], axis=1, inplace=True)

categories_info_df.rename(columns={'name':'CATEGORY', 'id': 'label_class'},inplace=True)
clipped_labels_gdf = clipped_labels_gdf.astype({'CATEGORY':'str'})
clipped_labels_w_id_gdf = clipped_labels_gdf.merge(categories_info_df, on='CATEGORY', how='left')


logger.info('Tag detections and get metrics...')

metrics_dict_by_cl = {}
for dataset in datasets_list:
    metrics_dict_by_cl[dataset] = []
metrics_cl_df_dict = {}
tagged_dets_gdf = gpd.GeoDataFrame()

for dataset in datasets_list:

    tmp_gdf = dets_gdf_dict[dataset].copy()
    tmp_gdf.to_crs(epsg=clipped_labels_w_id_gdf.crs.to_epsg(), inplace=True)

    tp_gdf, fp_gdf, fn_gdf, mismatched_class_gdf = metrics.get_fractional_sets(
        tmp_gdf, 
        clipped_labels_w_id_gdf[clipped_labels_w_id_gdf.dataset == dataset],
        IOU_THRESHOLD
    )
    tp_gdf['tag'] = 'TP'
    tp_gdf['dataset'] = dataset
    fp_gdf['tag'] = 'FP'
    fp_gdf['dataset'] = dataset
    fn_gdf['tag'] = 'FN'
    fn_gdf['dataset'] = dataset
    mismatched_class_gdf['tag'] = 'wrong class'
    mismatched_class_gdf['dataset'] = dataset

    tagged_dets_gdf = pd.concat([tagged_dets_gdf, tp_gdf, fp_gdf, fn_gdf, mismatched_class_gdf], ignore_index=True)
    tp_k, fp_k, fn_k, p_k, r_k, precision, recall, f1 = metrics.get_metrics(tp_gdf, fp_gdf, fn_gdf, mismatched_class_gdf, id_classes)
    logger.info(f'Dataset = {dataset} => precision = {precision:.3f}, recall = {recall:.3f}, f1 = {f1:.3f}')

    # label classes starting at 1 and detection classes starting at 0.
    for id_cl in id_classes:
        metrics_dict_by_cl[dataset].append({
            'class': id_cl,
            'precision_k': p_k[id_cl],
            'recall_k': r_k[id_cl],
            'TP_k' : tp_k[id_cl],
            'FP_k' : fp_k[id_cl],
            'FN_k' : fn_k[id_cl],
        })

    metrics_cl_df_dict[dataset] = pd.DataFrame.from_records(metrics_dict_by_cl[dataset])

file_to_write = os.path.join(OUTPUT_DIR, f'{"dst_" if KEEP_DATASETS else ""}tagged_detections.gpkg')
tagged_dets_gdf[['geometry', 'det_id', 'score', 'tag', 'dataset', 'label_class', 'CATEGORY', 'det_class', 'det_category', 'cluster_id']]\
    .to_file(file_to_write, driver='GPKG', index=False)
written_files.append(file_to_write)

# Save the metrics by class for each dataset
metrics_by_cl_df = pd.DataFrame()
for dataset in datasets_list:
    dataset_df = metrics_cl_df_dict[dataset].copy()
    dataset_df['dataset'] = dataset

    metrics_by_cl_df = pd.concat([metrics_by_cl_df, dataset_df], ignore_index=True)

metrics_by_cl_df['category'] = [
    categories_info_df.loc[categories_info_df.label_class==det_class+1, 'CATEGORY'].iloc[0] 
    for det_class in metrics_by_cl_df['class'].to_numpy()
] 

file_to_write = os.path.join(OUTPUT_DIR, f'{"dst_" if KEEP_DATASETS else ""}metrics_by_class.csv')
metrics_by_cl_df[
    ['class', 'category', 'TP_k', 'FP_k', 'FN_k', 'precision_k', 'recall_k', 'dataset']
].sort_values(by=['dataset', 'class']).to_csv(file_to_write, index=False)
written_files.append(file_to_write)

tmp_df = metrics_by_cl_df[['dataset', 'TP_k', 'FP_k', 'FN_k']].groupby(by='dataset', as_index=False).sum()
tmp_df2 =  metrics_by_cl_df[['dataset', 'precision_k', 'recall_k']].groupby(by='dataset', as_index=False).mean()
global_metrics_df = tmp_df.merge(tmp_df2, on='dataset')

file_to_write = os.path.join(OUTPUT_DIR, f'{"dst_" if KEEP_DATASETS else ""}global_metrics.csv')
global_metrics_df.to_csv(file_to_write, index=False)
written_files.append(file_to_write)

# Save the confusion matrix
na_value_category = tagged_dets_gdf.CATEGORY.isna()
sorted_classes =  tagged_dets_gdf.loc[~na_value_category, 'CATEGORY'].sort_values().unique().tolist() + ['background']
tagged_dets_gdf.loc[na_value_category, 'CATEGORY'] = 'background'
tagged_dets_gdf.loc[tagged_dets_gdf.det_category.isna(), 'det_category'] = 'background'

for dataset in datasets_list:
    tagged_dataset_gdf = tagged_dets_gdf[tagged_dets_gdf.dataset == dataset].copy()

    true_class = tagged_dataset_gdf.CATEGORY.to_numpy()
    detected_class = tagged_dataset_gdf.det_category.to_numpy()

    confusion_array = confusion_matrix(true_class, detected_class, labels=sorted_classes)
    confusion_df = pd.DataFrame(confusion_array, index=sorted_classes, columns=sorted_classes, dtype='int64')
    confusion_df.rename(columns={'background': 'missed labels'}, inplace=True)

    file_to_write = (os.path.join(OUTPUT_DIR, f'{dataset}_confusion_matrix.csv'))
    confusion_df.to_csv(file_to_write)
    written_files.append(file_to_write)

print()
logger.info("The following files were written. Let's check them out!")
for written_file in written_files:
    logger.info(written_file)

print()

toc = time()
logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

sys.stderr.flush()