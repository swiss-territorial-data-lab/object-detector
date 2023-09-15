#!/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import yaml
import json
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.graph_objects as go


from tqdm import tqdm
# the following lines allow us to import modules from within this file's parent folder
from inspect import getsourcefile
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from helpers import misc
from helpers.constants import DONE_MSG, SCATTER_PLOT_MODE

from loguru import logger
logger = misc.format_logger(logger)


def main(cfg_file_path):

    tic = time.time()
    logger.info('Starting...')

    logger.info(f"Using {cfg_file_path} as config file.")

    with open(cfg_file_path) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    OUTPUT_DIR = cfg['output_folder']
    IMG_METADATA_FILE = cfg['datasets']['image_metadata_json']
    PREDICTION_FILES = cfg['datasets']['detections']
    SPLIT_AOI_TILES_GEOJSON = cfg['datasets']['split_aoi_tiles_geojson']
    
    if 'ground_truth_labels_geojson' in cfg['datasets'].keys():
        GT_LABELS_GEOJSON = cfg['datasets']['ground_truth_labels_geojson']
    else:
        GT_LABELS_GEOJSON = None
    if 'other_labels_geojson' in cfg['datasets'].keys():
        OTH_LABELS_GEOJSON = cfg['datasets']['other_labels_geojson']
    else:
        OTH_LABELS_GEOJSON = None

    # let's make the output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    written_files = []
    
    # ------ Loading datasets

    logger.info("Loading split AoI tiles as a GeoPandas DataFrame...")
    split_aoi_tiles_gdf = gpd.read_file(SPLIT_AOI_TILES_GEOJSON)
    logger.success(f"{DONE_MSG} {len(split_aoi_tiles_gdf)} records were found.")

    if GT_LABELS_GEOJSON:
        logger.info("Loading Ground Truth Labels as a GeoPandas DataFrame...")
        gt_labels_gdf = gpd.read_file(GT_LABELS_GEOJSON)
        logger.success(f"{DONE_MSG} {len(gt_labels_gdf)} records were found.")

    if OTH_LABELS_GEOJSON:
        logger.info("Loading Other Labels as a GeoPandas DataFrame...")
        oth_labels_gdf = gpd.read_file(OTH_LABELS_GEOJSON)
        logger.success(f"{DONE_MSG} {len(oth_labels_gdf)} records were found.")

    if GT_LABELS_GEOJSON and OTH_LABELS_GEOJSON:
        labels_gdf = pd.concat([
            gt_labels_gdf,
            oth_labels_gdf
        ])
    elif GT_LABELS_GEOJSON and not OTH_LABELS_GEOJSON:
        labels_gdf = gt_labels_gdf.copy()
    elif not GT_LABELS_GEOJSON and OTH_LABELS_GEOJSON:
        labels_gdf = oth_labels_gdf.copy()
    else:
        labels_gdf = pd.DataFrame() 
        
    
    if len(labels_gdf)>0:
        logger.info("Clipping labels...")
        tic = time.time()

        assert(labels_gdf.crs == split_aoi_tiles_gdf.crs)

        clipped_labels_gdf = misc.clip_labels(labels_gdf, split_aoi_tiles_gdf, fact=0.999)

        file_to_write = os.path.join(OUTPUT_DIR, 'clipped_labels.geojson')

        clipped_labels_gdf.to_crs(epsg=4326).to_file(
            file_to_write, 
            driver='GeoJSON'
        )

        written_files.append(file_to_write)

        logger.success(f"{DONE_MSG} Elapsed time = {(time.time()-tic):.2f} seconds.")

    # ------ Loading image metadata

    with open(IMG_METADATA_FILE, 'r') as fp:
        tmp = json.load(fp)

    # let's extract filenames (w/o path)
    img_metadata_dict = {os.path.split(k)[-1]: v for (k, v) in tmp.items()}

    # ------ Loading detections

    preds_gdf_dict = {}

    for dataset, preds_file in PREDICTION_FILES.items():
        preds_gdf_dict[dataset] = gpd.read_file(preds_file)


    if len(labels_gdf)>0:
    
        # ------ Comparing detections with ground-truth data and computing metrics

        # init
        metrics = {}
        for dataset in preds_gdf_dict.keys():
            metrics[dataset] = []

        metrics_df_dict = {}
        thresholds = np.arange(0.05, 1., 0.05)

        outer_tqdm_log = tqdm(total=len(metrics.keys()), position=0)

        for dataset in metrics.keys():

            outer_tqdm_log.set_description_str(f'Current dataset: {dataset}')
            inner_tqdm_log = tqdm(total=len(thresholds), position=1, leave=False)

            for threshold in thresholds:

                inner_tqdm_log.set_description_str(f'Threshold = {threshold:.2f}')

                tmp_gdf = preds_gdf_dict[dataset].copy()
                tmp_gdf.to_crs(epsg=clipped_labels_gdf.crs.to_epsg(), inplace=True)
                tmp_gdf = tmp_gdf[tmp_gdf.score >= threshold].copy()

                tp_gdf, fp_gdf, fn_gdf = misc.get_fractional_sets(
                    tmp_gdf, 
                    clipped_labels_gdf[clipped_labels_gdf.dataset == dataset]
                )

                precision, recall, f1 = misc.get_metrics(tp_gdf, fp_gdf, fn_gdf)

                metrics[dataset].append({
                    'threshold': threshold, 
                    'precision': precision, 
                    'recall': recall, 
                    'f1': f1, 
                    'TP': len(tp_gdf), 
                    'FP': len(fp_gdf), 
                    'FN': len(fn_gdf)
                })

                inner_tqdm_log.update(1)

            metrics_df_dict[dataset] = pd.DataFrame.from_records(metrics[dataset])
            outer_tqdm_log.update(1)

        inner_tqdm_log.close()
        outer_tqdm_log.close()

        # let's generate some plots!

        fig = go.Figure()

        for dataset in metrics.keys():

            fig.add_trace(
                go.Scatter(
                    x=metrics_df_dict[dataset]['recall'],
                    y=metrics_df_dict[dataset]['precision'],
                    mode=SCATTER_PLOT_MODE,
                    text=metrics_df_dict[dataset]['threshold'], 
                    name=dataset
                )
            )

        fig.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            xaxis=dict(range=[0., 1]),
            yaxis=dict(range=[0., 1])
        )

        file_to_write = os.path.join(OUTPUT_DIR, 'precision_vs_recall.html')
        fig.write_html(file_to_write)
        written_files.append(file_to_write)


        for dataset in metrics.keys():

            fig = go.Figure()

            for y in ['TP', 'FN', 'FP']:

                fig.add_trace(
                    go.Scatter(
                        x=metrics_df_dict[dataset]['threshold'],
                        y=metrics_df_dict[dataset][y],
                        mode=SCATTER_PLOT_MODE,
                        name=y
                    )
                )

            fig.update_layout(xaxis_title="threshold", yaxis_title="#")

            file_to_write = os.path.join(OUTPUT_DIR, f'{dataset}_TP-FN-FP_vs_threshold.html')
            fig.write_html(file_to_write)
            written_files.append(file_to_write)


        for dataset in metrics.keys():

            fig = go.Figure()

            for y in ['precision', 'recall', 'f1']:

                fig.add_trace(
                    go.Scatter(
                        x=metrics_df_dict[dataset]['threshold'],
                        y=metrics_df_dict[dataset][y],
                        mode=SCATTER_PLOT_MODE,
                        name=y
                    )
                )

            fig.update_layout(xaxis_title="threshold")

            file_to_write = os.path.join(OUTPUT_DIR, f'{dataset}_metrics_vs_threshold.html')
            fig.write_html(file_to_write)
            written_files.append(file_to_write)


        # ------ tagging detections

        # we select the threshold which maximizes the f1-score on the val dataset
        selected_threshold = metrics_df_dict['val'].iloc[metrics_df_dict['val']['f1'].argmax()]['threshold']

        logger.info(f"Tagging detections with threshold = {selected_threshold:.2f}, which maximizes the f1-score on the val dataset.")

        tagged_preds_gdf_dict = {}

        # TRUE/FALSE POSITIVES, FALSE NEGATIVES

        for dataset in metrics.keys():

            tmp_gdf = preds_gdf_dict[dataset].copy()
            tmp_gdf.to_crs(epsg=clipped_labels_gdf.crs.to_epsg(), inplace=True)
            tmp_gdf = tmp_gdf[tmp_gdf.score >= selected_threshold].copy()

            tp_gdf, fp_gdf, fn_gdf = misc.get_fractional_sets(tmp_gdf, clipped_labels_gdf[clipped_labels_gdf.dataset == dataset])
            tp_gdf['tag'] = 'TP'
            tp_gdf['dataset'] = dataset
            fp_gdf['tag'] = 'FP'
            fp_gdf['dataset'] = dataset
            fn_gdf['tag'] = 'FN'
            fn_gdf['dataset'] = dataset

            tagged_preds_gdf_dict[dataset] = pd.concat([tp_gdf, fp_gdf, fn_gdf])
            precision, recall, f1 = misc.get_metrics(tp_gdf, fp_gdf, fn_gdf)
            logger.info(f'Dataset = {dataset} => precision = {precision:.3f}, recall = {recall:.3f}, f1 = {f1:.3f}')

        tagged_preds_gdf = pd.concat([
            tagged_preds_gdf_dict[x] for x in metrics.keys()
        ])

        file_to_write = os.path.join(OUTPUT_DIR, 'tagged_detections.gpkg')
        tagged_preds_gdf[['geometry', 'score', 'tag', 'dataset']].to_file(file_to_write, driver='GPKG', index=False)
        written_files.append(file_to_write)

    # ------ wrap-up

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    print()

    toc = time.time()
    logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This script assesses the quality of detections with respect to ground-truth/other labels.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    main(args.config_file)

    