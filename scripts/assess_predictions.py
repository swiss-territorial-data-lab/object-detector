#!/bin/python
# -*- coding: utf-8 -*-

import logging
import logging.config
import time
import argparse
import yaml
import os, sys
import pickle
import json
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from tqdm import tqdm

# the following allows us to import modules from within this file's parent folder
sys.path.insert(0, '.')
from helpers import misc

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('root')


if __name__ == '__main__':

    tic = time.time()
    logger.info('Starting...')

    parser = argparse.ArgumentParser(description="This script assesses the quality of predictions with respect to ground-truth/other labels.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # TODO: check whether the configuration file contains the required information
    OUTPUT_DIR = cfg['output_folder']
    IMG_METADATA_FILE = cfg['datasets']['image_metadata_json']
    PREDICTION_FILES = cfg['datasets']['predictions']
    SPLITTED_AOI_TILES_GEOJSON = cfg['datasets']['splitted_aoi_tiles_geojson']
    GT_LABELS_GEOJSON = cfg['datasets']['ground_truth_labels_geojson']
    OTH_LABELS_GEOJSON = cfg['datasets']['other_labels_geojson']

    # let's make the output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    written_files = []
    
    # ------ Loading datasets

    logger.info("Loading splitted AoI tiles as a GeoPandas DataFrame...")
    splitted_aoi_tiles_gdf = gpd.read_file(SPLITTED_AOI_TILES_GEOJSON)
    logger.info(f"...done. {len(splitted_aoi_tiles_gdf)} records were found.")

    logger.info("Loading Ground Truth Labels as a GeoPandas DataFrame...")
    gt_labels_gdf = gpd.read_file(GT_LABELS_GEOJSON)
    logger.info(f"...done. {len(gt_labels_gdf)} records were found.")

    logger.info("Loading Other Labels as a GeoPandas DataFrame...")
    oth_labels_gdf = gpd.read_file(OTH_LABELS_GEOJSON)
    logger.info(f"...done. {len(oth_labels_gdf)} records were found.")
    
    
    labels_gdf = pd.concat([
        gt_labels_gdf,
        oth_labels_gdf
    ])
    

    logging.info("Clipping labels...")
    tic = time.time()
    
    assert(labels_gdf.crs == splitted_aoi_tiles_gdf.crs)
    
    clipped_labels_gdf = misc.clip_labels(labels_gdf, splitted_aoi_tiles_gdf, fact=0.999)

    file_to_write = os.path.join(OUTPUT_DIR, 'clipped_labels.geojson')

    clipped_labels_gdf.to_crs(epsg=4326).to_file(
        file_to_write, 
        driver='GeoJSON'
    )

    written_files.append(file_to_write)

    logging.info(f"...done. Elapsed time = {(time.time()-tic):.2f} seconds.")

    # ------ Loading image metadata

    with open(IMG_METADATA_FILE, 'r') as fp:
        tmp = json.load(fp)

    # let's extract filenames (w/o path)
    img_metadata_dict = {os.path.split(k)[-1]: v for (k, v) in tmp.items()}

    # ------ Loading predictions

    preds_dict = {}

    for dataset, preds_file in PREDICTION_FILES.items():
        with open(preds_file, 'rb') as fp:
            preds_dict[dataset] = pickle.load(fp)

    # ------ Extracting vector features out of predictions

    preds_gdf_dict = {}
    
    logger.info(f'Extracting vector features...')
    tic = time.time()
    tqdm_log = tqdm(total=len(preds_dict.keys()), position=0)
    
    for dataset, preds in preds_dict.items():

        tqdm_log.set_description_str(f'Current dataset: {dataset}')
        
        features = misc.fast_predictions_to_features(preds, img_metadata_dict=img_metadata_dict)
        gdf = gpd.GeoDataFrame.from_features(features)
        gdf['dataset'] = dataset
        gdf.crs = features[0]['properties']['crs']
        
        preds_gdf_dict[dataset] = gdf[gdf.raster_val == 1.0][['geometry', 'score', 'dataset']]

        tqdm_log.update(1)

    tqdm_log.close()
    logger.info(f'...done. Elapsed time = {(time.time()-tic):.2f} seconds.')

    # ------ Comparing predictions with ground-truth data and computing metrics

    # init
    metrics = {}
    for dataset in preds_dict.keys():
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
                mode='markers+lines',
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
                    mode='markers+lines',
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
                    mode='markers+lines',
                    name=y
                )
            )
                
        fig.update_layout(xaxis_title="threshold")

        file_to_write = os.path.join(OUTPUT_DIR, f'{dataset}_metrics_vs_threshold.html')
        fig.write_html(file_to_write)
        written_files.append(file_to_write)

    


    # ------ tagging predictions

    # we select the threshold which maximizes the f1-score on the val dataset
    selected_threshold = metrics_df_dict['val'].iloc[metrics_df_dict['val']['f1'].argmax()]['threshold']

    logger.info(f"Tagging predictions with threshold = {selected_threshold:.2f}, which maximizes the f1-score on the val dataset.")

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
        tagged_preds_gdf_dict['trn'], 
        tagged_preds_gdf_dict['val'], 
        tagged_preds_gdf_dict['tst'],
        tagged_preds_gdf_dict['oth']
    ])

    file_to_write = os.path.join(OUTPUT_DIR, f'tagged_predictions.geojson')
    tagged_preds_gdf[['geometry', 'score', 'tag', 'dataset']].to_crs(epsg=4326).to_file(file_to_write, driver='GeoJSON', index=False)
    written_files.append(file_to_write)

    # ------ wrap-up

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    print()

    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()