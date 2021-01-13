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

    parser = argparse.ArgumentParser(description="This script ...")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # TODO: check whether the configuration file contains the required information
    OUTPUT_DIR = cfg['folders']['output']
    LABELS_GEOJSON = cfg['datasets']['labels_geojson']
    IMG_METADATA_FILE = cfg['datasets']['image_metadata_json']
    PREDICTION_FILES = cfg['datasets']['predictions']
    OK_TILES_FILE = cfg['datasets']['OK_tiles_geojson']
    AOI_TILES_FILE = cfg['datasets']['aoi_tiles_geojson']

    # let's make the output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    written_files = []

    # ------ Loading ground-truth labels

    logging.info("Loading labels...")

    labels_gdf = gpd.read_file(LABELS_GEOJSON)
    # reprojection to EPSG:3857
    labels_gdf = labels_gdf.to_crs(epsg=3857)
    labels_crs = labels_gdf.crs
    
    # ------ Loading tiling system

    logging.info("Loading tiles...")

    # swimming pool (sp) tiles are a subset of AoI tiles, w/ the added "dataset" column
    sp_tiles_gdf = gpd.read_file(OK_TILES_FILE)
    # we need clipped labels even beyond the trn, val, tst sets, in order to assess predictions over the entire AoI
    aoi_tiles_gdf = gpd.read_file(AOI_TILES_FILE)
    # the following allows us to add the dataset column to aoi_tiles_gdf
    aoi_tiles_gdf = pd.merge(aoi_tiles_gdf, sp_tiles_gdf, how='left') 
    # oth = other dataset != {trn,val,tst}
    aoi_tiles_gdf.dataset.fillna('oth', inplace=True)
    aoi_tiles_gdf = aoi_tiles_gdf.to_crs(labels_crs)

    logging.info("Clipping labels...")
    tic = time.time()

    clipped_labels_gdf = misc.clip_labels(labels_gdf, aoi_tiles_gdf, fact=0.999)

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

    # N.B.: the "all" dataset also includes predictions over the trn, val, tst tiles; let's factor out those predictions and generate the "other" (oth) dataset

    trn_val_tst_image_keys = \
        list(preds_dict['trn'].keys()) + \
        list(preds_dict['val'].keys()) + \
        list(preds_dict['tst'].keys())

    trn_val_tst_image_names = set([ 
        os.path.split(k)[-1] for k in trn_val_tst_image_keys
    ])

    if 'all' in preds_dict.keys():

        preds_dict['oth'] = {
            k: v for k, v in preds_dict['all'].items() \
                if os.path.split(k)[-1] not in trn_val_tst_image_names
        }

        # let's free up some memory
        del preds_dict['all']

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

    metrics = {k: [] for k in preds_dict.keys()}
    metrics_df_dict = {}
    thresholds = np.arange(0.05, 1., 0.05)

    outer_tqdm_log = tqdm(total=3, position=0)

    for dataset in metrics.keys():

        outer_tqdm_log.set_description_str(f'Current dataset: {dataset}')
        inner_tqdm_log = tqdm(total=len(thresholds), position=1, leave=False)
    
        for threshold in thresholds:

            inner_tqdm_log.set_description_str(f'Threshold = {threshold:.2f}')

            tmp_gdf = preds_gdf_dict[dataset].copy()
            tmp_gdf = tmp_gdf[tmp_gdf.score >= threshold]

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

    FONT_DICT = dict(family='CMU Sans Serif', size=16)
    MARGIN_DICT = dict(r=20, l=20, b=20, t=20)
    # BGCOLOR_DICT = dict(paper_bgcolor='rgba(211,211,211,0)', plot_bgcolor='rgba(211,211,211,0)')

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
        margin=MARGIN_DICT,
        font=FONT_DICT,
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0., 1]),
        yaxis=dict(range=[0., 1]),
        #**BGCOLOR_DICT
    )


    file_to_write = os.path.join(OUTPUT_DIR, 'precision_vs_recall.html')
    fig.write_html(file_to_write)
    written_files.append(file_to_write)

    fig.update_layout(
            width=1000,
            height=600
        )

    #file_to_write = os.path.join(OUTPUT_DIR, 'precision_vs_recall.eps')
    #fig.write_image(file_to_write)
    #written_files.append(file_to_write)


    for dataset in metrics.keys():

        fig = go.Figure()

        for y in ['TP', 'FP', 'FN']:

            fig.add_trace(
                go.Scatter(
                    x=metrics_df_dict[dataset]['threshold'],
                    y=metrics_df_dict[dataset][y],
                    mode='markers+lines',
                    name=y
                )
            )
                    
        fig.update_layout(
            margin=MARGIN_DICT,
            font=FONT_DICT,
            #**BGCOLOR_DICT,
            xaxis_title="threshold", 
            yaxis_title="#")

        file_to_write = os.path.join(OUTPUT_DIR, f'{dataset}_TP-FN-FP_vs_threshold.html')
        fig.write_html(file_to_write)
        written_files.append(file_to_write)

        fig.update_layout(
            width=1000,
            height=600
        )

        #file_to_write = os.path.join(OUTPUT_DIR, f'{dataset}_TP-FN-FP_vs_threshold.eps')
        #fig.write_image(file_to_write)
        #written_files.append(file_to_write)


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
                
        fig.update_layout(
            margin=MARGIN_DICT,
            font=FONT_DICT,
            #**BGCOLOR_DICT,
            xaxis_title="threshold"
        )

        file_to_write = os.path.join(OUTPUT_DIR, f'{dataset}_metrics_vs_threshold.html')
        fig.write_html(file_to_write)
        written_files.append(file_to_write)

        fig.update_layout(
            width=1000,
            height=600
        )

        #file_to_write = os.path.join(OUTPUT_DIR, f'{dataset}_metrics_vs_threshold.eps')
        #fig.write_image(file_to_write)
        #written_files.append(file_to_write)

    


    # ------ tagging predictions

    # we select the threshold which maximizes the f1-score on the val dataset
    #selected_threshold = metrics_df_dict['val'].iloc[metrics_df_dict['val']['f1'].argmax()]['threshold']
    
    selected_threshold = 0.05

    logger.info(f"Tagging predictions with threshold = {selected_threshold:.2f}, which maximizes the f1-score on the val dataset.")

    tagged_preds_gdf_dict = {}

    # TRUE/FALSE POSITIVES, FALSE NEGATIVES

    for dataset in metrics.keys():

        tmp_gdf = preds_gdf_dict[dataset].copy()
        tmp_gdf = tmp_gdf[tmp_gdf.score >= selected_threshold]

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


    tagged_preds_gdf = pd.DataFrame()

    for dataset in metrics.keys():

        tagged_preds_gdf = pd.concat([
            tagged_preds_gdf, 
            tagged_preds_gdf_dict[dataset]
        ])

    # TMP
    if 'oth' in metrics.keys():
        tagged_preds_gdf_dict['oth'][['geometry', 'score', 'tag', 'dataset']].to_crs(epsg=4326).to_file("oth_tagged_preds.geojson", driver='GeoJSON', index=False)
    # -----

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