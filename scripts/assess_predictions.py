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

# the following lines allow us to import modules from within this file's parent folder
from inspect import getsourcefile
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

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

    # with open('examples/swimming-pool-detection/GE/config_GE.yaml') as fp:
    #     cfg=yaml.load(fp, Loader=yaml.FullLoader)['assess_predictions.py']
    # os.chdir('examples/swimming-pool-detection/GE')

    # TODO: check whether the configuration file contains the required information
    OUTPUT_DIR = cfg['output_folder']
    IMG_METADATA_FILE = cfg['datasets']['image_metadata_json']
    PREDICTION_FILES = cfg['datasets']['predictions']
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
    logger.info(f"...done. {len(split_aoi_tiles_gdf)} records were found.")

    if GT_LABELS_GEOJSON:
        logger.info("Loading Ground Truth Labels as a GeoPandas DataFrame...")
        gt_labels_gdf = gpd.read_file(GT_LABELS_GEOJSON)
        logger.info(f"...done. {len(gt_labels_gdf)} records were found.")

    if OTH_LABELS_GEOJSON:
        logger.info("Loading Other Labels as a GeoPandas DataFrame...")
        oth_labels_gdf = gpd.read_file(OTH_LABELS_GEOJSON)
        logger.info(f"...done. {len(oth_labels_gdf)} records were found.")

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
        logging.info("Clipping labels...")
        tic = time.time()

        assert(labels_gdf.crs == split_aoi_tiles_gdf.crs)

        clipped_labels_gdf = misc.clip_labels(labels_gdf, split_aoi_tiles_gdf, fact=0.999)

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

        preds_gdf_dict[dataset] = gdf[gdf.raster_val == 1.0][['geometry', 'pred_class', 'score', 'dataset']]

        file_to_write = os.path.join(OUTPUT_DIR, f"{dataset}_predictions.geojson")
        preds_gdf_dict[dataset].to_crs(epsg=4326).to_file(file_to_write, driver='GeoJSON', index=False)
        written_files.append(file_to_write)

        tqdm_log.update(1)

    tqdm_log.close()
    logger.info(f'...done. Elapsed time = {(time.time()-tic):.2f} seconds.')

    if len(labels_gdf)>0:
    
        # ------ Comparing predictions with ground-truth data and computing metrics

        # init
        metrics = {}
        metrics_cl = {}
        for dataset in preds_dict.keys():
            metrics[dataset] = []
            metrics_cl[dataset] = []

        metrics_df_dict = {}
        metrics_cl_df_dict = {}
        thresholds = np.arange(0.05, 1., 0.05)

        # get classes ids
        id_classes = {}
        for dataset in metrics.keys():
            
            id_classes[dataset] = preds_gdf_dict[dataset].pred_class.unique()
            id_classes[dataset].sort()
            
            try:
                assert (id_classes['trn'] == id_classes[dataset]).all()
            except AssertionError:
                logger.info(f"There are not the same classes in the 'trn' and '{dataset}' datasets: {id_classes['trn']} vs {id_classes[dataset]}. Please correct this.")
                sys.exit(1)
                
        id_classes = id_classes['trn']

        # get labels ids
        filepath = open(os.path.join(OUTPUT_DIR, 'labels_id.json'))
        labels_json = json.load(filepath)
        filepath.close()

        # create contiguous id which should correspond to the pred_class
        labels_info_df = pd.DataFrame()
        labels_temp = labels_json.copy()

        for key in labels_temp.keys():
            
            for info in labels_temp[key].keys():
                labels_temp[key][info] = [labels_temp[key][info]]
            
            df = pd.DataFrame(labels_temp[key])
            
            labels_info_df = pd.concat([labels_info_df, df], ignore_index=True)

            
        labels_info_df.sort_values(by=['id'], inplace=True, ignore_index=True)
        labels_info_df['contig_id'] = labels_info_df.index
        labels_info_df.drop(['supercategory','id'], axis=1, inplace=True)

        # get contiguous id on the clipped labels
        labels_info_df.rename(columns={'name':'CATEGORY'},inplace=True)
        clipped_labels_gdf = clipped_labels_gdf.astype({'CATEGORY':'str'})
        clipped_labels_w_id_gdf = clipped_labels_gdf.merge(labels_info_df, on='CATEGORY', how='left')


        # get metrics
        outer_tqdm_log = tqdm(total=len(metrics.keys()), position=0)

        for dataset in metrics.keys():

            outer_tqdm_log.set_description_str(f'Current dataset: {dataset}')
            inner_tqdm_log = tqdm(total=len(thresholds), position=1, leave=False)

            for threshold in thresholds:

                inner_tqdm_log.set_description_str(f'Threshold = {threshold:.2f}')

                tmp_gdf = preds_gdf_dict[dataset].copy()
                tmp_gdf.to_crs(epsg=clipped_labels_w_id_gdf.crs.to_epsg(), inplace=True)
                tmp_gdf = tmp_gdf[tmp_gdf.score >= threshold].copy()

                tp_gdf, fp_gdf, fn_gdf, non_diag_gdf = misc.get_fractional_sets(
                    tmp_gdf, 
                    clipped_labels_w_id_gdf[clipped_labels_w_id_gdf.dataset == dataset]
                )

                p_k, r_k, precision, recall, f1 = misc.get_metrics(tp_gdf, fp_gdf, fn_gdf, non_diag_gdf, id_classes)

                metrics[dataset].append({
                    'threshold': threshold, 
                    'precision': precision, 
                    'recall': recall, 
                    'f1': f1
                })

                for id_cl in id_classes:
                    metrics_cl[dataset].append({
                        'threshold': threshold,
                        'class': id_cl,
                        'precision_k': p_k[id_cl],
                        'recall_k': r_k[id_cl],
                        'TP_k' : len(tp_gdf[tp_gdf['pred_class']==id_cl]),
                        'FP_k' : len(fp_gdf[fp_gdf['pred_class']==id_cl]) + len(non_diag_gdf[non_diag_gdf['pred_class']==id_cl]),
                        'FN_k' : len(fn_gdf[fn_gdf['pred_class']==id_cl]) + len(non_diag_gdf[non_diag_gdf['contig_id']==id_cl]),
                    })

                metrics_cl_df_dict[dataset] = pd.DataFrame.from_records(metrics_cl[dataset])
                    
                inner_tqdm_log.update(1)

            metrics_df_dict[dataset] = pd.DataFrame.from_records(metrics[dataset])
            outer_tqdm_log.update(1)

        inner_tqdm_log.close()
        outer_tqdm_log.close()

        # let's generate some plots!

        fig = go.Figure()
        fig_k = go.Figure()


        for dataset in metrics.keys():
            # Plot of the precision vs recall

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

        if len(id_classes)>1:
            for dataset in metrics_cl.keys():

                for id_cl in id_classes:

                    fig_k.add_trace(
                        go.Scatter(
                            x=metrics_cl_df_dict[dataset]['recall_k'][metrics_cl_df_dict[dataset]['class']==id_cl],
                            y=metrics_cl_df_dict[dataset]['precision_k'][metrics_cl_df_dict[dataset]['class']==id_cl],
                            mode='markers+lines',
                            text=metrics_cl_df_dict[dataset]['threshold'][metrics_cl_df_dict[dataset]['class']==id_cl],
                            name=dataset+'_'+str(id_cl)
                        )
                    )

            fig_k.update_layout(
                xaxis_title="Recall",
                yaxis_title="Precision",
                xaxis=dict(range=[0., 1]),
                yaxis=dict(range=[0., 1])
            )

            file_to_write = os.path.join(OUTPUT_DIR, 'precision_vs_recall_dep_on_class.html')
            fig_k.write_html(file_to_write)
            written_files.append(file_to_write)


        for dataset in metrics_cl.keys():
            # Generate a plot of TP, FN and FP for each class

            fig = go.Figure()

            for id_cl in id_classes:
                
                for y in ['TP_k', 'FN_k', 'FP_k']:

                    fig.add_trace(
                        go.Scatter(
                            x=metrics_cl_df_dict[dataset]['threshold'][metrics_cl_df_dict[dataset]['class']==id_cl],
                            y=metrics_cl_df_dict[dataset][y][metrics_cl_df_dict[dataset]['class']==id_cl],
                            mode='markers+lines',
                            name=y[0:2]+'_'+str(id_cl)
                        )
                    )

                fig.update_layout(xaxis_title="threshold", yaxis_title="#")
                
            if len(id_classes)>1:
                file_to_write = os.path.join(OUTPUT_DIR, f'{dataset}_TP-FN-FP_vs_threshold_dep_on_class.html')

            else:
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
            tmp_gdf.to_crs(epsg=clipped_labels_w_id_gdf.crs.to_epsg(), inplace=True)
            tmp_gdf = tmp_gdf[tmp_gdf.score >= selected_threshold].copy()

            tp_gdf, fp_gdf, fn_gdf, non_diag_gdf = misc.get_fractional_sets(tmp_gdf, clipped_labels_w_id_gdf[clipped_labels_w_id_gdf.dataset == dataset])
            tp_gdf['tag'] = 'TP'
            tp_gdf['dataset'] = dataset
            fp_gdf['tag'] = 'FP'
            fp_gdf['dataset'] = dataset
            fn_gdf['tag'] = 'FN'
            fn_gdf['dataset'] = dataset
            non_diag_gdf['tag']='ND'
            non_diag_gdf['dataset']=dataset

            tagged_preds_gdf_dict[dataset] = pd.concat([tp_gdf, fp_gdf, fn_gdf, non_diag_gdf])
            p_k, r_k, precision, recall, f1 = misc.get_metrics(tp_gdf, fp_gdf, fn_gdf, non_diag_gdf, id_classes)
            logger.info(f'Dataset = {dataset} => precision = {precision:.3f}, recall = {recall:.3f}, f1 = {f1:.3f}')

        tagged_preds_gdf = pd.concat([
            tagged_preds_gdf_dict[x] for x in metrics.keys()
        ])

        file_to_write = os.path.join(OUTPUT_DIR, f'tagged_predictions.geojson')
        tagged_preds_gdf[['geometry', 'score', 'tag', 'dataset']].to_crs(epsg=4326).to_file(file_to_write, driver='GeoJSON', index=False)
        written_files.append(file_to_write)

    # ------ wrap-up

    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()