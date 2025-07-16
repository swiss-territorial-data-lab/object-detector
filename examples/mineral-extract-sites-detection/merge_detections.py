import argparse
import os
import sys
import time
import yaml

import geopandas as gpd
import numpy as np
import pandas as pd
import json

sys.path.insert(0, '../..')
import helpers.metrics as metrics
import helpers.misc as misc
from helpers.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


if __name__ == "__main__":

    # Chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script assess the post-processed detections")
    parser.add_argument('config_file', type=str, help='input geojson path')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_directory']
    LABELS = misc.none_if_undefined(cfg, 'labels')
    DETECTION_FILES = cfg['detections']
    DISTANCE = cfg['distance']
    SCORE_THD = cfg['score_threshold']
    IOU_THD = misc.none_if_undefined(cfg, 'iou_threshold')
    AREA_THD = misc.none_if_undefined(cfg, 'area_threshold')
    ASSESS = cfg['assess']['enable']
    METHOD = cfg['assess']['metrics_method']

    os.chdir(WORKING_DIR)
    logger.info(f'Working directory set to {WORKING_DIR}')

    written_files = [] 

    logger.info("Loading split AoI tiles as a GeoPandas DataFrame...")
    tiles_gdf = gpd.read_file('split_aoi_tiles.geojson')
    tiles_gdf = tiles_gdf.to_crs(2056)
    if 'year_tile' in tiles_gdf.keys(): 
        tiles_gdf['year_tile'] = tiles_gdf.year_tile.astype(int)
    logger.success(f"{DONE_MSG} {len(tiles_gdf)} features were found.")

    logger.info("Loading detections as a GeoPandas DataFrame...")

    detections_gdf = gpd.GeoDataFrame()

    for dataset, dets_file in DETECTION_FILES.items():
        detections_ds_gdf = gpd.read_file(dets_file)    
        detections_ds_gdf[f'dataset'] = dataset
        detections_gdf = pd.concat([detections_gdf, detections_ds_gdf], axis=0, ignore_index=True)
    detections_gdf = detections_gdf.to_crs(2056)
    detections_gdf['area'] = detections_gdf.area 
    detections_gdf['det_id'] = detections_gdf.index
    if 'year_det' in detections_gdf:
        if not detections_gdf['year_det'].all(): 
            detections_gdf['year_det'] = detections_gdf.year_det.astype(int)
    logger.success(f"{DONE_MSG} {len(detections_gdf)} features were found.")

    # get classe ids
    filepath = open(os.path.join('category_ids.json'))
    categories_json = json.load(filepath)
    filepath.close()
    categories_info_df = pd.DataFrame()
    for key in categories_json.keys():
        categories_tmp = {sub_key: [value] for sub_key, value in categories_json[key].items()}
        categories_info_df = pd.concat([categories_info_df, pd.DataFrame(categories_tmp)], ignore_index=True)
    categories_info_df.sort_values(by=['id'], inplace=True, ignore_index=True)
    categories_info_df.drop(['supercategory'], axis=1, inplace=True)
    categories_info_df.rename(columns={'name':'CATEGORY', 'id': 'label_class'},inplace=True)
    id_classes = range(len(categories_json))

    # Merge features
    logger.info(f"Merge adjacent polygons overlapping tiles with a buffer of {DISTANCE} m...")
    detections_year = gpd.GeoDataFrame()

    # Process detection by year
    for year in detections_gdf.year_det.unique():
        detections_gdf = detections_gdf.copy()
        detections_by_year_gdf = detections_gdf[detections_gdf['year_det']==year]

        # Merge overlapping polygons
        detections_merge_overlap_poly_gdf = misc.merge_polygons(detections_by_year_gdf, id_name='det_id')

        # Saves the id of polygons contained entirely within the tile (no merging with adjacent tiles), to avoid merging them if they are at a distance of less than thd  
        detections_buffer_gdf = detections_merge_overlap_poly_gdf.copy()
        detections_buffer_gdf['geometry'] = detections_buffer_gdf.geometry.buffer(1, join_style='mitre')
        detections_tiles_join_gdf = gpd.sjoin(tiles_gdf, detections_buffer_gdf, how='left', predicate='contains')
        remove_det_list = detections_tiles_join_gdf.det_id.unique().tolist()
        
        detections_within_tiles_gdf = gpd.GeoDataFrame()
        detections_within_tiles_gdf = detections_buffer_gdf[detections_buffer_gdf.det_id.isin(remove_det_list)].drop_duplicates(subset=['det_id'], ignore_index=True)

        # Merge adjacent polygons between tiles
        detections_overlap_tiles_gdf = gpd.GeoDataFrame()
        detections_overlap_tiles_gdf = detections_buffer_gdf[~detections_buffer_gdf.det_id.isin(remove_det_list)].drop_duplicates(subset=['det_id'], ignore_index=True)
        detections_overlap_tiles_gdf = misc.merge_polygons(detections_overlap_tiles_gdf)
    
        # Concat polygons contained within a tile and the merged ones
        detections_merge_gdf = pd.concat([detections_overlap_tiles_gdf, detections_within_tiles_gdf], axis=0, ignore_index=True)
        detections_merge_gdf['geometry'] = detections_merge_gdf.geometry.buffer(-1, join_style='mitre')

        # Merge adjacent polygons within the provided thd distance
        detections_merge_gdf['geometry'] = detections_merge_gdf.geometry.buffer(DISTANCE, join_style='mitre')
        detections_merge_gdf = misc.merge_polygons(detections_merge_gdf)
        detections_merge_gdf['geometry'] = detections_merge_gdf.geometry.buffer(-DISTANCE, join_style='mitre')
        detections_merge_gdf = detections_merge_gdf.explode(ignore_index=True)
        detections_merge_gdf['id'] = detections_merge_gdf.index


        # Spatially join merged detection with raw ones to retrieve relevant information (score, area,...)
        # Select the class of the largest polygon -> To Do: compute a parameter dependant of the area and the score
        # Score averaged over all the detection polygon (even if the class is different from the selected one)
        detections_join_gdf = gpd.sjoin(detections_merge_gdf, detections_by_year_gdf, how='inner', predicate='intersects')

        det_class_all = []
        det_score_all = []

        for id in detections_merge_gdf.id.unique():
            detections_by_year_gdf = detections_join_gdf.copy()
            detections_by_year_gdf = detections_by_year_gdf[(detections_by_year_gdf['id']==id)]
            detections_by_year_gdf = detections_by_year_gdf.rename(columns={'score_left': 'score'})
            det_score_all.append(detections_by_year_gdf['score'].mean())
            detections_by_year_gdf = detections_by_year_gdf.dissolve(by='det_class', aggfunc='sum', as_index=False)
            if len(detections_by_year_gdf) > 0:
                detections_by_year_gdf['det_class'] = detections_by_year_gdf.loc[detections_by_year_gdf['area']==detections_by_year_gdf['area'].max(), 
                                                                'det_class'].iloc[0]    
                det_class = detections_by_year_gdf['det_class'].drop_duplicates().tolist()
            else:
                det_class = [0]
            det_class_all.append(det_class[0])

        detections_merge_gdf['det_class'] = det_class_all
        detections_merge_gdf['score'] = det_score_all

        detections_merge_gdf = pd.merge(detections_merge_gdf, detections_join_gdf[
            ['id', 'dataset', 'year_det']], 
            on='id')
        detections_year = pd.concat([detections_year, detections_merge_gdf])

    detections_year['det_category'] = [
        categories_info_df.loc[categories_info_df.label_class==det_class+1, 'CATEGORY'].iloc[0] 
        if not np.isnan(det_class) else None
        for det_class in detections_year.det_class.to_numpy()
    ] 

    # Remove duplicate detection for a given year
    detections_merge_gdf = detections_year.drop_duplicates(subset=['geometry', 'year_det'])
    
    nb_detections = len(detections_merge_gdf)
    logger.info(f"... {nb_detections} detections remaining after union of the shapes.")
    
    # Filter dataframe by score value
    detections_merge_gdf = detections_merge_gdf[detections_merge_gdf.score > SCORE_THD]
    nb_score = len(detections_merge_gdf)
    logger.info(f"{nb_detections - nb_score} detections were removed by score filtering (score threshold = {SCORE_THD})")

    if ASSESS:
        logger.info("Loading labels as a GeoPandas DataFrame...")
        labels_gdf = gpd.read_file(LABELS)
        labels_gdf = labels_gdf.to_crs(2056)
        if 'year' in labels_gdf.keys():  
            labels_gdf['year'] = labels_gdf.year.astype(int)       
            labels_gdf = labels_gdf.rename(columns={"year": "year_label"})
        logger.success(f"{DONE_MSG} {len(labels_gdf)} features were found.")

        # append class ids to labels
        labels_gdf['CATEGORY'] = labels_gdf.CATEGORY.astype(str)
        labels_w_id_gdf = labels_gdf.merge(categories_info_df, on='CATEGORY', how='left')

        logger.info('Tag detections and get metrics...')

        metrics_dict = {}
        metrics_dict_by_cl = []
        metrics_cl_df_dict = {}

        tp_gdf, fp_gdf, fn_gdf, mismatched_class_gdf, small_poly_gdf = metrics.get_fractional_sets(
            detections_merge_gdf, labels_w_id_gdf, IOU_THD, AREA_THD)

        tp_gdf['tag'] = 'TP'
        fp_gdf['tag'] = 'FP'
        fn_gdf['tag'] = 'FN'
        mismatched_class_gdf['tag'] = 'wrong class'
        small_poly_gdf['tag'] = 'small polygon'

        tagged_dets_gdf = pd.concat([tp_gdf, fp_gdf, fn_gdf, mismatched_class_gdf], ignore_index=True)

        logger.info(f'TP = {len(tp_gdf)}, FP = {len(fp_gdf)}, FN = {len(fn_gdf)}')
        tagged_dets_gdf['det_category'] = [
            categories_info_df.loc[categories_info_df.label_class==det_class+1, 'CATEGORY'].iloc[0] 
            if not np.isnan(det_class) else None
            for det_class in tagged_dets_gdf.det_class.to_numpy()
        ] 

        tp_k, fp_k, fn_k, p_k, r_k, f1_k, accuracy, precision, recall, f1 = metrics.get_metrics(tp_gdf, fp_gdf, fn_gdf, mismatched_class_gdf, id_classes, method=METHOD)
        logger.info(f'Detection score threshold = {SCORE_THD}')
        logger.info(f'accuracy = {accuracy:.3f}')
        logger.info(f'Method = {METHOD}: precision = {precision:.3f}, recall = {recall:.3f}, f1 = {f1:.3f}')

        # Save tagged processed results 
        feature = os.path.join(f'tagged_merged_detections_at_{SCORE_THD}_threshold.gpkg'.replace('0.', '0dot'))
        tagged_dets_gdf = tagged_dets_gdf.to_crs(2056)
        tagged_dets_gdf = tagged_dets_gdf.rename(columns={'CATEGORY': 'label_category'}, errors='raise')
        if 'year_label' in tagged_dets_gdf.keys() and 'year_det' in tagged_dets_gdf.keys():
            tagged_dets_gdf[['geometry', 'det_id', 'score', 'tag', 'label_class', 'label_category', 'year_label', 'det_class', 'det_category', 'year_det']]\
                .to_file(feature, driver='GPKG', index=False)
        else:
            tagged_dets_gdf[['geometry', 'det_id', 'score', 'tag', 'label_class', 'label_category', 'det_class', 'det_category']]\
            .to_file(feature, driver='GPKG', index=False)
        written_files.append(feature)

        # label classes starting at 1 and detection classes starting at 0.
        for id_cl in id_classes:
            metrics_dict_by_cl.append({
                'class': id_cl,
                'precision_k': p_k[id_cl],
                'recall_k': r_k[id_cl],
                'f1_k': f1_k[id_cl],
                'TP_k' : tp_k[id_cl],
                'FP_k' : fp_k[id_cl],
                'FN_k' : fn_k[id_cl],
            }) 
            
        metrics_cl_df_dict = pd.DataFrame.from_records(metrics_dict_by_cl)

        # Save the metrics by class for each dataset
        metrics_by_cl_df = pd.DataFrame()
        dataset_df = metrics_cl_df_dict.copy()
        metrics_by_cl_df = pd.concat([metrics_by_cl_df, dataset_df], ignore_index=True)

        metrics_by_cl_df['category'] = [
            categories_info_df.loc[categories_info_df.label_class==det_class+1, 'CATEGORY'].iloc[0] 
            for det_class in metrics_by_cl_df['class'].to_numpy()
        ] 

        file_to_write = os.path.join('metrics_by_class_merged_detections.csv')
        metrics_by_cl_df[
            ['class', 'category', 'TP_k', 'FP_k', 'FN_k', 'precision_k', 'recall_k', 'f1_k']
        ].sort_values(by=['class']).to_csv(file_to_write, index=False)
        written_files.append(file_to_write)

    # Save processed results
    feature = os.path.join(f'merged_detections_at_{SCORE_THD}_threshold.gpkg'.replace('0.', '0dot'))
    detections_merge_gdf = detections_merge_gdf.to_crs(2056)
    detections_merge_gdf[['geometry', 'score', 'det_class', 'det_category', 'year_det']]\
        .to_file(feature, driver='GPKG', index=False)
    written_files.append(feature)     

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    toc = time.time()
    logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()