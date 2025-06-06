import argparse
import os
import sys
import time
import yaml

import geopandas as gpd
import numpy as np
import pandas as pd

import helpers.misc as misc

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
    OUTPUT_DIR = cfg['output_dir']
    LABELS = cfg['labels'] if 'labels' in cfg.keys() else None
    DETECTION_FILES = cfg['detections']

    DISTANCE = cfg['distance']
    SCORE_THD = cfg['score_threshold']
    IOU_THD = cfg['iou_threshold']
    AREA_THD = cfg['area_threshold'] if 'area_threshold' in cfg.keys() else None

    OVERWRITE = cfg['overwrite']

    os.chdir(WORKING_DIR)
    logger.info(f'Working directory set to {WORKING_DIR}')
    os.makedirs(OUTPUT_DIR, exist_ok=True)        

    written_files = [] 

    last_written_file = os.path.join(
        OUTPUT_DIR, 
        f'merged_detections_at_{SCORE_THD}_threshold.gpkg'.replace('0.', '0dot')
    )
    last_metric_file = os.path.join(
        OUTPUT_DIR, 
        f'reliability_diagram_merged_detections_single_class.jpeg'
    )
    if os.path.exists(last_written_file) and os.path.exists(last_metric_file) and not OVERWRITE:           
        logger.success(f"Done! All files already exist in folder {OUTPUT_DIR}. Exiting.")
        sys.exit(0)

    logger.info("Loading split AoI tiles as a GeoPandas DataFrame...")
    tiles_gdf = gpd.read_file('split_aoi_tiles.geojson')
    tiles_gdf = tiles_gdf.to_crs(2056)
    if 'year_tile' in tiles_gdf.keys(): 
        tiles_gdf['year_tile'] = tiles_gdf.year_tile.astype(int)
    logger.success(f"Done! {len(tiles_gdf)} features were found.")

    logger.info("Loading detections as a GeoPandas DataFrame...")

    detections_gdf = gpd.GeoDataFrame()

    for dataset, dets_file in DETECTION_FILES.items():
        detections_ds_gdf = gpd.read_file(dets_file)    
        detections_ds_gdf[f'dataset'] = dataset
        detections_gdf = pd.concat([detections_gdf, detections_ds_gdf], axis=0, ignore_index=True)
    detections_gdf = detections_gdf.to_crs(2056)
    detections_gdf['area'] = detections_gdf.area 
    detections_gdf['det_id'] = detections_gdf.index
    if 'year_det' in detections_gdf.keys(): 
        detections_gdf['year_det'] = detections_gdf.year_det.astype(int)
    logger.success(f"Done! {len(detections_gdf)} features were found.")

    # Merge features
    logger.info(f"Merge adjacent polygons overlapping tiles with a buffer of {DISTANCE} m...")
    detections_all_years_gdf = gpd.GeoDataFrame()

    # Process detection by year
    for year in detections_gdf.year_det.unique():
        detections_by_year_gdf = detections_gdf[detections_gdf['year_det']==year]

        detections_buffer_gdf = detections_by_year_gdf.copy()
        detections_buffer_gdf['geometry'] = detections_by_year_gdf.geometry.buffer(DISTANCE, resolution=2)

        # Saves the id of polygons contained entirely within the tile (no merging with adjacent tiles), to avoid merging them if they are at a distance of less than thd  
        detections_tiles_join_gdf = gpd.sjoin(tiles_gdf, detections_buffer_gdf, how='left', predicate='contains')
        remove_det_list = detections_tiles_join_gdf.det_id.unique().tolist()

        detections_overlap_tiles_gdf = detections_by_year_gdf[~detections_by_year_gdf.det_id.isin(remove_det_list)].drop_duplicates(subset=['det_id'], ignore_index=True)
        detections_within_tiles_gdf = detections_by_year_gdf[detections_by_year_gdf.det_id.isin(remove_det_list)].drop_duplicates(subset=['det_id'], ignore_index=True)

        # Merge polygons within the thd distance
        detections_overlap_tiles_gdf.loc[:, 'geometry'] = detections_overlap_tiles_gdf.buffer(DISTANCE, resolution=2)
        detections_dissolve_gdf = detections_overlap_tiles_gdf[['det_id', 'geometry']].dissolve(as_index=False)
        detections_merge_gdf = detections_dissolve_gdf.explode(ignore_index=True)
        del detections_dissolve_gdf, detections_overlap_tiles_gdf

        if detections_merge_gdf.isnull().values.any():
            detections_merge_gdf = gpd.GeoDataFrame()
        else:
            detections_merge_gdf.geometry = detections_merge_gdf.buffer(-DISTANCE, resolution=2)
    
        # Spatially join merged detection with raw ones to retrieve relevant information (score, area,...)
        detections_merge_gdf['index_merge'] = detections_merge_gdf.index
        detections_join_gdf = gpd.sjoin(detections_merge_gdf, detections_by_year_gdf, how='inner', predicate='intersects')

        det_class_all = []
        det_score_all = []

        for id in detections_merge_gdf.index_merge.unique():
            detections_by_year_gdf = detections_join_gdf.copy()
            detections_by_year_gdf = detections_by_year_gdf[(detections_by_year_gdf['index_merge']==id)]
            detections_by_year_gdf.rename(columns={'score_left': 'score'}, inplace=True)
            det_score_all.append(detections_by_year_gdf['score'].mean())
            detections_by_year_gdf = detections_by_year_gdf.dissolve(by='det_class', aggfunc='sum', as_index=False)
            # Keep class of largest det
            if len(detections_by_year_gdf) > 0:
                detections_by_year_gdf['det_class'] = detections_by_year_gdf.loc[detections_by_year_gdf['area'] == detections_by_year_gdf['area'].max(), 
                                                                'det_class'].iloc[0]    
                det_class = detections_by_year_gdf['det_class'].drop_duplicates().tolist()
            else:
                det_class = [0]
            det_class_all.append(det_class[0])

        detections_merge_gdf['det_class'] = det_class_all
        detections_merge_gdf['score'] = det_score_all

        complete_merge_dets_gdf = pd.merge(detections_merge_gdf, detections_join_gdf[['index_merge', 'year_det'] + ([] if 'dataset' in detections_merge_gdf.columns else ['dataset'])], on='index_merge')
        detections_all_years_gdf = pd.concat([detections_all_years_gdf, complete_merge_dets_gdf, detections_within_tiles_gdf], ignore_index=True)

        del complete_merge_dets_gdf, detections_merge_gdf, detections_by_year_gdf, detections_within_tiles_gdf, detections_join_gdf

    # get classe ids
    CATEGORIES = os.path.join('category_ids.json')
    categories_info_df, _ = misc.get_categories(CATEGORIES)

    detections_all_years_gdf['det_category'] = [
        categories_info_df.loc[categories_info_df.label_class==det_class+1, 'CATEGORY'].iloc[0] 
        if not np.isnan(det_class) else None
        for det_class in detections_all_years_gdf.det_class.to_numpy()
    ] 

    # Remove duplicate detection for a given year
    detections_all_years_gdf.drop_duplicates(subset=['geometry', 'year_det'], inplace=True)
    td = len(detections_all_years_gdf)
    
    # Filter dataframe by score value
    detections_all_years_gdf = detections_all_years_gdf[detections_all_years_gdf.score > SCORE_THD]
    sc = len(detections_all_years_gdf)
    logger.info(f"{td - sc} detections were removed by score filtering (score threshold = {SCORE_THD})")

    logger.success(f"Done! {len(detections_all_years_gdf)} features were kept.")
    if len(detections_all_years_gdf) > 0:
        logger.success(f'The covered area is {round(detections_all_years_gdf.unary_union.area/1000000, 2)} km2.')

    # Save processed results
    detections_all_years_gdf = detections_all_years_gdf.to_crs(2056)
    final_columns = ['geometry', 'score', 'det_class', 'det_category', 'year_det']
    detections_all_years_gdf[final_columns] .to_file(last_written_file, driver='GPKG', index=False)
    written_files.append(last_written_file)       

    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    toc = time.time()
    logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()