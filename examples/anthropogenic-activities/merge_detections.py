import argparse
import os
import sys
import time
import yaml

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(1, '../..')
from helpers.functions_for_examples import get_categories, merge_adjacent_detections, read_dets_and_aoi
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

    tiles_gdf, detections_gdf = read_dets_and_aoi(DETECTION_FILES)

    # Merge features
    logger.info(f"Merge adjacent polygons overlapping tiles with a buffer of {DISTANCE} m...")
    detections_all_years_gdf = gpd.GeoDataFrame()

    # Process detection by year
    for year in detections_gdf.year_det.unique():
        complete_merge_dets_gdf, detections_within_tiles_gdf = merge_adjacent_detections(detections_gdf, tiles_gdf, year, DISTANCE)
        detections_all_years_gdf = pd.concat([detections_all_years_gdf, complete_merge_dets_gdf, detections_within_tiles_gdf], ignore_index=True)

    # get classe ids
    CATEGORIES = os.path.join('category_ids.json')
    categories_info_df, _ = get_categories(CATEGORIES)

    detections_all_years_gdf['det_category'] = [
        categories_info_df.loc[categories_info_df.label_class==det_class+1, 'category'].iloc[0] 
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