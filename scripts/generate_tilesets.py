#!/bin/python
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import argparse
import json
import time
import yaml
import geopandas as gpd
import pandas as pd

from joblib import Parallel, delayed
from tqdm import tqdm

# the following lines allow us to import modules from within this file's parent folder
from inspect import getsourcefile
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from helpers import COCO
from helpers import misc
from helpers.constants import DONE_MSG
from helpers.download_tiles import download_tiles
from helpers.split_tiles import split_tiles

from loguru import logger
logger = misc.format_logger(logger)


class LabelOverflowException(Exception):
    "Raised when a label exceeds the tile size"
    pass


def get_coco_image_and_segmentations(tile, labels, coco_license_id, coco_category, output_dir):
    # From tiles and label, get COCO images, as well as the segmentations and their corresponding coco category for the coco annotations

    _id, _tile = tile

    coco_obj = COCO.COCO()

    this_tile_dirname = os.path.relpath(_tile['img_file'].replace('all', _tile['dataset']), output_dir)
    this_tile_dirname = this_tile_dirname.replace('\\', '/') # should the dirname be generated from Windows

    year = _tile.year_tile if 'year_tile' in _tile.keys() else None
    coco_image = coco_obj.image(output_dir, this_tile_dirname, year, coco_license_id)
    category_id = None
    segments = {}

    if len(labels) > 0:
        
        xmin, ymin, xmax, ymax = [float(x) for x in misc.bounds_to_bbox(_tile['geometry'].bounds).split(',')]
        
        # note the .explode() which turns Multipolygon into Polygons
        clipped_labels_gdf = gpd.clip(labels, _tile['geometry'], keep_geom_type=True).explode(ignore_index=True)

        if 'year_tile' in _tile.keys():
            clipped_labels_gdf = clipped_labels_gdf[clipped_labels_gdf['year_label']==_tile.year_tile] 
   
        for label in clipped_labels_gdf.itertuples():

            scaled_poly = misc.scale_polygon(label.geometry, xmin, ymin, xmax, ymax, 
                                            coco_image['width'], coco_image['height'])
            scaled_poly = scaled_poly[:-1] # let's remove the last point

            segmentation = misc.my_unpack(scaled_poly)

            # Check that label coordinates in the reference system of the image are consistent with image size.
            try:
                assert(min(segmentation) >= 0)
                assert(max(scaled_poly, key = lambda i : i[0])[0] <= coco_image['width'])
                assert(max(scaled_poly, key = lambda i : i[1])[1] <= coco_image['height'])
            except AssertionError:
                raise LabelOverflowException(f"Label boundaries exceed tile size - Tile ID = {_tile['id']}")
            
            # Category attribution
            key = str(label.CATEGORY) + '_' + str(label.SUPERCATEGORY)
            category_id = coco_category[key]['id']

            segments[label.Index] = (category_id, segmentation)
        
    return (coco_image, segments)


def read_labels(path):
    if path:
        logger.info(f"Loading labels from {path} as a GeoPandas DataFrame...")
        labels_gdf = gpd.read_file(path)
        logger.success(f"{DONE_MSG} {len(labels_gdf)} records were found.")
        labels_gdf = misc.find_category(labels_gdf)
        if 'year' in labels_gdf.keys(): 
            labels_gdf['year'] = labels_gdf.year.astype(int)
            labels_gdf = labels_gdf.rename(columns={"year": "year_label"})
    else:
        labels_gdf = gpd.GeoDataFrame()

    return labels_gdf


def main(cfg_file_path):

    tic = time.time()
    logger.info('Starting...')

    logger.info(f"Using {cfg_file_path} as config file.")

    with open(cfg_file_path) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    DEBUG_MODE = cfg['debug_mode']['enable']
    DEBUG_MODE_LIMIT = cfg['debug_mode']['nb_tiles_max']
   
    WORKING_DIR = cfg['working_directory']
    OUTPUT_DIR = cfg['output_folder']
    OVERWRITE = cfg['overwrite']
    
    DATASETS = cfg['datasets']

    # Get label info if available
    GT_LABELS = DATASETS['ground_truth_labels'] if 'ground_truth_labels' in DATASETS.keys() else None
    OTH_LABELS = DATASETS['other_labels'] if 'other_labels' in DATASETS.keys() else None            # Labels that are not good enough to be used for training

    # Choose to add emtpy and FP tiles
    EMPTY_TILES_DICT = cfg['empty_tiles'] if 'empty_tiles' in cfg.keys() else False          # Selected from oth tiles
    ADD_FP_LABELS = DATASETS['add_fp_labels'] if 'add_fp_labels' in DATASETS.keys() else False      # Determine FP tiles based on FP labels
    if ADD_FP_LABELS:
        FP_LABELS = ADD_FP_LABELS['fp_labels']
        FP_FRAC_TRN = ADD_FP_LABELS['frac_trn'] if 'frac_trn' in ADD_FP_LABELS.keys() else 0.7
    else:
        FP_LABELS = None
        FP_FRAC_TRN = None

    # Get tile download information
    IM_SOURCE_TYPE = DATASETS['image_source']['type'].upper()
    if IM_SOURCE_TYPE not in ['XYZ', 'FOLDER']:
        TILE_SIZE = cfg['tile_size']
    else:
        TILE_SIZE = None
    N_JOBS = cfg['n_jobs'] 

    # set seed to split tiles between trn, tst, val. If none, the best partition is chosen automatically based on class partition
    SEED = cfg['seed'] if 'seed' in cfg.keys() else False
    if SEED:
        logger.info(f'The seed is set to {SEED}.')

    if 'COCO_metadata' in cfg.keys():
        COCO_METADATA = cfg['COCO_metadata']
        COCO_YEAR = COCO_METADATA['year']
        COCO_VERSION = COCO_METADATA['version']
        COCO_DESCRIPTION = COCO_METADATA['description']
        COCO_CONTRIBUTOR = COCO_METADATA['contributor']
        COCO_URL = COCO_METADATA['url']
        COCO_LICENSE_NAME = COCO_METADATA['license']['name']
        COCO_LICENSE_URL = COCO_METADATA['license']['url']
        COCO_CATEGORIES_FILE = COCO_METADATA['categories_file'] if 'categories_file' in COCO_METADATA.keys() else None

    os.chdir(WORKING_DIR)
    logger.info(f'Working_directory set to {WORKING_DIR}.')
    # let's make the output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    written_files = []

    # ------ Loading datasets
    
    gt_labels_gdf = read_labels(GT_LABELS)

    oth_labels_gdf = read_labels(OTH_LABELS)

    fp_labels_gdf = read_labels(FP_LABELS)

    # ------ Tile download
    aoi_tiles_gdf, img_metadata_dict, id_list_ept_tiles, dt_written_files = download_tiles(
        DATASETS, gt_labels_gdf, oth_labels_gdf, fp_labels_gdf, EMPTY_TILES_DICT, TILE_SIZE, N_JOBS, OUTPUT_DIR, DEBUG_MODE, DEBUG_MODE_LIMIT, OVERWRITE
    )
    written_files.extend(dt_written_files)

    # ------ Split tiles between training/validation/test/other
    split_aoi_tiles_with_img_md_gdf, st_written_files = split_tiles(
        aoi_tiles_gdf, gt_labels_gdf, oth_labels_gdf, fp_labels_gdf, FP_FRAC_TRN, EMPTY_TILES_DICT, id_list_ept_tiles, img_metadata_dict, TILE_SIZE, SEED, 
        OUTPUT_DIR, DEBUG_MODE
    )
    written_files.extend(st_written_files)

    del aoi_tiles_gdf, fp_labels_gdf, id_list_ept_tiles

    # ------ Generating COCO annotations
    
    if GT_LABELS and OTH_LABELS:
        
        assert(gt_labels_gdf.crs == oth_labels_gdf.crs)
        
        labels_gdf = pd.concat([
            gt_labels_gdf,
            oth_labels_gdf
        ], ignore_index=True)

    elif GT_LABELS and not OTH_LABELS:
        labels_gdf = gt_labels_gdf.reset_index(drop=True)
    elif not GT_LABELS and OTH_LABELS:
        labels_gdf = oth_labels_gdf.reset_index(drop=True)
    else:
        labels_gdf = gpd.GeoDataFrame()
    del gt_labels_gdf, oth_labels_gdf

    if 'COCO_metadata' not in cfg.keys():
        print()
        toc = time.time()
        logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

        sys.stderr.flush()
        sys.exit(0)
    
    # Get possible combinations for category and supercategory
    if len(labels_gdf) > 0:
        combinations_category_dict = labels_gdf.groupby(['CATEGORY', 'SUPERCATEGORY'], as_index=False).size().drop(columns=['size']).to_dict('tight')
        combinations_category_lists = combinations_category_dict['data']

    elif 'category' in COCO_METADATA.keys():
        combinations_category_lists = [[COCO_METADATA['category']['name'], COCO_METADATA['category']['supercategory']]]

    elif COCO_CATEGORIES_FILE:
        logger.warning('The COCO file is generated with tiles only. No label was given.')
        logger.warning('The saved file for category ids is used.')
        categories_json = json.load(open(COCO_CATEGORIES_FILE))
        combinations_category_lists =  [(category['name'], category['supercategory']) for category in categories_json.values()]

    else:
        logger.warning('The COCO file is generated with tiles only. No label was given and no COCO category was defined.')
        logger.warning('A fake category and supercategory is defined for the COCO file.')
        combinations_category_lists = [['foo', 'bar ']]

    coco = COCO.COCO()

    coco_license = coco.license(name=COCO_LICENSE_NAME, url=COCO_LICENSE_URL)
    coco_license_id = coco.insert_license(coco_license)

    logger.info(f'Possible categories and supercategories:')
    coco_categories = {}
    for category, supercategory in combinations_category_lists:
        logger.info(f"    - {category}, {supercategory}")
        
        coco_category_name = str(category)
        coco_category_supercat = str(supercategory)
        key = coco_category_name + '_' + coco_category_supercat

        coco_categories[key] = coco.category(name=coco_category_name, supercategory=coco_category_supercat)

        _ = coco.insert_category(coco_categories[key])

    for dataset in split_aoi_tiles_with_img_md_gdf.dataset.unique():

        dst_coco = coco.copy()
        
        logger.info(f'Generating COCO annotations for the {dataset} dataset...')
        
        dst_coco.set_info(year=COCO_YEAR, 
                      version=COCO_VERSION, 
                      description=f"{COCO_DESCRIPTION} - {dataset} dataset", 
                      contributor=COCO_CONTRIBUTOR, 
                      url=COCO_URL)
        
        tmp_tiles_gdf = split_aoi_tiles_with_img_md_gdf[split_aoi_tiles_with_img_md_gdf.dataset == dataset].dropna()

        if len(labels_gdf) > 0:
            assert(labels_gdf.crs == tmp_tiles_gdf.crs)
        
        tiles_iterator = tmp_tiles_gdf.sort_index().iterrows()

        try:
            results = Parallel(n_jobs=N_JOBS, backend="loky") \
                    (delayed(get_coco_image_and_segmentations) \
                    (tile, labels_gdf, coco_license_id, coco_categories, OUTPUT_DIR) \
                    for tile in tqdm(tiles_iterator, total=len(tmp_tiles_gdf) ))
        except Exception as e:
            logger.critical(f"Tile generation failed. Exception: {e}")
            sys.exit(1)
    
        for result in results:
            
            coco_image, segments = result

            try:
                coco_image_id = dst_coco.insert_image(coco_image)
            except Exception as e:
                logger.critical(f"Could not insert image into the COCO data structure. Exception: {e}")
                sys.exit(1)

            for coco_category_id, segmentation in segments.values():

                coco_annotation = dst_coco.annotation(
                    coco_image_id,
                    coco_category_id,
                    [segmentation],
                    iscrowd=0
                )
                # The bbox for coco objects is defined as [x_min, y_min, width, height].
                # https://cocodataset.org/#format-data under "1. Object Detection"

                try:
                    dst_coco.insert_annotation(coco_annotation)
                except Exception as e:
                    logger.critical(f"Could not insert annotation into the COCO data structure. Exception: {e}")
                    sys.exit(1)
        
        COCO_file = os.path.join(OUTPUT_DIR, f'COCO_{dataset}.json')

        with open(COCO_file, 'w') as fp:
            json.dump(dst_coco.to_json(), fp)
        
        written_files.append(COCO_file)

    categories_file = os.path.join(OUTPUT_DIR, 'category_ids.json')
    with open(categories_file, 'w') as fp:
        json.dump(coco_categories, fp)
    written_files.append(categories_file)

    toc = time.time()
    logger.success(DONE_MSG)
    
    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    toc = time.time()
    logger.success(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()

    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This script generates COCO-annotated training/validation/test/other datasets for object detection tasks.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    main(args.config_file)