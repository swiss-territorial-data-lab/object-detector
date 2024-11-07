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

from helpers import MIL     # MIL stands for Map Image Layer, cf. https://pro.arcgis.com/en/pro-app/help/sharing/overview/map-image-layer.htm
from helpers import WMS     # Web Map Service
from helpers import XYZ     # XYZ link connection
from helpers import FOLDER  # Copy the tile from a folder
from helpers import COCO
from helpers import misc
from helpers.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


class LabelOverflowException(Exception):
    "Raised when a label exceeds the tile size"
    pass


class MissingIdException(Exception):
    "Raised when tiles are lacking IDs"
    pass


class TileDuplicationException(Exception):
    "Raised when the 'id' column contains duplicates"
    pass


class BadTileIdException(Exception):
    "Raised when tile IDs cannot be parsed into X, Y, Z"
    pass


def read_img_metadata(md_file, all_img_path):
    # Read images metadata and return them as dictionnaries with the image path as key.
    img_path = os.path.join(all_img_path, md_file.replace('json', 'tif'))
    
    with open(os.path.join(all_img_path, md_file), 'r') as fp:
        return {img_path: json.load(fp)}


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


def split_dataset(tiles_df, frac_trn=0.7, frac_left_val=0.5, seed=1):
    """Split the dataframe in the traning, validation and test set.

    Args:
        tiles_df (DataFrame): Dataset of the tiles
        frac_trn (float, optional): Fraction of the dataset to put in the training set. Defaults to 0.7.
        frac_left_val (float, optional): Fration of the leftover dataset to be in the validation set. Defaults to 0.5.
        seed (int, optional): random seed. Defaults to 1.

    Returns:
        tuple: 
            - list: tile ids going to the training set
            - list: tile ids going to the validation set
            - list: tile ids going to the test set
    """

    trn_tiles_ids = tiles_df\
        .sample(frac=frac_trn, random_state=seed)\
        .id.astype(str).to_numpy().tolist()

    val_tiles_ids = tiles_df[~tiles_df.id.astype(str).isin(trn_tiles_ids)]\
        .sample(frac=frac_left_val, random_state=seed)\
        .id.astype(str).to_numpy().tolist()

    tst_tiles_ids = tiles_df[~tiles_df.id.astype(str).isin(trn_tiles_ids + val_tiles_ids)]\
        .id.astype(str).to_numpy().tolist()
    
    return trn_tiles_ids, val_tiles_ids, tst_tiles_ids


def extract_xyz(aoi_tiles_gdf):
    
    def _id_to_xyz(row):
        """
        Convert 'id' string to list of ints for x, y, z and t if eligeable
        """

        try:
            assert (row['id'].startswith('(')) & (row['id'].endswith(')')), 'The id should be surrounded by parenthesis.'
        except AssertionError as e:
            raise AssertionError(e)

        if 'year_tile' in row.keys(): 
            try:
                t, x, y, z = row['id'].lstrip('(,)').rstrip('(,)').split(',')
            except ValueError:
                raise ValueError(f"Could not extract t, x, y, z from tile ID {row['id']}.")
        else: 
            try:
                x, y, z = row['id'].lstrip('(,)').rstrip('(,)').split(',')
            except ValueError:
                raise ValueError(f"Could not extract x, y, z from tile ID {row['id']}.")
        
        # check whether x, y, z are ints
        assert str(int(x)) == str(x).strip(' '), "tile x coordinate is not actually integer"
        assert str(int(y)) == str(y).strip(' '), "tile y coordinate is not actually integer"
        assert str(int(z)) == str(z).strip(' '), "tile z coordinate is not actually integer"

        row['x'] = int(x)
        row['y'] = int(y)
        row['z'] = int(z)

        if 'year_tile' in row.keys():
            assert str(int(t)) == str(t).strip(' '), "tile t  year is not actually integer"
            row['t'] = int(t) 
        
        return row

    if 'id' not in aoi_tiles_gdf.columns.to_list():
        raise MissingIdException("No 'id' column was found in the AoI tiles dataset.")
    if len(aoi_tiles_gdf[aoi_tiles_gdf.id.duplicated()]) > 0:
        if 'year_tile' in aoi_tiles_gdf.keys():
            pass
        else:
            raise TileDuplicationException("The 'id' column in the AoI tiles dataset should not contain any duplicate.")
    
    return aoi_tiles_gdf.apply(_id_to_xyz, axis=1)


def assert_year(img_src, year, tiles_gdf):
    """Assert if the year of the dataset is well supported

    Args:
        img_src (string): image source
        year (float, int or string): the year option
        tiles_gdf (GeoDataframe): tiles geodataframe
    """

    if img_src=='XYZ' or img_src=='FOLDER':
        if year=='multi-year':
            if 'year_tile' in tiles_gdf.keys():
                pass
            else:
                logger.error("Option 'multi-year' chosen but the tile geodataframe does not contain a 'year' column. " 
                "Please add it while producing the tile geodataframe or set a numeric year in the configuration file.")
                sys.exit(1)
        elif str(year).isnumeric() and 'year_tile' in tiles_gdf.keys():
            logger.error("Option 'year' chosen but the tile geodataframe contains a 'year' column. " 
            "Please delete it while producing the tile geodataframe or set the 'multi-year' option in the configuration file.")
            sys.exit(1)
        elif 'year_tile' in tiles_gdf.keys():
            logger.error("Option 'year' not chosen but the tile geodataframe contains a 'year' column. " 
            "Please delete it while producing the tile geodataframe or set the 'multi-year' option in the configuration file.")
            sys.exit(1) 
    elif img_src=='WMS' or img_src=='MIL':
        if year:
            logger.warning("The connectors WMS and MIL do not support year information. The input year (config file or 'year' col in gdf) will be ignored.") 
        elif 'year_tile' in tiles_gdf.keys():
            logger.error("The connectors WMS and MIL do not support year information. Please provide a tile geodataframe without a 'year' column.")
            sys.exit(1) 
 

def intersect_labels_with_aoi(aoi_tiles_gdf, labels_gdf):
    """Check the crs of the two GDF and perform an inner sjoin.

    Args:
        aoi_tiles_gdf (GeoDataFrame): tiles of the area of interest
        labels_gdf (GeoDataFrame): labels

    Returns:
        tuple: 
            - aoi_tiles_intersecting_labels (GeoDataFrame): tiles of the area of interest intersecting the labels
            - id_list_tiles (list): id of the tiles intersecting a label
    """

    assert( aoi_tiles_gdf.crs == labels_gdf.crs )
    _aoi_tiles_gdf = aoi_tiles_gdf.copy()
    _labels_gdf = labels_gdf.copy()
    aoi_tiles_intersecting_labels = gpd.sjoin(_aoi_tiles_gdf, _labels_gdf, how='inner', predicate='intersects')
    aoi_tiles_intersecting_labels = aoi_tiles_intersecting_labels[_aoi_tiles_gdf.columns]
    aoi_tiles_intersecting_labels.drop_duplicates(inplace=True)
    id_list_tiles = aoi_tiles_intersecting_labels.id.to_numpy().tolist()

    return aoi_tiles_intersecting_labels, id_list_tiles


def split_additional_tiles(tiles_gdf, gt_tiles_gdf, trn_tiles_ids, val_tiles_ids, tst_tiles_ids, tile_type, frac_trn, seed):
        _tiles_gdf = tiles_gdf.copy()
        _gt_tiles_gdf = gt_tiles_gdf.copy()

        logger.info(f'Add {int(frac_trn * 100)}% of {tile_type} tiles to the trn, val and tst datasets')
        trn_fp_tiles_ids, val_fp_tiles_ids, tst_fp_tiles_ids = split_dataset(_tiles_gdf, frac_trn=frac_trn, seed=seed)

        # Add the FP tiles to the GT gdf 
        trn_tiles_ids.extend(trn_fp_tiles_ids)
        val_tiles_ids.extend(val_fp_tiles_ids)
        tst_tiles_ids.extend(tst_fp_tiles_ids)

        _gt_tiles_gdf = pd.concat([_gt_tiles_gdf, _tiles_gdf])

        return trn_tiles_ids, val_tiles_ids, tst_tiles_ids, _gt_tiles_gdf

def concat_sampled_tiles(limit, aoi_tiles_gdf, gt_tiles_gdf=gpd.GeoDataFrame(), fp_tiles_gdf=gpd.GeoDataFrame(), oth_tiles_gdf=gpd.GeoDataFrame(),
                    gt_factor=1//2, fp_factor=1//4, oth_factor=1//4):
    """Concatenate samples of geodataframe

    Args:
        limit (int): number of tiles selected in debug mode
        aoi_tiles_gdf (GeoDataFrame): tiles of the area of interest
        gt_tiles_gdf (GeoDataFrame): tiles intersecting GT labels
        fp_tiles_gdf (GeoDataFrame): tiles intersecting FP labels
        oth_tiles_gdf (GeoDataFrame): tiles intersecting OTH labels
        gt_factor (float): proportion of tiles selected amont gt tiles
        fp_factor (float): proportion of tiles selected amont fp tiles
        oth_factor (float): proportion of tiles selected amont oth tiles

    Returns:
        geodataframe
    """

    aoi_tiles_gdf = pd.concat([
        gt_tiles_gdf.head(limit * gt_factor), # a sample of tiles covering GT labels
        fp_tiles_gdf.head(limit * fp_factor), # a sample of tiles convering FP labels
        oth_tiles_gdf.head(limit * oth_factor), # a sample of tiles convering OTH labels
        aoi_tiles_gdf # the entire tileset, so as to also have tiles covering no label at all (duplicates will be dropped)
    ])

    return aoi_tiles_gdf


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
    
    # Get info for the download of tiles
    IM_SOURCE_TYPE = cfg['datasets']['image_source']['type'].upper()
    IM_SOURCE_LOCATION = cfg['datasets']['image_source']['location']
    if IM_SOURCE_TYPE != 'XYZ':
        IM_SOURCE_SRS = cfg['datasets']['image_source']['srs']
    else:
        IM_SOURCE_SRS = "EPSG:3857" # <- NOTE: this is hard-coded
    YEAR = cfg['datasets']['image_source']['year'] if 'year' in cfg['datasets']['image_source'].keys() else None
    if 'layers' in cfg['datasets']['image_source'].keys():
        IM_SOURCE_LAYERS = cfg['datasets']['image_source']['layers']

    AOI_TILES = cfg['datasets']['aoi_tiles']
       
    # Get info for labels if available
    GT_LABELS = cfg['datasets']['ground_truth_labels'] if 'ground_truth_labels' in cfg['datasets'].keys() else None
    OTH_LABELS = cfg['datasets']['other_labels'] if 'other_labels' in cfg['datasets'].keys() else None

    # Choose to add emtpy and FP tiles and get related info if necessary
    FP_LABELS = cfg['datasets']['fp_labels'] if 'fp_labels' in cfg['datasets'].keys() else False
    if FP_LABELS:
        FP_SHP = cfg['datasets']['fp_labels']['fp_shp'] if 'fp_shp' in cfg['datasets']['fp_labels'].keys() else None
        FP_FRAC_TRN = cfg['datasets']['fp_labels']['frac_trn'] if 'frac_trn' in cfg['datasets']['fp_labels'].keys() else 0.7
    EMPTY_TILES = cfg['empty_tiles'] if 'empty_tiles' in cfg.keys() else False
    if EMPTY_TILES:
        NB_TILES_FRAC = cfg['empty_tiles']['tiles_frac'] if 'tiles_frac' in cfg['empty_tiles'].keys() else 0.5
        EPT_FRAC_TRN = cfg['empty_tiles']['frac_trn'] if 'frac_trn' in cfg['empty_tiles'].keys() else 0.7
        OTH_TILES = cfg['empty_tiles']['keep_oth_tiles'] if 'keep_oth_tiles' in cfg['empty_tiles'].keys() else None
        
    SAVE_METADATA = True
    OVERWRITE = cfg['overwrite']
    if IM_SOURCE_TYPE not in ['XYZ', 'FOLDER']:
        TILE_SIZE = cfg['tile_size']
    else:
        TILE_SIZE = None
    N_JOBS = cfg['n_jobs']

    SEED = cfg['seed'] if 'seed' in cfg.keys() else False
    if SEED:
        logger.info(f'The seed is set to {SEED}.')

    if 'COCO_metadata' in cfg.keys():
        COCO_YEAR = cfg['COCO_metadata']['year']
        COCO_VERSION = cfg['COCO_metadata']['version']
        COCO_DESCRIPTION = cfg['COCO_metadata']['description']
        COCO_CONTRIBUTOR = cfg['COCO_metadata']['contributor']
        COCO_URL = cfg['COCO_metadata']['url']
        COCO_LICENSE_NAME = cfg['COCO_metadata']['license']['name']
        COCO_LICENSE_URL = cfg['COCO_metadata']['license']['url']
        COCO_CATEGORIES_FILE = cfg['COCO_metadata']['categories_file'] if 'categories_file' in cfg['COCO_metadata'].keys() else None

    os.chdir(WORKING_DIR)
    logger.info(f'Working_directory set to {WORKING_DIR}.')
    # let's make the output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    written_files = []

    # ------ Loading datasets
    logger.info("Loading AoI tiles as a GeoPandas DataFrame...")
    aoi_tiles_gdf = gpd.read_file(AOI_TILES)
    logger.success(f"{DONE_MSG} {len(aoi_tiles_gdf)} records were found.")
    if 'year' in aoi_tiles_gdf.keys(): 
        aoi_tiles_gdf = aoi_tiles_gdf.rename(columns={"year": "year_tile"})
        logger.info("Extracting tile coordinates (t, x, y, z) from tile IDs...")
    else:
        logger.info("Extracting tile coordinates (x, y, z) from tile IDs...")
    
    try:
        aoi_tiles_gdf = extract_xyz(aoi_tiles_gdf)
    except Exception as e:
        logger.critical(f"[...] Exception: {e}")
        sys.exit(1)
    logger.success(DONE_MSG)
    
    if GT_LABELS:
        logger.info("Loading Ground Truth Labels as a GeoPandas DataFrame...")
        gt_labels_gdf = gpd.read_file(GT_LABELS)
        logger.success(f"{DONE_MSG} {len(gt_labels_gdf)} records were found.")
        gt_labels_gdf = misc.find_category(gt_labels_gdf)
        if 'year' in gt_labels_gdf.keys(): 
            gt_labels_gdf = gt_labels_gdf.rename(columns={"year": "year_label"})

    if OTH_LABELS:
        logger.info("Loading Other Labels as a GeoPandas DataFrame...")
        oth_labels_gdf = gpd.read_file(OTH_LABELS)
        logger.success(f"{DONE_MSG} {len(oth_labels_gdf)} records were found.")
        if 'year' in oth_labels_gdf.keys(): 
            oth_labels_gdf = oth_labels_gdf.rename(columns={"year": "year_label"})

    if FP_LABELS:
        logger.info("Loading FP Labels as a GeoPandas DataFrame...")
        fp_labels_gdf = gpd.read_file(FP_SHP)
        logger.success(f"{DONE_MSG} {len(fp_labels_gdf)} records were found.")
        if 'year' in fp_labels_gdf.keys(): 
            fp_labels_gdf = fp_labels_gdf.rename(columns={"year": "year_label"})

    logger.info("Generating the list of tasks to be executed (one task per tile)...")

    if EMPTY_TILES or DEBUG_MODE:
        id_list_gt_tiles = []
        id_list_fp_tiles = []
        id_list_oth_tiles = []

        if GT_LABELS:
            aoi_tiles_intersecting_gt_labels, id_list_gt_tiles = intersect_labels_with_aoi(aoi_tiles_gdf, gt_labels_gdf)

        if FP_LABELS:
            aoi_tiles_intersecting_fp_labels, id_list_fp_tiles = intersect_labels_with_aoi(aoi_tiles_gdf, fp_labels_gdf)

        if OTH_LABELS:
            aoi_tiles_intersecting_oth_labels, id_list_oth_tiles = intersect_labels_with_aoi(aoi_tiles_gdf, oth_labels_gdf)
            
        # sampling tiles according to whether GT and/or OTH labels are provided
        if EMPTY_TILES:
            logger.info('Adding emtpy tiles to the datasets...')
            tmp_gdf = aoi_tiles_gdf.copy()
            tmp_gdf = tmp_gdf[~tmp_gdf['id'].isin(id_list_gt_tiles)] if GT_LABELS else tmp_gdf
            tmp_gdf = tmp_gdf[~tmp_gdf['id'].isin(id_list_fp_tiles)] if FP_LABELS else tmp_gdf
            all_emtpy_tiles_gdf = tmp_gdf[~tmp_gdf['id'].isin(id_list_oth_tiles)] if OTH_LABELS else tmp_gdf

            nb_gt_tiles = len(id_list_gt_tiles) if GT_LABELS else 0
            nb_fp_tiles = len(id_list_fp_tiles) if FP_LABELS else 0
            nb_oth_tiles = len(id_list_oth_tiles) if OTH_LABELS else 0
            id_list_ept_tiles = all_emtpy_tiles_gdf.id.to_numpy().tolist()
            nb_ept_tiles = len(id_list_ept_tiles)
            logger.info(f"- Number of tiles intersecting GT labels = {nb_gt_tiles}")
            logger.info(f"- Number of tiles intersecting FP labels = {nb_fp_tiles}")
            logger.info(f"- Number of tiles intersecting OTH labels = {nb_oth_tiles}")

            nb_frac_ept_tiles = int(NB_TILES_FRAC * (nb_gt_tiles - nb_fp_tiles))
            logger.info(f"- Add {int(NB_TILES_FRAC * 100)}% of GT tiles as empty tiles = {nb_frac_ept_tiles}")

            if nb_ept_tiles == 0:
                EMPTY_TILES = False 
                logger.warning("No empty tiles. No tiles added to the empty tile dataset")
            else:  
                if nb_frac_ept_tiles >= nb_ept_tiles:
                    nb_frac_ept_tiles = nb_ept_tiles
                    logger.warning(f"The number of empty tile available ({nb_ept_tiles}) is less than or equal to the ones to add ({nb_frac_ept_tiles}). The remaing tiles were attributed to the empty tiles dataset")
                empty_tiles_gdf = all_emtpy_tiles_gdf.sample(n=nb_frac_ept_tiles, random_state=1)
                id_list_ept_tiles = empty_tiles_gdf.id.to_numpy().tolist()

                id_keep_list_tiles = id_list_ept_tiles
                id_keep_list_tiles = id_keep_list_tiles + id_list_gt_tiles if GT_LABELS else id_keep_list_tiles
                id_keep_list_tiles = id_keep_list_tiles + id_list_fp_tiles if FP_LABELS else id_keep_list_tiles
                id_keep_list_tiles = id_keep_list_tiles + id_list_oth_tiles if OTH_LABELS else id_keep_list_tiles

                if OTH_TILES:                
                    logger.warning(f"Keep all tiles.")
                else:
                    logger.warning(f"Remove other tiles.")
                    aoi_tiles_gdf = aoi_tiles_gdf[aoi_tiles_gdf['id'].isin(id_keep_list_tiles)]

        if DEBUG_MODE:
            logger.warning(f"Debug mode: ON => Only {DEBUG_MODE_LIMIT} tiles will be processed.")

            # sampling tiles according to whether GT and/or OTH labels are provided

            if GT_LABELS and (FP_LABELS or OTH_LABELS):

                # Ensure that extending labels to not create duplicates in the tile selection
                nbr_duplicated_id = len(set(id_list_gt_tiles) & set(id_list_fp_tiles) & set(id_list_oth_tiles))

                if nbr_duplicated_id != 0:
                    initial_nbr_gt_tiles = aoi_tiles_intersecting_gt_labels.shape[0]
                    aoi_tiles_intersecting_gt_labels = aoi_tiles_intersecting_gt_labels[
                                                        ~aoi_tiles_intersecting_gt_labels['id'].isin(id_list_fp_tiles)]
                    aoi_tiles_intersecting_gt_labels = aoi_tiles_intersecting_gt_labels[
                                                        ~aoi_tiles_intersecting_gt_labels['id'].isin(id_list_oth_tiles)]
                    final_nbr_gt_tiles = aoi_tiles_intersecting_gt_labels.shape[0]

                    logger.warning(f'{nbr_duplicated_id} tiles were in common to the GT, OTH and FP datasets')
                    logger.warning(f'{initial_nbr_gt_tiles - final_nbr_gt_tiles} GT tiles were removed because of their presence in the FP or OTH dataset.')

                if FP_LABELS:
                    aoi_tiles_gdf = concat_sampled_tiles(DEBUG_MODE_LIMIT, aoi_tiles_gdf, aoi_tiles_intersecting_gt_labels, aoi_tiles_intersecting_fp_labels)
                elif OTH_LABELS:
                    aoi_tiles_gdf = concat_sampled_tiles(DEBUG_MODE_LIMIT, aoi_tiles_gdf, aoi_tiles_intersecting_gt_labels, aoi_tiles_intersecting_oth_labels)
                else:
                    aoi_tiles_gdf = concat_sampled_tiles(DEBUG_MODE_LIMIT, aoi_tiles_gdf, aoi_tiles_intersecting_gt_labels, aoi_tiles_intersecting_fp_labels, aoi_tiles_intersecting_oth_labels)
            
            elif GT_LABELS and not FP_LABELS and not OTH_LABELS:
                aoi_tiles_gdf = concat_sampled_tiles(DEBUG_MODE_LIMIT, aoi_tiles_gdf, aoi_tiles_intersecting_gt_labels, gt_factor=3//4)
            
            elif not GT_LABELS and not FP_LABELS and OTH_LABELS:
                aoi_tiles_gdf = concat_sampled_tiles(DEBUG_MODE_LIMIT, aoi_tiles_gdf, aoi_tiles_intersecting_oth_labels, oth_factor=3//4)
            
            else:
                pass # the following two lines of code would apply in this case
                
            aoi_tiles_gdf.drop_duplicates(inplace=True)
            aoi_tiles_gdf = aoi_tiles_gdf.head(DEBUG_MODE_LIMIT).copy()

    ALL_IMG_PATH = os.path.join(OUTPUT_DIR, f"all-images-{TILE_SIZE}" if TILE_SIZE else "all-images")

    if not os.path.exists(ALL_IMG_PATH):
        os.makedirs(ALL_IMG_PATH)

    if IM_SOURCE_TYPE == 'MIL':
        
        logger.info("(using the MIL connector)")

        assert_year(IM_SOURCE_TYPE, YEAR, aoi_tiles_gdf) 
        if YEAR:
            YEAR = None

        job_dict = MIL.get_job_dict(
            tiles_gdf=aoi_tiles_gdf.to_crs(IM_SOURCE_SRS), # <- note the reprojection
            mil_url=IM_SOURCE_LOCATION, 
            width=TILE_SIZE, 
            height=TILE_SIZE, 
            img_path=ALL_IMG_PATH, 
            image_sr=IM_SOURCE_SRS.split(":")[1], 
            save_metadata=SAVE_METADATA,
            overwrite=OVERWRITE
        )

        image_getter = MIL.get_geotiff

    elif IM_SOURCE_TYPE == 'WMS':
        
        logger.info("(using the WMS connector)")

        assert_year(IM_SOURCE_TYPE, YEAR, aoi_tiles_gdf) 
        if YEAR:
            YEAR = None

        job_dict = WMS.get_job_dict(
            tiles_gdf=aoi_tiles_gdf.to_crs(IM_SOURCE_SRS), # <- note the reprojection
            wms_url=IM_SOURCE_LOCATION, 
            layers=IM_SOURCE_LAYERS,
            width=TILE_SIZE, 
            height=TILE_SIZE, 
            img_path=ALL_IMG_PATH, 
            srs=IM_SOURCE_SRS, 
            save_metadata=SAVE_METADATA,
            overwrite=OVERWRITE
        )

        image_getter = WMS.get_geotiff

    elif IM_SOURCE_TYPE == 'XYZ':
        
        logger.info("(using the XYZ connector)")

        assert_year(IM_SOURCE_TYPE, YEAR, aoi_tiles_gdf)    

        job_dict = XYZ.get_job_dict(
            tiles_gdf=aoi_tiles_gdf.to_crs(IM_SOURCE_SRS), # <- note the reprojection
            xyz_url=IM_SOURCE_LOCATION, 
            img_path=ALL_IMG_PATH, 
            year=YEAR,
            save_metadata=SAVE_METADATA,
            overwrite=OVERWRITE
        )

        image_getter = XYZ.get_geotiff

    elif IM_SOURCE_TYPE == 'FOLDER':

        logger.info(f'(using the files in the folder "{IM_SOURCE_LOCATION}")')

        assert_year(IM_SOURCE_TYPE, YEAR, aoi_tiles_gdf)
            
        job_dict = FOLDER.get_job_dict(
            tiles_gdf=aoi_tiles_gdf.to_crs(IM_SOURCE_SRS), # <- note the reprojection
            base_path=IM_SOURCE_LOCATION, 
            end_path=ALL_IMG_PATH, 
            year=YEAR,
            save_metadata=SAVE_METADATA,
            overwrite=OVERWRITE
        )

        image_getter = FOLDER.get_image_to_folder

    else:
        logger.critical(f'Web Services of type "{IM_SOURCE_TYPE}" are not supported. Exiting.')
        sys.exit(1)

    logger.success(DONE_MSG)

    logger.info(f"Executing tasks, {N_JOBS} at a time...")
    job_outcome = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(image_getter)(**v) for k, v in tqdm( sorted(list(job_dict.items())) )
    )
    logger.info("Checking whether all the expected tiles were actually downloaded...")

    all_tiles_were_downloaded = True
    for job in job_dict.keys():
        if not os.path.isfile(job) or not os.path.isfile(job.replace('.tif', '.json')):
            all_tiles_were_downloaded = False
            logger.warning(f"Failed job: {job}")

    if all_tiles_were_downloaded:
        logger.success(DONE_MSG)
    else:
        logger.critical("Some tiles were not downloaded. Please try to run this script again.")
        sys.exit(1)


    # ------ Collecting image metadata, to be used when assessing detections

    logger.info("Collecting image metadata...")

    md_files = [f for f in os.listdir(ALL_IMG_PATH) if os.path.isfile(os.path.join(ALL_IMG_PATH, f)) and f.endswith('.json')]
    
    img_metadata_list = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(read_img_metadata)(md_file, ALL_IMG_PATH) for md_file in tqdm(md_files))
    img_metadata_dict = { k: v for img_md in img_metadata_list for (k, v) in img_md.items() }

    if YEAR:
        for key, value in job_dict.items():
            img_metadata_dict[key]['year_img'] = job_dict[key]['year']

    # let's save metadata... (kind of an image catalog)
    IMG_METADATA_FILE = os.path.join(OUTPUT_DIR, 'img_metadata.json')
    with open(IMG_METADATA_FILE, 'w') as fp:
        json.dump(img_metadata_dict, fp)

    written_files.append(IMG_METADATA_FILE)
    logger.success(f"{DONE_MSG} A file was written: {IMG_METADATA_FILE}")    


    # ------ Training/validation/test/other dataset generation
    if GT_LABELS:
        try:
            assert( aoi_tiles_gdf.crs == gt_labels_gdf.crs ), "CRS Mismatch between AoI tiles and labels."
        except Exception as e:
            logger.critical(e)
            sys.exit(1)

        gt_tiles_gdf = gpd.sjoin(aoi_tiles_gdf, gt_labels_gdf, how='inner', predicate='intersects')
    
        # get the number of labels per class
        labels_per_class_dict = {}
        for category in gt_tiles_gdf.CATEGORY.unique():
            labels_per_class_dict[category] = gt_tiles_gdf[gt_tiles_gdf.CATEGORY == category].shape[0]
        # Get the number of labels per tile
        labels_per_tiles_gdf = gt_tiles_gdf.groupby(['id', 'CATEGORY'], as_index=False).size()

        gt_tiles_gdf = gt_tiles_gdf.drop_duplicates(subset=aoi_tiles_gdf.columns)
        gt_tiles_gdf.drop(columns=['index_right'], inplace=True)

        # Get the tiles containing at least one "FP" label but no "GT" label (if applicable)
        if FP_LABELS:
            tmp_fp_tiles_gdf, _ = intersect_labels_with_aoi(aoi_tiles_gdf, fp_labels_gdf)
            fp_tiles_gdf = tmp_fp_tiles_gdf[~tmp_fp_tiles_gdf.id.astype(str).isin(gt_tiles_gdf.id.astype(str))].copy()
            del tmp_fp_tiles_gdf
        else:
            fp_tiles_gdf = gpd.GeoDataFrame(columns=['id'])

        # remove tiles including at least one "oth" label (if applicable)
        if OTH_LABELS:
            oth_tiles_to_remove_gdf, _ = intersect_labels_with_aoi(gt_tiles_gdf, oth_labels_gdf)
            gt_tiles_gdf = gt_tiles_gdf[~gt_tiles_gdf.id.astype(str).isin(oth_tiles_to_remove_gdf.id.astype(str))].copy()
            del oth_tiles_to_remove_gdf

        # add ramdom tiles not intersecting labels to the dataset 
        oth_tiles_gdf = aoi_tiles_gdf[~aoi_tiles_gdf.id.astype(str).isin(gt_tiles_gdf.id.astype(str))].copy()
        oth_tiles_gdf = oth_tiles_gdf[~oth_tiles_gdf.id.astype(str).isin(fp_tiles_gdf.id.astype(str))].copy()

        # OTH tiles = AoI tiles with labels, but which are not GT
        if EMPTY_TILES:           
            empty_tiles_gdf = aoi_tiles_gdf[aoi_tiles_gdf.id.astype(str).isin(id_list_ept_tiles)].copy()

            if DEBUG_MODE:
                try:
                    assert(len(empty_tiles_gdf != 0))
                except AssertionError:
                    logger.error("No emtpy tile was selected for the debug mode. Increase the number of sampled tiles in debug mode")
                    exit(1)
            
            oth_tiles_gdf = oth_tiles_gdf[~oth_tiles_gdf.id.astype(str).isin(empty_tiles_gdf.id.astype(str))].copy()
            oth_tiles_gdf['dataset'] = 'oth'
            assert( len(aoi_tiles_gdf) == len(gt_tiles_gdf) + len(fp_tiles_gdf) + len(empty_tiles_gdf) + len(oth_tiles_gdf) )
        else: 
            oth_tiles_gdf['dataset'] = 'oth'
            assert( len(aoi_tiles_gdf) == len(gt_tiles_gdf) + len(fp_tiles_gdf) + len(oth_tiles_gdf) )
        
        # 70%, 15%, 15% split
        categories_arr = labels_per_tiles_gdf.CATEGORY.unique()
        categories_arr.sort()
        if not SEED:
            max_seed = 50
            best_split = 0
            for seed in tqdm(range(max_seed), desc='Test seeds for splitting tiles between datasets'):
                ok_split = 0
                trn_tiles_ids, val_tiles_ids, tst_tiles_ids = split_dataset(gt_tiles_gdf, seed=seed)
                
                for category in categories_arr:
                    
                    ratio_trn = labels_per_tiles_gdf.loc[
                        (labels_per_tiles_gdf.CATEGORY == category) & labels_per_tiles_gdf.id.astype(str).isin(trn_tiles_ids), 'size'
                    ].sum() / labels_per_class_dict[category]
                    ratio_val = labels_per_tiles_gdf.loc[
                        (labels_per_tiles_gdf.CATEGORY == category) & labels_per_tiles_gdf.id.astype(str).isin(val_tiles_ids), 'size'
                    ].sum() / labels_per_class_dict[category]
                    ratio_tst = labels_per_tiles_gdf.loc[
                        (labels_per_tiles_gdf.CATEGORY == category) & labels_per_tiles_gdf.id.astype(str).isin(tst_tiles_ids), 'size'
                    ].sum() / labels_per_class_dict[category]

                    ok_split = ok_split + 1 if ratio_trn >= 0.60 else ok_split
                    ok_split = ok_split + 1 if ratio_val >= 0.12 else ok_split
                    ok_split = ok_split + 1 if ratio_tst >= 0.12 else ok_split

                    ok_split = ok_split - 1 if 0 in [ratio_trn, ratio_val, ratio_tst] else ok_split
                
                if ok_split == len(categories_arr)*3:
                    logger.info(f'A seed of {seed} produces a good repartition of the labels.')
                    SEED = seed
                    break
                elif ok_split > best_split:
                    SEED = seed
                    best_split = ok_split
                
                if seed == max_seed-1:
                    logger.warning(f'No satisfying seed found between 0 and {max_seed}.')
                    logger.info(f'The best seed was {SEED} with ~{best_split} class subsets containing the correct proportion (trn~0.7, val~0.15, tst~0.15).')
                    logger.info('The user should set a seed manually if not satisfied.')

        else:
            trn_tiles_ids, val_tiles_ids, tst_tiles_ids = split_dataset(gt_tiles_gdf, seed=SEED)
        
        if FP_LABELS:
            trn_tiles_ids, val_tiles_ids, tst_tiles_ids, gt_tiles_gdf = split_additional_tiles(
                fp_tiles_gdf, gt_tiles_gdf, trn_tiles_ids, val_tiles_ids, tst_tiles_ids, 'FP', FP_FRAC_TRN, SEED
            )
            del fp_tiles_gdf
        if EMPTY_TILES:
            trn_tiles_ids, val_tiles_ids, tst_tiles_ids, gt_tiles_gdf = split_additional_tiles(
                empty_tiles_gdf, gt_tiles_gdf, trn_tiles_ids, val_tiles_ids, tst_tiles_ids, 'emtpy', EPT_FRAC_TRN, SEED
            )
            del empty_tiles_gdf

        for df in [gt_tiles_gdf, labels_per_tiles_gdf]:
            df.loc[df.id.astype(str).isin(trn_tiles_ids), 'dataset'] = 'trn'
            df.loc[df.id.astype(str).isin(val_tiles_ids), 'dataset'] = 'val'
            df.loc[df.id.astype(str).isin(tst_tiles_ids), 'dataset'] = 'tst'

        logger.info('Repartition in the datasets by category:')
        for dst in ['trn', 'val', 'tst']:
            for category in categories_arr:
                row_ids = labels_per_tiles_gdf.index[(labels_per_tiles_gdf.dataset==dst) & (labels_per_tiles_gdf.CATEGORY==category)]
                logger.info(f'   {category} labels in {dst} dataset: {labels_per_tiles_gdf.loc[labels_per_tiles_gdf.index.isin(row_ids), "size"].sum()}')

        # remove columns generated by the Spatial Join
        gt_tiles_gdf = gt_tiles_gdf[aoi_tiles_gdf.columns.tolist() + ['dataset']].copy()

        assert( len(gt_tiles_gdf) == len(trn_tiles_ids) + len(val_tiles_ids) + len(tst_tiles_ids) ), \
            'Tiles were lost in the split between training, validation and test sets.'
        
        split_aoi_tiles_gdf = pd.concat(
            [
                gt_tiles_gdf,
                oth_tiles_gdf
            ]
        )

        # let's free up some memory
        del gt_tiles_gdf
        del oth_tiles_gdf
         
    else:
        split_aoi_tiles_gdf = aoi_tiles_gdf.copy()
        split_aoi_tiles_gdf['dataset'] = 'oth'
        
        
    assert( len(split_aoi_tiles_gdf) == len(aoi_tiles_gdf) ) # it means that all the tiles were actually used
    
    
    SPLIT_AOI_TILES = os.path.join(OUTPUT_DIR, 'split_aoi_tiles.geojson')

    try:
        split_aoi_tiles_gdf.to_file(SPLIT_AOI_TILES, driver='GeoJSON')
    except Exception as e:
        logger.error(e)
    written_files.append(SPLIT_AOI_TILES)
    logger.success(f'{DONE_MSG} A file was written {SPLIT_AOI_TILES}')

    img_md_df = pd.DataFrame.from_dict(img_metadata_dict, orient='index')
    img_md_df.reset_index(inplace=True)
    img_md_df.rename(columns={"index": "img_file"}, inplace=True)

    img_md_df['id'] = img_md_df.apply(misc.img_md_record_to_tile_id, axis=1)

    split_aoi_tiles_with_img_md_gdf = split_aoi_tiles_gdf.merge(img_md_df, on='id', how='left')
    for dst in split_aoi_tiles_with_img_md_gdf.dataset.to_numpy():
        os.makedirs(os.path.join(OUTPUT_DIR, f'{dst}-images{f"-{TILE_SIZE}" if TILE_SIZE else ""}'), exist_ok=True)

    split_aoi_tiles_with_img_md_gdf['dst_file'] = [
        src_file.replace('all', dataset) 
        for src_file, dataset in zip(split_aoi_tiles_with_img_md_gdf.img_file, split_aoi_tiles_with_img_md_gdf.dataset)
    ]
    for src_file, dst_file in zip(split_aoi_tiles_with_img_md_gdf.img_file, split_aoi_tiles_with_img_md_gdf.dst_file):
        misc.make_hard_link(src_file, dst_file)

    # ------ Generating COCO annotations
    
    if GT_LABELS and OTH_LABELS:
        
        assert(gt_labels_gdf.crs == oth_labels_gdf.crs)
        
        labels_gdf = pd.concat([
            gt_labels_gdf,
            oth_labels_gdf
        ]).reset_index()

    elif GT_LABELS and not OTH_LABELS:
        labels_gdf = gt_labels_gdf.copy().reset_index()
    elif not GT_LABELS and OTH_LABELS:
        labels_gdf = oth_labels_gdf.copy().reset_index()
    else:
        labels_gdf = gpd.GeoDataFrame()

    if 'COCO_metadata' not in cfg.keys():
        print()
        toc = time.time()
        logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

        sys.stderr.flush()
        sys.exit(0)
        
    
    if len(labels_gdf) > 0:
        # Get possibles combination for category and supercategory
        combinations_category_dict = labels_gdf.groupby(['CATEGORY', 'SUPERCATEGORY'], as_index=False).size().drop(columns=['size']).to_dict('tight')
        combinations_category_lists = combinations_category_dict['data']

    elif 'category' in cfg['COCO_metadata'].keys():
        combinations_category_lists = [[cfg['COCO_metadata']['category']['name'], cfg['COCO_metadata']['category']['supercategory']]]

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
    for category, supercategory in combinations_category_lists:
        logger.info(f"    - {category}, {supercategory}")

    # Put categories in coco objects and keep them in a dict
    coco_categories = {}
    for category, supercategory in combinations_category_lists:
        
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