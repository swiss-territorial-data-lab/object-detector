#!/bin/python
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import json
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
from helpers import misc
from helpers.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


class MissingIdException(Exception):
    "Raised when tiles are lacking IDs"
    pass


class TileDuplicationException(Exception):
    "Raised when the 'id' column contains duplicates"
    pass


def assert_year_for_tiles(img_src, year, tiles_gdf):
    """Assert if the year of the dataset is well supported

    Args:
        img_src (string): image source
        year (int or string): the year option
        tiles_gdf (GeoDataframe): tiles geodataframe
    """

    if img_src=='XYZ' or img_src=='FOLDER':
        if year=='multi-year':
            if 'year_tile' not in tiles_gdf.keys():
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


def concat_sampled_tiles(limit, aoi_tiles_gdf, gt_tiles_gdf=gpd.GeoDataFrame(), fp_tiles_gdf=gpd.GeoDataFrame(), oth_tiles_gdf=gpd.GeoDataFrame(),
                    gt_factor=1//2, fp_factor=1//4, oth_factor=1//4):
    """Concatenate samples of geodataframe

    Args:
        limit (int): number of tiles selected in debug mode
        aoi_tiles_gdf (GeoDataFrame): tiles of the area of interest
        gt_tiles_gdf (GeoDataFrame): tiles intersecting GT labels
        fp_tiles_gdf (GeoDataFrame): tiles intersecting FP labels
        oth_tiles_gdf (GeoDataFrame): tiles intersecting OTH labels
        gt_factor (float): proportion of tiles selected among gt tiles
        fp_factor (float): proportion of tiles selected among fp tiles
        oth_factor (float): proportion of tiles selected among oth tiles

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


def extract_xyz(aoi_tiles_gdf):
    
    def _id_to_xyz(row):
        """
        Convert 'id' string to list of ints for x, y, z and t if eligeable
        """

        assert (row['id'].startswith('(')) & (row['id'].endswith(')')), 'The id should be surrounded by parenthesis.'

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
        raise TileDuplicationException("The 'id' column in the AoI tiles dataset should not contain any duplicate.")
    
    return aoi_tiles_gdf.apply(_id_to_xyz, axis=1)


def read_img_metadata(md_file, all_img_path):
    # Read images metadata and return them as dictionnaries with the image path as key.
    img_path = os.path.join(all_img_path, md_file.replace('json', 'tif'))
    
    with open(os.path.join(all_img_path, md_file), 'r') as fp:
        return {img_path: json.load(fp)}


def download_tiles(datasets_dict, gt_labels_gdf, oth_labels_gdf, fp_labels_gdf, empty_tiles_dict, tile_size, n_jobs, 
                   output_dir, debug_mode, debug_mode_limit, overwrite):

    # Get tile download information
    IM_SOURCE_TYPE = datasets_dict['image_source']['type'].upper()
    IM_SOURCE_LOCATION = datasets_dict['image_source']['location']
    if IM_SOURCE_TYPE != 'XYZ':
        IM_SOURCE_SRS = datasets_dict['image_source']['srs']
    else:
        IM_SOURCE_SRS = "EPSG:3857" # <- NOTE: this is hard-coded
    YEAR = datasets_dict['image_source']['year'] if 'year' in datasets_dict['image_source'].keys() else None
    SAVE_METADATA = True

    AOI_TILES = datasets_dict['aoi_tiles']
    
    GT_LABELS = False if gt_labels_gdf.empty else True
    OTH_LABELS = False if oth_labels_gdf.empty else True
    FP_LABELS = False if fp_labels_gdf.empty else True

    written_files = []
    id_list_ept_tiles = []

    # ------ Loading datasets
    logger.info("Loading AoI tiles as a GeoPandas DataFrame...")
    aoi_tiles_gdf = gpd.read_file(AOI_TILES)
    logger.success(f"{DONE_MSG} {len(aoi_tiles_gdf)} records were found.")
    if 'year' in aoi_tiles_gdf.keys(): 
        aoi_tiles_gdf['year'] = aoi_tiles_gdf.year.astype(int)
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
    

    logger.info("Generating the list of tasks to be executed (one task per tile)...")

    if empty_tiles_dict or debug_mode:
        id_list_gt_tiles = []
        id_list_fp_tiles = []
        id_list_oth_tiles = []

        if GT_LABELS:
            aoi_tiles_intersecting_gt_labels, id_list_gt_tiles = misc.intersect_labels_with_aoi(aoi_tiles_gdf, gt_labels_gdf)

        if FP_LABELS:
            aoi_tiles_intersecting_fp_labels, id_list_fp_tiles = misc.intersect_labels_with_aoi(aoi_tiles_gdf, fp_labels_gdf)

        if OTH_LABELS:
            aoi_tiles_intersecting_oth_labels, id_list_oth_tiles = misc.intersect_labels_with_aoi(aoi_tiles_gdf, oth_labels_gdf)
            
        # sampling tiles according to whether GT and/or OTH labels are provided
        if empty_tiles_dict:
            NB_TILES_FRAC = empty_tiles_dict['tiles_frac'] if 'tiles_frac' in empty_tiles_dict.keys() else 0.5
            OTH_TILES = empty_tiles_dict['keep_oth_tiles'] if 'keep_oth_tiles' in empty_tiles_dict.keys() else True

            logger.info('Adding empty tiles to the datasets...')
            label_tiles = \
                (id_list_gt_tiles if GT_LABELS else [])\
                + (id_list_fp_tiles if FP_LABELS else [])\
                + (id_list_oth_tiles if OTH_LABELS else [])
            all_empty_tiles_gdf = aoi_tiles_gdf[~aoi_tiles_gdf['id'].isin(label_tiles)].copy()

            nb_gt_tiles = len(id_list_gt_tiles) if GT_LABELS else 0
            nb_fp_tiles = len(id_list_fp_tiles) if FP_LABELS else 0
            nb_oth_tiles = len(id_list_oth_tiles) if OTH_LABELS else 0
            id_list_ept_tiles = all_empty_tiles_gdf.id.to_numpy().tolist()
            nb_ept_tiles = len(id_list_ept_tiles)
            logger.info(f"- Number of tiles intersecting GT labels = {nb_gt_tiles}")
            logger.info(f"- Number of tiles intersecting FP labels = {nb_fp_tiles}")
            logger.info(f"- Number of tiles intersecting OTH labels = {nb_oth_tiles}")

            nb_frac_ept_tiles = int(NB_TILES_FRAC * (nb_gt_tiles - nb_fp_tiles))
            logger.info(f"- Add {int(NB_TILES_FRAC * 100)}% of GT tiles as empty tiles = {nb_frac_ept_tiles}")

            if nb_ept_tiles == 0:
                empty_tiles_dict = False 
                logger.warning("No empty tiles. No tiles added to the empty tile dataset.")
            else:  
                if nb_frac_ept_tiles >= nb_ept_tiles:
                    nb_frac_ept_tiles = nb_ept_tiles
                    logger.warning(
                        f"The number of empty tile available ({nb_ept_tiles}) is less than or equal to the ones to add ({nb_frac_ept_tiles}). The remaing tiles were attributed to the empty tiles dataset"
                    )
                empty_tiles_gdf = all_empty_tiles_gdf.sample(n=nb_frac_ept_tiles, random_state=1)
                id_list_ept_tiles = empty_tiles_gdf.id.to_numpy().tolist()

                if OTH_TILES:                
                    logger.warning(f"Keep all tiles.")
                else:
                    logger.warning(f"Remove other tiles.")
                    id_keep_list_tiles = id_list_ept_tiles
                    id_keep_list_tiles = id_keep_list_tiles + id_list_gt_tiles if GT_LABELS else id_keep_list_tiles
                    id_keep_list_tiles = id_keep_list_tiles + id_list_fp_tiles if FP_LABELS else id_keep_list_tiles
                    id_keep_list_tiles = id_keep_list_tiles + id_list_oth_tiles if OTH_LABELS else id_keep_list_tiles
                    aoi_tiles_gdf = aoi_tiles_gdf[aoi_tiles_gdf['id'].isin(id_keep_list_tiles)]

        if debug_mode:
            logger.warning(f"Debug mode: ON => Only {debug_mode_limit} tiles will be processed.")

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
                    aoi_tiles_gdf = concat_sampled_tiles(
                        debug_mode_limit, aoi_tiles_gdf, aoi_tiles_intersecting_gt_labels, fp_tiles_gdf = aoi_tiles_intersecting_fp_labels
                    )
                if OTH_LABELS:
                    aoi_tiles_gdf = concat_sampled_tiles(
                        debug_mode_limit, aoi_tiles_gdf, aoi_tiles_intersecting_gt_labels, oth_tiles_gdf=aoi_tiles_intersecting_oth_labels
                    )
            
            elif GT_LABELS and not FP_LABELS and not OTH_LABELS:
                aoi_tiles_gdf = concat_sampled_tiles(debug_mode_limit, aoi_tiles_gdf, aoi_tiles_intersecting_gt_labels, gt_factor=3//4)
            
            elif not GT_LABELS and not FP_LABELS and OTH_LABELS:
                aoi_tiles_gdf = concat_sampled_tiles(debug_mode_limit, aoi_tiles_gdf, aoi_tiles_intersecting_oth_labels, oth_factor=3//4)
                
            aoi_tiles_gdf.drop_duplicates(inplace=True)
            aoi_tiles_gdf = aoi_tiles_gdf.head(debug_mode_limit).copy()

    ALL_IMG_PATH = os.path.join(output_dir, f"all-images-{tile_size}" if tile_size else "all-images")
    base_infos_dict = {
        'tiles_gdf': aoi_tiles_gdf.to_crs(IM_SOURCE_SRS), # <- note the reprojection
        'img_path': ALL_IMG_PATH,
        'save_metadata': SAVE_METADATA,
        'overwrite': overwrite
    }
    assert_year_for_tiles(IM_SOURCE_TYPE, YEAR, aoi_tiles_gdf)

    os.makedirs(ALL_IMG_PATH, exist_ok=True)

    if IM_SOURCE_TYPE in ['MIL', 'WMS'] and YEAR:
        YEAR = None

    if IM_SOURCE_TYPE == 'MIL':
        
        logger.info("(using the MIL connector)")

        job_dict = MIL.get_job_dict(
            mil_url=IM_SOURCE_LOCATION, 
            width=tile_size, 
            height=tile_size, 
            image_sr=IM_SOURCE_SRS.split(":")[1], 
            **base_infos_dict
        )

        image_getter = MIL.get_geotiff

    elif IM_SOURCE_TYPE == 'WMS':
        
        logger.info("(using the WMS connector)")
        IM_SOURCE_LAYERS = datasets_dict['image_source']['layers']

        job_dict = WMS.get_job_dict(
            wms_url=IM_SOURCE_LOCATION, 
            layers=IM_SOURCE_LAYERS,
            width=tile_size, 
            height=tile_size, 
            srs=IM_SOURCE_SRS, 
            **base_infos_dict
        )

        image_getter = WMS.get_geotiff

    elif IM_SOURCE_TYPE == 'XYZ':
        
        logger.info("(using the XYZ connector)")

        job_dict = XYZ.get_job_dict(
            xyz_url=IM_SOURCE_LOCATION, 
            year=YEAR,
            **base_infos_dict
        )

        image_getter = XYZ.get_geotiff

    elif IM_SOURCE_TYPE == 'FOLDER':

        logger.info(f'(using the files in the folder "{IM_SOURCE_LOCATION}")')
            
        job_dict = FOLDER.get_job_dict(
            base_path=IM_SOURCE_LOCATION, 
            year=YEAR,
            **base_infos_dict
        )

        image_getter = FOLDER.get_image_to_folder

    else:
        logger.critical(f'Web Services of type "{IM_SOURCE_TYPE}" are not supported. Exiting.')
        sys.exit(1)

    logger.success(DONE_MSG)

    logger.info(f"Executing tasks, {n_jobs} at a time...")
    _ = Parallel(n_jobs=n_jobs, backend="loky")(
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
    
    img_metadata_list = Parallel(n_jobs=n_jobs, backend="loky")(delayed(read_img_metadata)(md_file, ALL_IMG_PATH) for md_file in tqdm(md_files))
    img_metadata_dict = { k: v for img_md in img_metadata_list for (k, v) in img_md.items() }

    # let's save metadata... (kind of an image catalog)
    IMG_METADATA_FILE = os.path.join(output_dir, 'img_metadata.json')
    with open(IMG_METADATA_FILE, 'w') as fp:
        json.dump(img_metadata_dict, fp)

    written_files.append(IMG_METADATA_FILE)
    logger.success(f"{DONE_MSG} A file was written: {IMG_METADATA_FILE}")

    return aoi_tiles_gdf, img_metadata_dict, id_list_ept_tiles, written_files