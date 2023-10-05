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

    coco_image = coco_obj.image(output_dir, this_tile_dirname, coco_license_id)
    segmentations = []
    
    if len(labels) > 0:
        
        xmin, ymin, xmax, ymax = [float(x) for x in MIL.bounds_to_bbox(_tile['geometry'].bounds).split(',')]
        
        # note the .explode() which turns Multipolygon into Polygons
        clipped_labels_gdf = gpd.clip(labels, _tile['geometry']).explode()

        for label in clipped_labels_gdf.itertuples():
            scaled_poly = misc.scale_polygon(label.geometry, xmin, ymin, xmax, ymax, 
                                             coco_image['width'], coco_image['height'])
            scaled_poly = scaled_poly[:-1] # let's remove the last point

            segmentation = misc.my_unpack(scaled_poly)

            try:
                assert(min(segmentation) >= 0)
                assert(max(segmentation) <= min(coco_image['width'], coco_image['height']))
            except AssertionError:
                raise LabelOverflowException(f"Label boundaries exceed tile size - Tile ID = {_tile['id']}")
            
            # Category attribution
            key = str(label.CATEGORY) + '_' + str(label.SUPERCATEGORY)
            category_id = coco_category[key]['id']
                
            segmentations.append(segmentation)
            
    return (coco_image, category_id, segmentations)

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
        convert 'id' string to list of ints for x,y,z
        """

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
        
        return row

    if 'id' not in aoi_tiles_gdf.columns.to_list():
        raise MissingIdException("No 'id' column was found in the AoI tiles dataset.")
    if len(aoi_tiles_gdf[aoi_tiles_gdf.id.duplicated()]) > 0:
        raise TileDuplicationException("The 'id' column in the AoI tiles dataset should not contain any duplicate.")
    
    return aoi_tiles_gdf.apply(_id_to_xyz, axis=1)


def main(cfg_file_path):

    tic = time.time()
    logger.info('Starting...')

    logger.info(f"Using {cfg_file_path} as config file.")

    with open(cfg_file_path) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    DEBUG_MODE = cfg['debug_mode']
    
    WORKING_DIR = cfg['working_directory']
    OUTPUT_DIR = cfg['output_folder']
    
    ORTHO_WS_TYPE = cfg['datasets']['orthophotos_web_service']['type']
    ORTHO_WS_URL = cfg['datasets']['orthophotos_web_service']['url']
    if ORTHO_WS_TYPE != 'XYZ':
        ORTHO_WS_SRS = cfg['datasets']['orthophotos_web_service']['srs']
    else:
        ORTHO_WS_SRS = "EPSG:3857" # <- NOTE: this is hard-coded
    if 'layers' in cfg['datasets']['orthophotos_web_service'].keys():
        ORTHO_WS_LAYERS = cfg['datasets']['orthophotos_web_service']['layers']

    AOI_TILES_GEOJSON = cfg['datasets']['aoi_tiles_geojson']
    
    if 'ground_truth_labels_geojson' in cfg['datasets'].keys():
        GT_LABELS_GEOJSON = cfg['datasets']['ground_truth_labels_geojson']
    else:
        GT_LABELS_GEOJSON = None
    if 'other_labels_geojson' in cfg['datasets'].keys():
        OTH_LABELS_GEOJSON = cfg['datasets']['other_labels_geojson']
    else:
        OTH_LABELS_GEOJSON = None

    SAVE_METADATA = True
    OVERWRITE = cfg['overwrite']
    if ORTHO_WS_TYPE != 'XYZ':
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
    else:
        COCO_YEAR=None


    os.chdir(WORKING_DIR)
    logger.info(f'Working_directory set to {WORKING_DIR}.')
    # let's make the output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    written_files = []

    # ------ Loading datasets

    logger.info("Loading AoI tiles as a GeoPandas DataFrame...")
    aoi_tiles_gdf = gpd.read_file(AOI_TILES_GEOJSON)
    logger.success(f"{DONE_MSG} {len(aoi_tiles_gdf)} records were found.")

    logger.info("Extracting tile coordinates (x, y, z) from tile IDs...")
    try:
        aoi_tiles_gdf = extract_xyz(aoi_tiles_gdf)
    except Exception as e:
        logger.critical(f"[...] Exception: {e}")
        sys.exit(1)
    logger.success(DONE_MSG)
    
    if GT_LABELS_GEOJSON:
        logger.info("Loading Ground Truth Labels as a GeoPandas DataFrame...")
        gt_labels_gdf = gpd.read_file(GT_LABELS_GEOJSON)
        logger.success(f"{DONE_MSG} {len(gt_labels_gdf)} records were found.")

    if OTH_LABELS_GEOJSON:
        logger.info("Loading Other Labels as a GeoPandas DataFrame...")
        oth_labels_gdf = gpd.read_file(OTH_LABELS_GEOJSON)
        logger.success(f"{DONE_MSG} {len(oth_labels_gdf)} records were found.")

    logger.info("Generating the list of tasks to be executed (one task per tile)...")

    DEBUG_MODE_LIMIT = 100
    if DEBUG_MODE:
        logger.warning(f"Debug mode: ON => Only {DEBUG_MODE_LIMIT} tiles will be processed.")

        if GT_LABELS_GEOJSON:
            assert( aoi_tiles_gdf.crs == gt_labels_gdf.crs )
            aoi_tiles_intersecting_gt_labels = gpd.sjoin(aoi_tiles_gdf, gt_labels_gdf, how='inner', predicate='intersects')
            aoi_tiles_intersecting_gt_labels = aoi_tiles_intersecting_gt_labels[aoi_tiles_gdf.columns]
            aoi_tiles_intersecting_gt_labels.drop_duplicates(inplace=True)

        if OTH_LABELS_GEOJSON:
            assert( aoi_tiles_gdf.crs == oth_labels_gdf.crs )
            aoi_tiles_intersecting_oth_labels = gpd.sjoin(aoi_tiles_gdf, oth_labels_gdf, how='inner', predicate='intersects')
            aoi_tiles_intersecting_oth_labels = aoi_tiles_intersecting_oth_labels[aoi_tiles_gdf.columns]
            aoi_tiles_intersecting_oth_labels.drop_duplicates(inplace=True)
            
        # sampling tiles according to whether GT and/or GT labels are provided
        if GT_LABELS_GEOJSON and OTH_LABELS_GEOJSON:

            # Ensure that extending labels to not create duplicates in the tile selection
            id_list_oth_tiles = aoi_tiles_intersecting_oth_labels.id.to_numpy().tolist()
            id_list_gt_tiles = aoi_tiles_intersecting_gt_labels.id.to_numpy().tolist()
            nbr_duplicated_id = len(set(id_list_gt_tiles) & set(id_list_oth_tiles))

            if nbr_duplicated_id != 0:
                aoi_tiles_intersecting_gt_labels=aoi_tiles_intersecting_gt_labels[
                                                    ~aoi_tiles_intersecting_gt_labels['id'].isin(id_list_oth_tiles)]
                logger.info(f'{nbr_duplicated_id} tiles were in the GT and the OTH dataset')

            aoi_tiles_gdf = pd.concat([
                aoi_tiles_intersecting_gt_labels.head(DEBUG_MODE_LIMIT//2), # a sample of tiles covering GT labels
                aoi_tiles_intersecting_oth_labels.head(DEBUG_MODE_LIMIT//4), # a sample of tiles convering OTH labels
                aoi_tiles_gdf # the entire tileset, so as to also have tiles covering no label at all (duplicates will be dropped)
            ])
            
        elif GT_LABELS_GEOJSON and not OTH_LABELS_GEOJSON:
            aoi_tiles_gdf = pd.concat([
                aoi_tiles_intersecting_gt_labels.head(DEBUG_MODE_LIMIT*3//4),
                aoi_tiles_gdf
            ])
        
        elif not GT_LABELS_GEOJSON and OTH_LABELS_GEOJSON:
            aoi_tiles_gdf = pd.concat([
                aoi_tiles_intersecting_oth_labels.head(DEBUG_MODE_LIMIT*3//4),
                aoi_tiles_gdf
            ])
        else:
            pass # the following two lines of code would apply in this case
            
        aoi_tiles_gdf.drop_duplicates(inplace=True)
        aoi_tiles_gdf = aoi_tiles_gdf.head(DEBUG_MODE_LIMIT).copy()


    ALL_IMG_PATH = os.path.join(OUTPUT_DIR, f"all-images-{TILE_SIZE}" if TILE_SIZE else "all-images")

    if not os.path.exists(ALL_IMG_PATH):
        os.makedirs(ALL_IMG_PATH)

    if ORTHO_WS_TYPE == 'MIL':
        
        logger.info("(using the MIL connector)")
      
        job_dict = MIL.get_job_dict(
            tiles_gdf=aoi_tiles_gdf.to_crs(ORTHO_WS_SRS), # <- note the reprojection
            mil_url=ORTHO_WS_URL, 
            width=TILE_SIZE, 
            height=TILE_SIZE, 
            img_path=ALL_IMG_PATH, 
            image_sr=ORTHO_WS_SRS.split(":")[1], 
            save_metadata=SAVE_METADATA,
            overwrite=OVERWRITE
        )

        image_getter = MIL.get_geotiff

    elif ORTHO_WS_TYPE == 'WMS':
        
        logger.info("(using the WMS connector)")

        job_dict = WMS.get_job_dict(
            tiles_gdf=aoi_tiles_gdf.to_crs(ORTHO_WS_SRS), # <- note the reprojection
            wms_url=ORTHO_WS_URL, 
            layers=ORTHO_WS_LAYERS,
            width=TILE_SIZE, 
            height=TILE_SIZE, 
            img_path=ALL_IMG_PATH, 
            srs=ORTHO_WS_SRS, 
            save_metadata=SAVE_METADATA,
            overwrite=OVERWRITE
        )

        image_getter = WMS.get_geotiff

    elif ORTHO_WS_TYPE == 'XYZ':
        
        logger.info("(using the XYZ connector)")

        job_dict = XYZ.get_job_dict(
            tiles_gdf=aoi_tiles_gdf.to_crs(ORTHO_WS_SRS), # <- note the reprojection
            xyz_url=ORTHO_WS_URL, 
            img_path=ALL_IMG_PATH, 
            save_metadata=SAVE_METADATA,
            overwrite=OVERWRITE
        )

        image_getter = XYZ.get_geotiff

    else:
        logger.critical(f'Web Services of type "{ORTHO_WS_TYPE}" are not supported. Exiting.')
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

    # let's save metadata... (kind of an image catalog)
    IMG_METADATA_FILE = os.path.join(OUTPUT_DIR, 'img_metadata.json')
    with open(IMG_METADATA_FILE, 'w') as fp:
        json.dump(img_metadata_dict, fp)

    written_files.append(IMG_METADATA_FILE)
    logger.success(f"{DONE_MSG} A file was written: {IMG_METADATA_FILE}")    


    # ------ Training/validation/test/other dataset generation

    if GT_LABELS_GEOJSON:
        try:
            assert( aoi_tiles_gdf.crs == gt_labels_gdf.crs ), "CRS Mismatch between AoI tiles and labels."
        except Exception as e:
            logger.critical(e)
            sys.exit(1)

        GT_tiles_gdf = gpd.sjoin(aoi_tiles_gdf, gt_labels_gdf, how='inner', predicate='intersects')

        # get the number of labels per class
        labels_per_class_dict={}
        for category in GT_tiles_gdf.CATEGORY.unique():
            labels_per_class_dict[category] = GT_tiles_gdf[GT_tiles_gdf.CATEGORY == category].shape[0]
        # Get the number of labels per tile
        labels_per_tiles_gdf = GT_tiles_gdf.groupby(['id', 'CATEGORY'], as_index=False).size()

        GT_tiles_gdf = GT_tiles_gdf.drop_duplicates(subset=aoi_tiles_gdf.columns)
        GT_tiles_gdf.drop(columns=['index_right'], inplace=True)

        # remove tiles including at least one "oth" label (if applicable)
        if OTH_LABELS_GEOJSON:
            tmp_GT_tiles_gdf = GT_tiles_gdf.copy()
            tiles_to_remove_gdf = gpd.sjoin(tmp_GT_tiles_gdf, oth_labels_gdf, how='inner', predicate='intersects')
            GT_tiles_gdf = tmp_GT_tiles_gdf[~tmp_GT_tiles_gdf.id.astype(str).isin(tiles_to_remove_gdf.id.astype(str))].copy()
            del tmp_GT_tiles_gdf

        # OTH tiles = AoI tiles which are not GT
        OTH_tiles_gdf = aoi_tiles_gdf[~aoi_tiles_gdf.id.astype(str).isin(GT_tiles_gdf.id.astype(str)) ].copy()
        OTH_tiles_gdf['dataset'] = 'oth'

        assert( len(aoi_tiles_gdf) == len(GT_tiles_gdf) + len(OTH_tiles_gdf) )
        
        # 70%, 15%, 15% split
        if not SEED:
            for seed in range(11):
                ok_split = 0
                trn_tiles_ids, val_tiles_ids, tst_tiles_ids = split_dataset(GT_tiles_gdf, seed=seed)
                
                for category in labels_per_tiles_gdf.CATEGORY.unique():
                    
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
                
                if ok_split == len(GT_tiles_gdf.CATEGORY.unique())*3:
                    logger.info(f'A seed of {seed} produces a good repartition of the labels.')
                    SEED = seed
                    break
                
                if seed == 10:
                    logger.warning('No good seed found between 0 and 10. The user should set a seed manually.')
                    SEED = seed
            
        else:
            trn_tiles_ids, val_tiles_ids, tst_tiles_ids = split_dataset(GT_tiles_gdf, seed=SEED)


            
        for df in [GT_tiles_gdf, labels_per_tiles_gdf]:
            df.loc[df.id.astype(str).isin(trn_tiles_ids), 'dataset'] = 'trn'
            df.loc[df.id.astype(str).isin(val_tiles_ids), 'dataset'] = 'val'
            df.loc[df.id.astype(str).isin(tst_tiles_ids), 'dataset'] = 'tst'

        logger.info('Repartition in the datasets by category:')
        for dst in ['trn', 'val', 'tst']:
            for category in labels_per_tiles_gdf.CATEGORY.unique():
                id_list = labels_per_tiles_gdf.loc[(labels_per_tiles_gdf.dataset==dst) & (labels_per_tiles_gdf.CATEGORY==category), 'id']
                logger.info(f'   {category} labels in {dst} dataset: {labels_per_tiles_gdf.loc[labels_per_tiles_gdf.id.astype(str).isin(id_list), "size"].sum()}')

        # remove columns generated by the Spatial Join
        GT_tiles_gdf = GT_tiles_gdf[aoi_tiles_gdf.columns.tolist() + ['dataset']].copy()

        assert( len(GT_tiles_gdf) == len(trn_tiles_ids) + len(val_tiles_ids) + len(tst_tiles_ids) ), \
            'Tiles were lost in the split between training, validation and test sets.'
        
        split_aoi_tiles_gdf = pd.concat(
            [
                GT_tiles_gdf,
                OTH_tiles_gdf
            ]
        )
        
        # let's free up some memory
        del GT_tiles_gdf
        del OTH_tiles_gdf
         
    else:
        split_aoi_tiles_gdf = aoi_tiles_gdf.copy()
        split_aoi_tiles_gdf['dataset'] = 'oth'
        
        
    assert( len(split_aoi_tiles_gdf) == len(aoi_tiles_gdf) ) # it means that all the tiles were actually used
    
    
    SPLIT_AOI_TILES_GEOJSON = os.path.join(OUTPUT_DIR, 'split_aoi_tiles.geojson')

    try:
        split_aoi_tiles_gdf.to_file(SPLIT_AOI_TILES_GEOJSON, driver='GeoJSON')
        # sp_tiles_gdf.to_crs(epsg=2056).to_file(os.path.join(OUTPUT_DIR, 'swimmingpool_tiles.shp'))
    except Exception as e:
        logger.error(e)
    written_files.append(SPLIT_AOI_TILES_GEOJSON)
    logger.success(f'{DONE_MSG} A file was written {SPLIT_AOI_TILES_GEOJSON}')

    img_md_df = pd.DataFrame.from_dict(img_metadata_dict, orient='index')
    img_md_df.reset_index(inplace=True)
    img_md_df.rename(columns={"index": "img_file"}, inplace=True)

    img_md_df['id'] = img_md_df.apply(misc.img_md_record_to_tile_id, axis=1)

    split_aoi_tiles_with_img_md_gdf = split_aoi_tiles_gdf.merge(img_md_df, on='id', how='left')
    split_aoi_tiles_with_img_md_gdf.apply(misc.make_hard_link, axis=1)

    # ------ Generating COCO Annotations
    
    if GT_LABELS_GEOJSON and OTH_LABELS_GEOJSON:
        
        assert( gt_labels_gdf.crs == oth_labels_gdf.crs)
        
        labels_gdf = pd.concat([
            gt_labels_gdf,
            oth_labels_gdf
        ]).reset_index()

    elif GT_LABELS_GEOJSON and not OTH_LABELS_GEOJSON:
        
        labels_gdf = gt_labels_gdf.copy().reset_index()
        
    elif not GT_LABELS_GEOJSON and OTH_LABELS_GEOJSON:
        
        labels_gdf = oth_labels_gdf.copy().reset_index()
    
    else:
        
        labels_gdf = gpd.GeoDataFrame()


    if COCO_YEAR==None:
        print()
        toc = time.time()
        logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

        sys.stderr.flush()
        sys.exit(0)
    
    # Get possibles combination for category and supercategory
    combinations_category_dict = labels_gdf.groupby(['CATEGORY','SUPERCATEGORY'], as_index=False).size().drop(columns=['size']).to_dict('tight')
    combinations_category_lists=combinations_category_dict['data']
    logger.info(f'Possible categories and supercategories:')
    for category, supercategory in combinations_category_lists:
        print(f"- {category}, {supercategory}")

    for dataset in split_aoi_tiles_with_img_md_gdf.dataset.unique():
        
        logger.info(f'Generating COCO annotations for the {dataset} dataset...')
        
        coco = COCO.COCO()
        coco.set_info(year=COCO_YEAR, 
                      version=COCO_VERSION, 
                      description=f"{COCO_DESCRIPTION} - {dataset} dataset", 
                      contributor=COCO_CONTRIBUTOR, 
                      url=COCO_URL)
        
        coco_license = coco.license(name=COCO_LICENSE_NAME, url=COCO_LICENSE_URL)
        coco_license_id = coco.insert_license(coco_license)

        # Put categories in coco objects and keep them in a dict
        coco_category={}
        for category, supercategory in combinations_category_lists:
            
            coco_category_name = str(category)
            coco_category_supercat = str(supercategory)
            key=coco_category_name + '_' + coco_category_supercat

            coco_category[key] = coco.category(name=coco_category_name, supercategory=coco_category_supercat)

            _ = coco.insert_category(coco_category[key])
        
        tmp_tiles_gdf = split_aoi_tiles_with_img_md_gdf[split_aoi_tiles_with_img_md_gdf.dataset == dataset].dropna()
        
        if len(labels_gdf) > 0:
            assert(labels_gdf.crs == tmp_tiles_gdf.crs)
        
        tiles_iterator = tmp_tiles_gdf.sort_index().iterrows()
    
        try:
            results = Parallel(n_jobs=N_JOBS, backend="loky") \
                    (delayed(get_coco_image_and_segmentations) \
                    (tile, labels_gdf, coco_license_id, coco_category, OUTPUT_DIR) \
                    for tile in tqdm(tiles_iterator, total=len(tmp_tiles_gdf) ))
        except Exception as e:
            logger.critical(f"Tile generation failed. Exception: {e}")
            sys.exit(1)
        
        for result in results:
            
            coco_image, coco_category_id, segmentations = result

            try:
                coco_image_id = coco.insert_image(coco_image)
            except Exception as e:
                logger.critical(f"Could not insert image into the COCO data structure. Exception: {e}")
                sys.exit(1)

            for segmentation in segmentations:

                coco_annotation = coco.annotation(
                    coco_image_id,
                    coco_category_id,
                    [segmentation],
                    iscrowd=0
                )
                # The bbox for coco objects is defined as [x_min, y_min, width, height].
                # https://cocodataset.org/#format-data under "1. Object Detection"

                try:
                    coco.insert_annotation(coco_annotation)
                except Exception as e:
                    logger.critical(f"Could not insert annotation into the COCO data structure. Exception: {e}")
                    sys.exit(1)
        
        COCO_file = os.path.join(OUTPUT_DIR, f'COCO_{dataset}.json')

        with open(COCO_file, 'w') as fp:
            json.dump(coco.to_json(), fp)
        
        written_files.append(COCO_file)

    labels_dict_file = os.path.join(OUTPUT_DIR, 'labels_id.json')
    with open(labels_dict_file, 'w') as fp:
        json.dump(coco_category, fp)
    written_files.append(labels_dict_file)

    toc = time.time()
    logger.success(DONE_MSG)

    logger.info("You can now open a Linux shell and type the following command in order to create a .tar.gz archive including images and COCO annotations:")
    if GT_LABELS_GEOJSON:
        if TILE_SIZE:
            logger.info(f"cd {OUTPUT_DIR}; tar -cvf images-{TILE_SIZE}.tar COCO_{{trn,val,tst,oth}}.json && tar -rvf images-{TILE_SIZE}.tar {{trn,val,tst,oth}}-images-{TILE_SIZE} && gzip < images-{TILE_SIZE}.tar > images-{TILE_SIZE}.tar.gz && rm images-{TILE_SIZE}.tar; cd -")
        else:
            logger.info(f"cd {OUTPUT_DIR}; tar -cvf images.tar COCO_{{trn,val,tst,oth}}.json && tar -rvf images.tar {{trn,val,tst,oth}}-images && gzip < images.tar > images.tar.gz && rm images.tar; cd -")
    else:
        if TILE_SIZE:
            logger.info(f"cd {OUTPUT_DIR}; tar -cvf images-{TILE_SIZE}.tar COCO_oth.json && tar -rvf images-{TILE_SIZE}.tar oth-images-{TILE_SIZE} && gzip < images-{TILE_SIZE}.tar > images-{TILE_SIZE}.tar.gz && rm images-{TILE_SIZE}.tar; cd -")
        else:
            logger.info(f"cd {OUTPUT_DIR}; tar -cvf images.tar COCO_oth.json && tar -rvf images.tar oth-images && gzip < images.tar > images.tar.gz && rm images.tar; cd -")
    
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
