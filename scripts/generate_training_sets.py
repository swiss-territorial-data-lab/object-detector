#!/bin/python
# -*- coding: utf-8 -*-

import logging
import logging.config
import time
import argparse
import yaml
import os, sys
import requests
import geopandas as gpd
import pandas as pd
import json

from joblib import Parallel, delayed
from tqdm import tqdm

# the following lines allow us to import modules from within this file's parent folder
from inspect import getsourcefile
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from helpers import MIL # MIL stands for Map Image Layer, cf. https://pro.arcgis.com/en/pro-app/help/sharing/overview/map-image-layer.htm
from helpers import WMS # Web Map Service
from helpers import COCO
from helpers import misc

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('root')


def img_md_record_to_tile_id(img_md_record):
    
    filename = os.path.split(img_md_record.img_file)[-1]
    
    z_x_y = filename.split('.')[0]
    z, x, y = z_x_y.split('_')
    
    return f"({x}, {y}, {z})"


def make_hard_link(row):

    if not os.path.isfile(row.img_file):
        raise Exception('File not found.')

    src_file = row.img_file
    dst_file = src_file.replace('all', row.dataset)

    dirname = os.path.dirname(dst_file)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    if os.path.exists(dst_file):
        os.remove(dst_file)

    os.link(src_file, dst_file)

    return None


def my_unpack(list_of_tuples):
    # cf. https://www.geeksforgeeks.org/python-convert-list-of-tuples-into-list/
    
    return [item for t in list_of_tuples for item in t]


def read_img_metadata(md_file, all_img_path):
    img_path = os.path.join(all_img_path, md_file.replace('json', 'tif'))
    
    with open(os.path.join(all_img_path, md_file), 'r') as fp:
        return {img_path: json.load(fp)}


def get_COCO_image_and_segmentations(tile, labels, COCO_license_id, output_dir):
    
    _id, _tile = tile

    coco_obj = COCO.COCO()

    this_tile_dirname = os.path.relpath(_tile['img_file'].replace('all', _tile['dataset']), output_dir)
    this_tile_dirname = this_tile_dirname.replace('\\', '/') # should the dirname be generated from Windows

    COCO_image = coco_obj.image(output_dir, this_tile_dirname, COCO_license_id)
    
    xmin, ymin, xmax, ymax = [float(x) for x in MIL.bounds_to_bbox(_tile['geometry'].bounds).split(',')]
    
    # note the .explode() which turns Multipolygon into Polygons
    clipped_labels_gdf = gpd.clip(labels, _tile['geometry']).explode()

    #try:
    #    assert( len(clipped_labels_gdf) > 0 ) 
    #except:
    #    raise Exception(f'No labels found within this tile! Tile ID = {tile.id}')

    segmentations = []
    
    for label in clipped_labels_gdf.itertuples():
        scaled_poly = misc.scale_polygon(label.geometry, xmin, ymin, xmax, ymax, 
                                         COCO_image['width'], COCO_image['height'])
        scaled_poly = scaled_poly[:-1] # let's remove the last point
        
        segmentation = my_unpack(scaled_poly)

        try:
            assert(min(segmentation) >= 0)
            assert(max(segmentation) <= min(COCO_image['width'], COCO_image['height']))
        except Exception as e:
            raise Exception(f"Label boundaries exceed this tile size! Tile ID = {_tile['id']}")
            
        segmentations.append(segmentation)

    return (COCO_image, segmentations)



if __name__ == "__main__":


    tic = time.time()
    logger.info('Starting...')

    parser = argparse.ArgumentParser(description="This script generates COCO-annotated training/validation/test/other datasets for object detection tasks.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # TODO: check whether the configuration file contains the required information
    DEBUG_MODE = cfg['debug_mode']
    
    OUTPUT_DIR = cfg['output_folder']
    
    ORTHO_WS_TYPE = cfg['datasets']['orthophotos_web_service']['type']
    ORTHO_WS_URL = cfg['datasets']['orthophotos_web_service']['url']
    ORTHO_WS_SRS = cfg['datasets']['orthophotos_web_service']['srs']
    if 'layers' in cfg['datasets']['orthophotos_web_service'].keys():
        ORTHO_WS_LAYERS = cfg['datasets']['orthophotos_web_service']['layers']

    AOI_TILES_GEOJSON = cfg['datasets']['aoi_tiles_geojson']
    GT_LABELS_GEOJSON = cfg['datasets']['ground_truth_labels_geojson']
    if 'other_labels_geojson' in cfg['datasets'].keys():
        OTH_LABELS_GEOJSON = cfg['datasets']['other_labels_geojson']
    else:
        OTH_LABELS_GEOJSON = None

    SAVE_METADATA = True
    OVERWRITE = cfg['overwrite']
    TILE_SIZE = cfg['tile_size']
    N_JOBS = cfg['n_jobs']
    COCO_YEAR = cfg['COCO_metadata']['year']
    COCO_VERSION = cfg['COCO_metadata']['version']
    COCO_DESCRIPTION = cfg['COCO_metadata']['description']
    COCO_CONTRIBUTOR = cfg['COCO_metadata']['contributor']
    COCO_URL = cfg['COCO_metadata']['url']
    COCO_LICENSE_NAME = cfg['COCO_metadata']['license']['name']
    COCO_LICENSE_URL = cfg['COCO_metadata']['license']['url']
    COCO_CATEGORY_NAME = cfg['COCO_metadata']['category']['name']
    COCO_CATEGORY_SUPERCATEGORY = cfg['COCO_metadata']['category']['supercategory']


    # let's make the output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    written_files = []

    # ------ Loading datasets

    logger.info("Loading AoI tiles as a GeoPandas DataFrame...")
    aoi_tiles_gdf = gpd.read_file(AOI_TILES_GEOJSON)
    logger.info(f"...done. {len(aoi_tiles_gdf)} records were found.")

    logger.info("Loading Ground Truth Labels as a GeoPandas DataFrame...")
    gt_labels_gdf = gpd.read_file(GT_LABELS_GEOJSON)
    logger.info(f"...done. {len(gt_labels_gdf)} records were found.")

    if OTH_LABELS_GEOJSON:
        logger.info("Loading Other Labels as a GeoPandas DataFrame...")
        oth_labels_gdf = gpd.read_file(OTH_LABELS_GEOJSON)
        logger.info(f"...done. {len(oth_labels_gdf)} records were found.")

    logger.info("Generating the list of tasks to be executed (one task per tile)...")

    DEBUG_MODE_LIMIT = 100
    if DEBUG_MODE:
        logger.warning(f"Debug mode: ON => Only {DEBUG_MODE_LIMIT} tiles will be processed.")

        assert( aoi_tiles_gdf.crs == gt_labels_gdf.crs )
        if OTH_LABELS_GEOJSON:
            assert( aoi_tiles_gdf.crs == oth_labels_gdf.crs )

        aoi_tiles_intersecting_gt_labels = gpd.sjoin(aoi_tiles_gdf, gt_labels_gdf, how='inner', op='intersects')
        aoi_tiles_intersecting_gt_labels = aoi_tiles_intersecting_gt_labels[aoi_tiles_gdf.columns]
        aoi_tiles_intersecting_gt_labels.drop_duplicates(inplace=True)

        if OTH_LABELS_GEOJSON:
            aoi_tiles_intersecting_oth_labels = gpd.sjoin(aoi_tiles_gdf, oth_labels_gdf, how='inner', op='intersects')
            aoi_tiles_intersecting_oth_labels = aoi_tiles_intersecting_oth_labels[aoi_tiles_gdf.columns]
            aoi_tiles_intersecting_oth_labels.drop_duplicates(inplace=True)

            aoi_tiles_gdf = pd.concat([
                aoi_tiles_intersecting_gt_labels.head(DEBUG_MODE_LIMIT//2),
                aoi_tiles_intersecting_oth_labels.head(DEBUG_MODE_LIMIT) # just to make sure to have enough tiles even in the case of duplicates
            ])
            
        else:
            aoi_tiles_gdf = pd.concat([
                aoi_tiles_intersecting_gt_labels.head(DEBUG_MODE_LIMIT),
            ])
            
        aoi_tiles_gdf.drop_duplicates(inplace=True)
        aoi_tiles_gdf = aoi_tiles_gdf.head(DEBUG_MODE_LIMIT).copy()


    ALL_IMG_PATH = os.path.join(OUTPUT_DIR, f"all-images-{TILE_SIZE}")

    if not os.path.exists(ALL_IMG_PATH):
        os.makedirs(ALL_IMG_PATH)

    if ORTHO_WS_TYPE == 'MIL':
      
        job_dict = MIL.get_job_dict(
            tiles_gdf=aoi_tiles_gdf.to_crs(ORTHO_WS_SRS), # <- note the reprojection
            mil_url=ORTHO_WS_URL, 
            width=TILE_SIZE, 
            height=TILE_SIZE, 
            img_path=ALL_IMG_PATH, 
            imageSR=ORTHO_WS_SRS.split(":")[1], 
            save_metadata=SAVE_METADATA,
            overwrite=OVERWRITE
        )

        image_getter = MIL.get_geotiff

    elif ORTHO_WS_TYPE == 'WMS':

        job_dict = WMS.get_job_dict(
            tiles_gdf=aoi_tiles_gdf.to_crs(ORTHO_WS_SRS), # <- note the reprojection
            WMS_url=ORTHO_WS_URL, 
            layers=ORTHO_WS_LAYERS,
            width=TILE_SIZE, 
            height=TILE_SIZE, 
            img_path=ALL_IMG_PATH, 
            srs=ORTHO_WS_SRS, 
            save_metadata=SAVE_METADATA,
            overwrite=OVERWRITE
        )

        image_getter = WMS.get_geotiff

    else:
        logger.critical(f'Web Service of type "{ORTHO_WS_TYPE}" are not yet supported. Exiting.')
        sys.exit(1)

    logger.info("...done.")

    logger.info(f"Executing tasks, {N_JOBS} at a time...")
    job_outcome = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(image_getter)(**v) for k, v in tqdm( sorted(list(job_dict.items())) )
    )
    logger.info("Checking whether all the expected tiles were actually downloaded...")

    all_tiles_were_downloaded = True
    for job in job_dict.keys():
        if not os.path.isfile(job) or not os.path.isfile(job.replace('.tif', '.json')):
            all_tiles_were_downloaded = False
            logger.warning('Failed task: ', job)

    if all_tiles_were_downloaded:
        logger.info("...done.")
    else:
        logger.critical("Some tiles were not downloaded. Please try to run this script again.")
        sys.exit(1)


    # ------ Collecting image metadata, to be used when assessing predictions

    logger.info("Collecting image metadata...")

    md_files = [f for f in os.listdir(ALL_IMG_PATH) if os.path.isfile(os.path.join(ALL_IMG_PATH, f)) and f.endswith('.json')]
    
    img_metadata_list = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(read_img_metadata)(md_file, ALL_IMG_PATH) for md_file in tqdm(md_files))
    img_metadata_dict = { k: v for img_md in img_metadata_list for (k, v) in img_md.items() }

    # let's save metadata... (kind of an image catalog)
    IMG_METADATA_FILE = os.path.join(OUTPUT_DIR, 'img_metadata.json')
    with open(IMG_METADATA_FILE, 'w') as fp:
        json.dump(img_metadata_dict, fp)

    written_files.append(IMG_METADATA_FILE)
    logger.info(f"...done. A file was written: {IMG_METADATA_FILE}")    


    # ------ Training/validation/test/other dataset generation

    try:
        assert( aoi_tiles_gdf.crs == gt_labels_gdf.crs ), "CRS Mismatch between AoI tiles and labels."
    except Exception as e:
        logger.critical(e)
        sys.exit(1)

    GT_tiles_gdf = gpd.sjoin(aoi_tiles_gdf, gt_labels_gdf, how='inner', op='intersects')
    # remove columns generated by the Spatial Join
    GT_tiles_gdf = GT_tiles_gdf[aoi_tiles_gdf.columns].copy()
    GT_tiles_gdf.drop_duplicates(inplace=True)
    
    if OTH_LABELS_GEOJSON:
        # OTH tiles = AoI tiles which are not GT
        OTH_tiles_gdf = aoi_tiles_gdf[ ~aoi_tiles_gdf.id.astype(str).isin(GT_tiles_gdf.id.astype(str)) ].copy()

        assert( len(aoi_tiles_gdf) == len(GT_tiles_gdf) + len(OTH_tiles_gdf) )

    # 70%, 15%, 15% split
    trn_tiles_ids = GT_tiles_gdf\
        .sample(frac=.7, random_state=1)\
        .id.astype(str).values.tolist()

    val_tiles_ids = GT_tiles_gdf[~GT_tiles_gdf.id.astype(str).isin(trn_tiles_ids)]\
        .sample(frac=.5, random_state=1)\
        .id.astype(str).values.tolist()

    tst_tiles_ids = GT_tiles_gdf[~GT_tiles_gdf.id.astype(str).isin(trn_tiles_ids + val_tiles_ids)]\
        .id.astype(str).values.tolist()

    assert( len(trn_tiles_ids) + len(val_tiles_ids) + len(tst_tiles_ids) == len(GT_tiles_gdf) )

    GT_tiles_gdf.loc[GT_tiles_gdf.id.astype(str).isin(trn_tiles_ids), 'dataset'] = 'trn'
    GT_tiles_gdf.loc[GT_tiles_gdf.id.astype(str).isin(val_tiles_ids), 'dataset'] = 'val'
    GT_tiles_gdf.loc[GT_tiles_gdf.id.astype(str).isin(tst_tiles_ids), 'dataset'] = 'tst'
    
    if OTH_LABELS_GEOJSON:
        OTH_tiles_gdf['dataset'] = 'oth'

        assert( len(aoi_tiles_gdf) == len(GT_tiles_gdf) + len(OTH_tiles_gdf) )

        split_aoi_tiles_gdf = pd.concat(
            [
                GT_tiles_gdf,
                OTH_tiles_gdf
            ]
        )
        
    else:
        assert( len(aoi_tiles_gdf) == len(GT_tiles_gdf) )
        split_aoi_tiles_gdf = GT_tiles_gdf.copy()

    assert( len(split_aoi_tiles_gdf) == len(aoi_tiles_gdf) )

    # let's free up some memory
    del GT_tiles_gdf
    if OTH_LABELS_GEOJSON: 
        del OTH_tiles_gdf

    if OTH_LABELS_GEOJSON:
        logger.info("Exporting a vector layer including masks for the training/validation/test/other datasets...")
    else:
        logger.info("Exporting a vector layer including masks for the training/validation/test datasets...")
    SPLIT_AOI_TILES_GEOJSON = os.path.join(OUTPUT_DIR, 'split_aoi_tiles.geojson')

    try:
        split_aoi_tiles_gdf.to_file(SPLIT_AOI_TILES_GEOJSON, driver='GeoJSON')
        # sp_tiles_gdf.to_crs(epsg=2056).to_file(os.path.join(OUTPUT_DIR, 'swimmingpool_tiles.shp'))
    except Exception as e:
        logger.error(e)
    written_files.append(SPLIT_AOI_TILES_GEOJSON)
    logger.info(f'...done. A file was written {SPLIT_AOI_TILES_GEOJSON}')

    img_md_df = pd.DataFrame.from_dict(img_metadata_dict, orient='index')
    img_md_df.reset_index(inplace=True)
    img_md_df.rename(columns={"index": "img_file"}, inplace=True)

    img_md_df['id'] = img_md_df.apply(img_md_record_to_tile_id, axis=1)

    split_aoi_tiles_with_img_md_gdf = split_aoi_tiles_gdf.merge(img_md_df, on='id', how='left')
    split_aoi_tiles_with_img_md_gdf.apply(make_hard_link, axis=1)

    # ------ Generating COCO Annotations

    if OTH_LABELS_GEOJSON:
        labels_gdf = pd.concat([
            gt_labels_gdf,
            oth_labels_gdf
        ]).reset_index()
    else:
        labels_gdf = gt_labels_gdf.copy().reset_index()
        

    for dataset in split_aoi_tiles_with_img_md_gdf.dataset.unique():
        
        logger.info(f'Generating COCO annotations for the {dataset} dataset...')
        
        coco = COCO.COCO()
        coco.set_info(the_year=COCO_YEAR, 
                      the_version=COCO_VERSION, 
                      the_description=f"{COCO_DESCRIPTION} - {dataset} dataset", 
                      the_contributor=COCO_CONTRIBUTOR, 
                      the_url=COCO_URL)
        
        coco_license = coco.license(the_name=COCO_LICENSE_NAME, the_url=COCO_LICENSE_URL)
        coco_license_id = coco.insert_license(coco_license)

        # TODO: read (super)category from the labels datataset
        coco_category = coco.category(the_name=COCO_CATEGORY_NAME, the_supercategory=COCO_CATEGORY_SUPERCATEGORY)                      
        coco_category_id = coco.insert_category(coco_category)
        
        tmp_tiles_gdf = split_aoi_tiles_with_img_md_gdf[split_aoi_tiles_with_img_md_gdf.dataset == dataset].dropna()
        #tmp_tiles_gdf = tmp_tiles_gdf.to_crs(epsg=3857)
        
        assert(labels_gdf.crs == tmp_tiles_gdf.crs)
        
        tiles_iterator = tmp_tiles_gdf.sort_index().iterrows()
    
        results = Parallel(n_jobs=N_JOBS, backend="loky") \
                        (delayed(get_COCO_image_and_segmentations) \
                        (tile, labels_gdf, coco_license_id, OUTPUT_DIR) \
                        for tile in tqdm( tiles_iterator, total=len(tmp_tiles_gdf) ))
        
        for result in results:
            coco_image, segmentations = result
            coco_image_id = coco.insert_image(coco_image)

            for segmentation in segmentations:

                coco_annotation = coco.annotation(coco_image_id,
                                                  coco_category_id,
                                                  [segmentation],
                                                  the_iscrowd=0)

                coco.insert_annotation(coco_annotation)
                
        with open(os.path.join(OUTPUT_DIR, f'COCO_{dataset}.json'), 'w') as fp:
            json.dump(coco.to_json(), fp)


    toc = time.time()
    logger.info("...done.")

    logger.info("You can now open a Linux shell and type the following command in order to create a .tar.gz archive including images and COCO annotations:")
    if OTH_LABELS_GEOJSON:
        logger.info(f"cd {OUTPUT_DIR}; tar -cvf images-{TILE_SIZE}.tar COCO_{{trn,val,tst,oth}}.json && tar -rvf images-{TILE_SIZE}.tar {{trn,val,tst,oth}}-images-256 && gzip < images-{TILE_SIZE}.tar > images-{TILE_SIZE}.tar.gz && rm images-{TILE_SIZE}.tar; cd -")
    else:
        logger.info(f"cd {OUTPUT_DIR}; tar -cvf images-{TILE_SIZE}.tar COCO_{{trn,val,tst}}.json && tar -rvf images-{TILE_SIZE}.tar {{trn,val,tst}}-images-256 && gzip < images-{TILE_SIZE}.tar > images-{TILE_SIZE}.tar.gz && rm images-{TILE_SIZE}.tar; cd -")
    
    print()
    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)
    print()

    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()