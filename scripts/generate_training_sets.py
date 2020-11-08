#!/bin/python
# -*- coding: utf-8 -*-

import logging
import logging.config
import time
import argparse
import yaml
import os, sys, inspect
import requests
import geopandas as gpd
import pandas as pd
import json

from joblib import Parallel, delayed
from tqdm import tqdm

# the following allows us to import modules from within this file's parent folder
sys.path.insert(0, '.')
from helpers import MIL # MIL stands for Map Image Layer, cf. https://pro.arcgis.com/en/pro-app/help/sharing/overview/map-image-layer.htm
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


def read_img_metadata(md_file):
    img_path = os.path.join(ALL_IMG_PATH, md_file.replace('json', 'tif'))
    
    with open(os.path.join(ALL_IMG_PATH, md_file), 'r') as fp:
        return {img_path: json.load(fp)}


def get_COCO_image_and_segmentations(tile, labels, COCO_license_id, output_dir):

    coco_obj = COCO.COCO()

    this_tile_dirname = os.path.relpath(tile.img_file.replace('all', tile.dataset), output_dir)
    this_tile_dirname = this_tile_dirname.replace('\\', '/')

    # this_tile_dirname = tile.img_file.replace('all', tile.dataset)
    
    COCO_image = coco_obj.image(output_dir, this_tile_dirname, COCO_license_id)
    #COCO_image_id = COCO_obj.insert_image(COCO_image) 
    
    xmin, ymin, xmax, ymax = [float(x) for x in MIL.bounds_to_bbox(tile.geometry.bounds).split(',')]
    
    # note the .explode() which turns Multipolygon into Polygons
    clipped_labels_gdf = gpd.clip(labels_gdf, tile.geometry).explode()
    
    assert( len(clipped_labels_gdf) > 0 ) 

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
            raise Exception(f'Label boundaries exceed this tile size! Tile ID = {tile.id}')
            
        segmentations.append(segmentation)

    return (COCO_image, segmentations)



if __name__ == "__main__":


    tic = time.time()
    logger.info('Starting...')

    parser = argparse.ArgumentParser(description="This script generates COCO-annotated training/validation/test datasets for the Geneva's Swimming Pools detection task.")
    parser.add_argument('config_file', type=str, help='a YAML config file')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # TODO: check whether the configuration file contains the required information
    OUTPUT_DIR = cfg['folders']['output']
    LAKES_SHPFILE = cfg['datasets']['lakes_shapefile']
    PARCELS_SHPFILE = cfg['datasets']['parcels_shapefile']
    SWIMMINGPOOLS_SHPFILE = cfg['datasets']['swimmingpools_shapefile']
    MIL_URL = cfg['datasets']['orthophotos_map_image_layer_url']
    OK_TILE_IDS_CSV = cfg['datasets']['OK_z18_tile_IDs_csv']
    ZOOM_LEVEL = 18 # this is hard-coded 'cause we only know "OK tile IDs" for this zoom level
    SAVE_METADATA = cfg['save_image_metadata']
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


    # let's make the output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


    # ------ Down(loading) datasets

    dataset_dict = {}

    for dataset in ['lakes', 'parcels', 'swimmingpools']:

        if eval(f'{dataset.upper()}_SHPFILE').startswith('http'):

            logger.info(f"Downloading the {dataset} dataset...")

            shpfile_name = eval(f'{dataset.upper()}_SHPFILE').split('/')[-1]
            shpfile_path = os.path.join(OUTPUT_DIR, shpfile_name)
        
            r = requests.get(eval(f'{dataset.upper()}_SHPFILE'))  
            with open(shpfile_path, 'wb') as f:
                f.write(r.content)

            logger.info(f"...done. A file was written: {shpfile_path}")

        # TODO: check file integrity (ex.: md5sum)
        logger.info(f"Loading the {dataset} dataset as a GeoPandas DataFrame...")
        dataset_dict[dataset] = gpd.read_file(f'zip://{shpfile_path}')
        logger.info(f"...done. {len(dataset_dict[dataset])} records were found.")


    # ------ Computing the Area of Interest (AOI) = cadastral parcels - Léman lake

    logger.info("Computing the Area of Interest (AOI)...")

    # N.B.: 
    # it's faster to first compute Slippy Map Tiles (cf. https://developers.planet.com/tutorials/slippy-maps-101/), 
    # then suppress the tiles which "fall" within the Léman lake.
    # We rely on supermercado, mercantile and fiona for the computation of Slippy Map Tiles.

    # lake_gdf
    l_gdf = dataset_dict['lakes'].copy()
    # parcels_gdf
    p_gdf = dataset_dict['parcels'].copy()

    PARCELS_TILES_GEOJSON_FILE = os.path.join(OUTPUT_DIR, f"parcels_z{ZOOM_LEVEL}_tiles.geojson")

    if not os.path.isfile(PARCELS_TILES_GEOJSON_FILE):
        logger.info("Exporting the parcels dataset to a GeoJSON file...")
        PARCELS_GEOJSON_FILE = os.path.join(OUTPUT_DIR, 'parcels.geojson')
        p_gdf[['geometry']].to_crs(epsg=4326).to_file(PARCELS_GEOJSON_FILE, driver='GeoJSON')
        logger.info(f"...done. The {PARCELS_GEOJSON_FILE} was written.")

        logger.info(f"You should now open a Linux shell and run the following command from the working directory (./{OUTPUT_DIR}), then run this script again:")
        logger.info(f"cat parcels.geojson | supermercado burn {ZOOM_LEVEL} | mercantile shapes | fio collect > parcels_z{ZOOM_LEVEL}_tiles.geojson")
        sys.exit(0) 
        
    else:
        parcels_tiles_gdf = gpd.read_file(PARCELS_TILES_GEOJSON_FILE)
        
    # parcels tiles falling within the lake
    tiles_to_remove_gdf = gpd.sjoin(parcels_tiles_gdf.to_crs(epsg=l_gdf.crs.to_epsg()), l_gdf[l_gdf.NOM == 'Léman'], how='right', op='within')

    aoi_tiles_gdf = parcels_tiles_gdf[ ~parcels_tiles_gdf.index.isin(tiles_to_remove_gdf.index_left) ]
    assert ( len(aoi_tiles_gdf.drop_duplicates(subset='id')) == len(aoi_tiles_gdf) ) # make sure there are no duplicates

    AOI_TILES_GEOJSON_FILE = os.path.join(OUTPUT_DIR, f'aoi_{ZOOM_LEVEL}_tiles.geojson')
    aoi_tiles_gdf.to_crs(epsg=4326).to_file(AOI_TILES_GEOJSON_FILE, driver='GeoJSON')

    logger.info("Generating the list of tasks to be executed (one task per tile)...")

    ALL_IMG_PATH = os.path.join(OUTPUT_DIR, f"all-images-{TILE_SIZE}")

    if not os.path.exists(ALL_IMG_PATH):
        os.makedirs(ALL_IMG_PATH)

    job_dict = MIL.get_job_dict(aoi_tiles_gdf, 
                                MIL_URL, 
                                TILE_SIZE, 
                                TILE_SIZE, 
                                ALL_IMG_PATH, 
                                imageSR=3857, 
                                save_metadata=SAVE_METADATA,
                                overwrite=OVERWRITE)

    logger.info("...done.")

    logger.info(f"Executing tasks, {N_JOBS} at a time...")
    job_outcome = Parallel(n_jobs=N_JOBS)(delayed(MIL.get_geotiff)(**v) for k, v in tqdm( sorted(list(job_dict.items()))))

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
    
    img_metadata_list = Parallel(n_jobs=N_JOBS)(delayed(read_img_metadata)(md_file) for md_file in tqdm(md_files))
    img_metadata_dict = { k: v for img_md in img_metadata_list for (k, v) in img_md.items() }

    # let's save metadata... (kind of an image catalog)
    IMG_METADATA_FILE = os.path.join(OUTPUT_DIR, 'img_metadata.json')
    with open(IMG_METADATA_FILE, 'w') as fp:
        json.dump(img_metadata_dict, fp)

    logger.info(f"...done. A file was written: {IMG_METADATA_FILE}")    


    # ------ Training/validation/test dataset generation

    # OK tiles: the subset of tiles containing neither false positives nor false negatives
    OK_ids = pd.read_csv(OK_TILE_IDS_CSV)
    OK_tiles_gdf = aoi_tiles_gdf[aoi_tiles_gdf.id.isin(OK_ids.id)]

    # 70%, 15%, 15% split
    trn_tiles_idx = OK_tiles_gdf.sample(frac=.7, random_state=1).index
    val_tiles_idx = OK_tiles_gdf[~OK_tiles_gdf.index.isin(trn_tiles_idx)].sample(frac=.5, random_state=1).index
    tst_tiles_idx = OK_tiles_gdf[~OK_tiles_gdf.index.isin(trn_tiles_idx.union(val_tiles_idx))].index

    # let's tag tiles according to the dataset they belong to
    trn_tiles_gdf = OK_tiles_gdf.loc[trn_tiles_idx].assign(dataset='trn')
    val_tiles_gdf = OK_tiles_gdf.loc[val_tiles_idx].assign(dataset='val')
    tst_tiles_gdf = OK_tiles_gdf.loc[tst_tiles_idx].assign(dataset='tst')

    # sp = swimming pool
    sp_tiles_gdf = pd.concat([trn_tiles_gdf.set_index('id'), 
                              val_tiles_gdf.set_index('id'), 
                              tst_tiles_gdf.set_index('id')], sort=False)
    sp_tiles_gdf = sp_tiles_gdf.reset_index()

    # let's free up some memory
    del trn_tiles_gdf, val_tiles_gdf, tst_tiles_gdf, OK_tiles_gdf

    logger.info("Exporting a vector layer including masks for the training/validation/test datasets...")
    SP_GEOJSON_FILE = os.path.join(OUTPUT_DIR, 'swimmingpool_tiles.geojson')
    try:
        sp_tiles_gdf.to_file(SP_GEOJSON_FILE, driver='GeoJSON')
        # sp_tiles_gdf.to_crs(epsg=2056).to_file(os.path.join(OUTPUT_DIR, 'swimmingpool_tiles.shp'))
    except Exception as e:
        logger.error(e)
    logger.info(f'...done. A file was written {SP_GEOJSON_FILE}')

    img_md_df = pd.DataFrame.from_dict(img_metadata_dict, orient='index')
    img_md_df.reset_index(inplace=True)
    img_md_df.rename(columns={"index": "img_file"}, inplace=True)

    img_md_df['id'] = img_md_df.apply(img_md_record_to_tile_id, axis=1)

    sp_tiles_with_img_md_gdf = sp_tiles_gdf.merge(img_md_df, on='id', how='left')
    sp_tiles_with_img_md_gdf.apply(make_hard_link, axis=1)

    # ------ Generating COCO Annotations

    labels_gdf = dataset_dict['swimmingpools'].copy()
    labels_gdf = labels_gdf.to_crs(epsg=3857)

    for dataset in ['trn', 'val', 'tst']:
        
        logger.info(f'Generating COCO annotations for the {dataset} dataset...')
        
        coco = COCO.COCO()
        coco.set_info(the_year=COCO_YEAR, 
                      the_version=COCO_VERSION, 
                      the_description=f"{COCO_DESCRIPTION} - {dataset} dataset", 
                      the_contributor=COCO_CONTRIBUTOR, 
                      the_url=COCO_URL)
        
        coco_license = coco.license(the_name=COCO_LICENSE_NAME, the_url=COCO_LICENSE_URL)
        coco_license_id = coco.insert_license(coco_license)

        coco_category = coco.category(the_name='swimming pool', the_supercategory='facility')                      
        coco_category_id = coco.insert_category(coco_category)
        
        tmp_tiles_gdf = sp_tiles_with_img_md_gdf[sp_tiles_with_img_md_gdf.dataset == dataset].dropna()
        tmp_tiles_gdf = tmp_tiles_gdf.to_crs(epsg=3857)
        
        assert(labels_gdf.crs == tmp_tiles_gdf.crs)
    
        results = Parallel(n_jobs=N_JOBS) \
                        (delayed(get_COCO_image_and_segmentations) \
                        (tile, labels_gdf, coco_license_id, OUTPUT_DIR) \
                        for tile in tqdm( tmp_tiles_gdf.sort_index().itertuples(), total=len(tmp_tiles_gdf) ))
        
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
    logger.info(f"cd {OUTPUT_DIR}; tar -cvf images-{TILE_SIZE}.tar COCO_{{trn,val,tst}}.json && tar -rvf images-{TILE_SIZE}.tar {{trn,val,tst}}-images-256 && gzip < images-{TILE_SIZE}.tar > images-{TILE_SIZE}.tar.gz && rm images-{TILE_SIZE}.tar; cd -")
    
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")
