import os
import sys
import argparse
import re
import yaml
from loguru import logger
from time import time
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
import morecantile

import fct_misc
sys.path.insert(1, '../..')
from helpers.misc import format_logger

logger = format_logger(logger)

tic = time()	
logger.info('Starting...')

# Get the configuration
parser = argparse.ArgumentParser(description="This script prepares datasets for the determination of the road cover type.")
parser.add_argument('config_file', type=str, help='a YAML config file')
args = parser.parse_args()

logger.info(f"Using {args.config_file} as config file.")
with open(args.config_file) as fp:
    cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

# Define constants -----------------------------------------

WORKING_DIRECTORY = cfg['working_directory']

# Task to do
DETERMINE_ROAD_SURFACES = cfg['tasks']['determine_roads_surfaces']
GENERATE_TILES_INFO=cfg['tasks']['generate_tiles_info']
GENERATE_LABELS=cfg['tasks']['generate_labels']

if not (DETERMINE_ROAD_SURFACES or GENERATE_TILES_INFO or GENERATE_LABELS) :
    logger.info('Nothing to do. Exiting!')
    sys.exit(0)

else:
    INPUT = cfg['input']
    INPUT_DIR =INPUT['input_folder']

    if DETERMINE_ROAD_SURFACES:	
        ROADS_IN = os.path.join(INPUT_DIR, INPUT['input_files']['roads'])	
        FORESTS = os.path.join(INPUT_DIR, INPUT['input_files']['forests'])	
    ROADS_PARAM = os.path.join(INPUT_DIR, INPUT['input_files']['roads_param'])
    AOI = os.path.join(INPUT_DIR, INPUT['input_files']['aoi'])

    OUTPUT_DIR = cfg['output_folder']
    
    # Based on the metadata	
    # Remove places, motorail, ferry, marked trace, climbing path and provisory pathes of soft mobility.
    NOT_ROAD = [12, 13, 14, 19, 22, 23]
    # Only keep roads and bridges with an artificial or natural suface
    KUNSTBAUTE_TO_KEEP = [100, 200]
    BELAGSART_TO_KEEP = [100, 200]

    if 'restricted_aoi_training' in INPUT['input_files'].keys():
        RESTRICTED_AOI_TRAIN = os.path.join(INPUT_DIR, INPUT['input_files']['restricted_aoi_training'])
    else:
        RESTRICTED_AOI_TRAIN = False

    if GENERATE_TILES_INFO or GENERATE_LABELS:
        ZOOM_LEVEL = cfg['zoom_level']

os.chdir(WORKING_DIRECTORY)

path_shp_gpkg = fct_misc.ensure_dir_exists(os.path.join(OUTPUT_DIR, 'shp'))
path_json = fct_misc.ensure_dir_exists(os.path.join(OUTPUT_DIR,'json_inputs'))

# Define functions --------------------------------------------

def determine_category(row):
    if row['BELAGSART'] == 100:
        return 'artificial'
    if row['BELAGSART'] == 200:
        return 'natural'
    else:
        return 'else'

# Information treatment ------------------------------------------

if DETERMINE_ROAD_SURFACES:
    logger.info('Importing files...')

    ## Geodata
    roads = gpd.read_file(ROADS_IN)
    forests = gpd.read_file(FORESTS)

    ## Other informations
    roads_parameters = pd.read_excel(ROADS_PARAM)

    # Filter the roads to consider
    logger.info('Filtering the considered roads...')
    
    roads_of_interest = roads[~roads['OBJEKTART'].isin(NOT_ROAD)]
    uncovered_roads = roads_of_interest[roads_of_interest['KUNSTBAUTE'].isin(KUNSTBAUTE_TO_KEEP)]

    roads_parameters_filtered = roads_parameters[~roads_parameters['Width'].isna()].copy()
    roads_parameters_filtered.drop_duplicates(subset='GDB-Code',inplace=True)       # Keep first by default 

    uncovered_roads = uncovered_roads.merge(roads_parameters_filtered[['GDB-Code','Width']], how='inner',left_on='OBJEKTART',right_on='GDB-Code')

    uncovered_roads.drop(columns=[
                                'DATUM_AEND', 'DATUM_ERST', 'ERSTELLUNG', 'ERSTELLU_1', 'UUID',
                                'REVISION_J', 'REVISION_M', 'GRUND_AEND', 'HERKUNFT', 'HERKUNFT_J',
                                'HERKUNFT_M', 'REVISION_Q', 'WANDERWEGE', 'VERKEHRSBE', 
                                'BEFAHRBARK', 'EROEFFNUNG', 'STUFE', 'RICHTUNGSG', 
                                'KREISEL', 'EIGENTUEME', 'VERKEHRS_1', 'NAME',
                                'TLM_STRASS', 'STRASSENNA', 'SHAPE_Leng'
                                ], inplace=True)

    logger.info('Determining the surface of the roads from lines...')

    uncovered_roads['road_len'] = round(uncovered_roads.length, 3)


    logger.info('-- Transform the roads into polygons...')

    buffered_roads = uncovered_roads.copy()
    buffered_roads['geometry'] = uncovered_roads.buffer(uncovered_roads['Width']/2, cap_style=2)

    # Erease artifact polygons produced by roundabouts
    buff_geometries = []
    for geom in buffered_roads['geometry'].values:
        if geom.geom_type == 'MultiPolygon':
            buff_geometries.append(max(geom.geoms, key=lambda a: a.area))
        else:
            buff_geometries.append(geom)
        
    buffered_roads['geometry'] = buff_geometries

    # Erase overlapping zones of roads buffer
    logger.info('-- Comparing roads for intersections of different classes to remove...')

    buffered_roads['saved_geom'] = buffered_roads.geometry
    joined_roads_in_aoi = gpd.sjoin(buffered_roads,buffered_roads[['OBJECTID','OBJEKTART','saved_geom','geometry']],how='left', lsuffix='1', rsuffix='2')

    ### Drop excessive rows
    intersected = joined_roads_in_aoi[joined_roads_in_aoi['OBJECTID_2'].notna()].copy()
    intersected_not_itself = intersected[intersected['OBJECTID_1']!=intersected['OBJECTID_2']].copy()
    intersected_roads = intersected_not_itself.drop_duplicates(subset=['OBJECTID_1','OBJECTID_2'])

    intersected_roads.reset_index(inplace=True, drop=True)

    ### Sort the roads so that the widest ones come first
    intersected_roads.loc[intersected_roads['OBJEKTART_1']==20,'OBJEKTART_1'] = 8.5
    intersected_roads.loc[intersected_roads['OBJEKTART_1']==21,'OBJEKTART_1'] = 2.5

    intersect_other_width=intersected_roads[intersected_roads['OBJEKTART_1']<intersected_roads['OBJEKTART_2']].copy()

    intersect_other_width.sort_values(by=['OBJEKTART_1'],inplace=True)
    intersect_other_width.loc[intersect_other_width['OBJEKTART_1']==8.5,'OBJEKTART_1'] = 20
    intersect_other_width.loc[intersect_other_width['OBJEKTART_1']==2.5,'OBJEKTART_1'] = 21

    intersect_other_width.sort_values(by=['KUNSTBAUTE'], ascending=False, inplace=True, ignore_index=True)

    ### Suppress the overlapping intersection
    ### from https://stackoverflow.com/questions/71738629/expand-polygons-in-geopandas-so-that-they-do-not-overlap-each-other
    corr_overlap = buffered_roads.copy()

    for idx in tqdm(intersect_other_width.index, total=intersect_other_width.shape[0],
                desc='-- Suppressing the overlap of roads with different width'):
        
        poly1_id = corr_overlap.index[corr_overlap['OBJECTID'] == intersect_other_width.loc[idx,'OBJECTID_1']].to_numpy().astype(int)[0]
        poly2_id = corr_overlap.index[corr_overlap['OBJECTID'] == intersect_other_width.loc[idx,'OBJECTID_2']].to_numpy().astype(int)[0]
        
        corr_overlap=fct_misc.polygons_diff_without_artifacts(corr_overlap, poly1_id, poly2_id, keep_everything=True)

    corr_overlap.drop(columns=['saved_geom'],inplace=True)
    corr_overlap.set_crs(epsg=2056, inplace=True)

    logger.info('-- Excluding roads under forest canopy ...')

    fct_misc.test_crs(corr_overlap.crs, forests.crs)

    forests['buffered_geom'] = forests.buffer(3)
    forests.drop(columns=['geometry'], inplace=True)
    forests.rename(columns={'buffered_geom':'geometry'}, inplace=True)

    non_forest_roads = corr_overlap.copy()
    non_forest_roads = non_forest_roads.overlay(forests[['UUID','geometry']],how='difference')

    non_forest_roads.drop(columns=['GDB-Code'],inplace=True)
    non_forest_roads.rename(columns={'Width':'road_width'}, inplace=True)

    logger.success('Done determining the surface of the roads from lines!')


if (GENERATE_TILES_INFO or GENERATE_LABELS) and (not DETERMINE_ROAD_SURFACES):

    ROADS_FOR_LABELS = cfg['processed_input']['roads_for_labels']

    logger.info('Importing files...')
    if 'layer' in cfg['processed_input'].keys():
        non_forest_roads = gpd.read_file(os.path.join(path_shp_gpkg, ROADS_FOR_LABELS), layer=cfg['processed_input']['layer'])
    else:
        non_forest_roads = gpd.read_file(os.path.join(path_shp_gpkg, ROADS_FOR_LABELS))
    roads_parameters = pd.read_excel(ROADS_PARAM)

if GENERATE_TILES_INFO:
    print()
    aoi=gpd.read_file(AOI)

    logger.info('Determination of the information for the tiles to consider...')

    roads_parameters_filtered = roads_parameters[roads_parameters['to keep']=='yes'].copy()
    roads_parameters_filtered.drop_duplicates(subset='GDB-Code',inplace=True)       # Keep first by default 

    roads_of_interest = non_forest_roads.merge(roads_parameters_filtered[['GDB-Code']], how='right',left_on='OBJEKTART',right_on='GDB-Code')
    roads_to_exclude = roads_of_interest[~roads_of_interest['BELAGSART'].isin(BELAGSART_TO_KEEP)]
    road_id_to_exclude = roads_to_exclude['OBJECTID'].unique().tolist()

    aoi_geom=gpd.GeoDataFrame({'id': [0], 'geometry': [aoi['geometry'].unary_union]}, crs=aoi.crs)

    try:
        assert(aoi_geom.crs==roads_of_interest.crs)
    except Exception:
        aoi_geom.to_crs(crs=roads_of_interest.crs, inplace=True)
    roi_in_aoi = roads_of_interest.overlay(aoi_geom, how='intersection')

    del roads_parameters, roads_parameters_filtered, roads_of_interest

    roi_in_aoi = fct_misc.test_valid_geom(roi_in_aoi, gdf_obj_name='roads')

    roi_in_aoi.drop(columns=['BELAGSART', 'road_width', 'OBJEKTART',
                            'KUNSTBAUTE', 'GDB-Code', 'road_len'], inplace=True)
    
    roi_4326 = roi_in_aoi.to_crs(epsg=4326)
    valid_roi_4326 = fct_misc.test_valid_geom(roi_4326, correct=True, gdf_obj_name="reprojected roads")
    bboxes_extent_4326 = valid_roi_4326.unary_union.bounds

    # cf. https://developmentseed.org/morecantile/usage/
    tms = morecantile.tms.get("WebMercatorQuad")    # epsg:3857

    logger.info('-- Generating the tiles...')
    epsg3857_tiles_gdf = gpd.GeoDataFrame.from_features([tms.feature(x, projected=True) for x in tqdm(tms.tiles(*bboxes_extent_4326, zooms=[ZOOM_LEVEL]))])
    epsg3857_tiles_gdf.set_crs(epsg=3857, inplace=True)

    roi_in_aoi_3857 = roi_in_aoi.to_crs(epsg=3857)
    roi_in_aoi_3857.rename(columns={'FID': 'id_aoi'}, inplace=True)

    logger.info('-- Checking for intersections with the restricted area of interest...')
    fct_misc.test_crs(tms.crs, roi_in_aoi_3857.crs)

    tiles_in_raoi_w_unknown = gpd.sjoin(epsg3857_tiles_gdf, roi_in_aoi_3857, how='inner')

    tile_id_to_exclude = []
    for road_id in road_id_to_exclude:
        tiles_intersects_roads = tiles_in_raoi_w_unknown[tiles_in_raoi_w_unknown['OBJECTID']==road_id].copy()
        if not tiles_intersects_roads.empty:
            tile_id_to_exclude.extend(tiles_intersects_roads['title'].unique().tolist())
    tile_id_to_exclude = list(dict.fromkeys(tile_id_to_exclude))
    logger.warning(f"{len(tile_id_to_exclude)} tiles are to be excluded, because they contain unknown roads.")

    tiles_in_raoi_w_unknown.drop_duplicates('title', inplace=True)
    tiles_in_raoi_w_unknown.drop(columns=['grid_name', 'grid_crs', 'index_right'], inplace=True)
    tiles_in_raoi_w_unknown.reset_index(drop=True, inplace=True)

    tiles_in_restricted_aoi = tiles_in_raoi_w_unknown[~tiles_in_raoi_w_unknown['title'].isin(tile_id_to_exclude)].copy()
    tiles_in_restricted_aoi.drop(columns=['OBJECTID'], inplace=True)
    tiles_in_restricted_aoi.reset_index(drop=True, inplace=True)
    logger.warning(f"{tiles_in_raoi_w_unknown.shape[0] - tiles_in_restricted_aoi.shape[0]} have been excluded.")

    logger.info('-- Setting a formatted id...')
    xyz=[]
    for idx in tiles_in_restricted_aoi.index:
        xyz.append([re.sub('\D','',coor) for coor in tiles_in_restricted_aoi.loc[idx,'title'].split(',')])

    tiles_in_restricted_aoi['id'] = ['('+ x +', '+y+', '+z + ')' for x, y, z in xyz]

    logger.success('Done determining the tiles!')

if GENERATE_LABELS:
    print()
    logger.info('Generating the labels for the object detector...')

    if not GENERATE_TILES_INFO:
        tiles_in_restricted_aoi_4326 = gpd.read_file(os.path.join(path_json, 'tiles_aoi.geojson'))
        tiles_in_restricted_aoi_4326 = tiles_in_restricted_aoi_4326[['title','id','geometry']]
    else:
        tiles_in_restricted_aoi_4326 = tiles_in_restricted_aoi.to_crs(epsg=4326)

    if RESTRICTED_AOI_TRAIN:
        logger.info('A subset of the AOI is used for the training.')
        restricted_aoi_training = gpd.read_file(RESTRICTED_AOI_TRAIN)
        restricted_aoi_training_4326 = restricted_aoi_training.to_crs(epsg=4326)
        tiles_in_restricted_aoi_4326 = gpd.sjoin(tiles_in_restricted_aoi_4326,
                                            restricted_aoi_training_4326[['KBNUM', 'geometry']],
                                            how='inner')
        tiles_in_restricted_aoi_4326.drop(columns=['index_right'], inplace=True)
    
    # Attribute object category and supercategory to labels
    labels_gdf_2056=non_forest_roads[non_forest_roads['BELAGSART'].isin(BELAGSART_TO_KEEP)].copy()
    labels_gdf_2056['CATEGORY'] = labels_gdf_2056.apply(lambda row: determine_category(row), axis=1)
    labels_gdf_2056['SUPERCATEGORY'] = 'road'
    labels_gdf = labels_gdf_2056.to_crs(epsg=4326)
    labels_gdf = fct_misc.test_valid_geom(labels_gdf, correct=True, gdf_obj_name='labels')

    logger.info('Labels on tiles...')
    fct_misc.test_crs(labels_gdf.crs, tiles_in_restricted_aoi_4326.crs)

    GT_labels_gdf = gpd.sjoin(labels_gdf, tiles_in_restricted_aoi_4326, how='inner', predicate='intersects')

    # Exclude tile with undetermined roads
    tiles_w_undet_road = GT_labels_gdf[GT_labels_gdf['CATEGORY']=='else']['id'].unique().tolist()
    GT_labels_gdf = GT_labels_gdf[~GT_labels_gdf['id'].isin(tiles_w_undet_road)]

    # the following two lines make sure that no object is counted more than once in case it intersects multiple tiles
    GT_labels_gdf = GT_labels_gdf[labels_gdf.columns]
    GT_labels_gdf.drop_duplicates(inplace=True)
    OTH_labels_gdf = labels_gdf[ ~labels_gdf.index.isin(GT_labels_gdf.index)]

    try:
        assert( len(labels_gdf) == len(GT_labels_gdf) + len(OTH_labels_gdf) ),\
            f"Something went wrong when splitting labels into Ground Truth Labels and Other Labels." +\
            f" Total no. of labels = {len(labels_gdf)}; no. of Ground Truth Labels = {len(GT_labels_gdf)}; no. of Other Labels = {len(OTH_labels_gdf)}"
    except Exception as e:
        logger.warning(e)
        sys.exit(1)

    print()
    logger.info(f'{GT_labels_gdf.shape[0]} labels are saved as ground truth.')
    logger.info(f'   - {GT_labels_gdf[GT_labels_gdf.BELAGSART==100].shape[0]} labels are tagged artificial')
    logger.info(f'   - {GT_labels_gdf[GT_labels_gdf.BELAGSART==200].shape[0]} labels are tagged natural.')
    logger.info(f'{OTH_labels_gdf.shape[0]} labels are saved as the other lables.')
    print()

    logger.success('Done generating the labels for the object detector...')

    # In the current case, OTH_labels_gdf should be empty


# Save results ------------------------------------------------------------------
logger.info('Saving files...')

written_files=[]

if DETERMINE_ROAD_SURFACES:
    filepath = os.path.join(path_shp_gpkg, 'roads_for_OD.shp')
    non_forest_roads.to_file(filepath)
    written_files.append(filepath)

if GENERATE_TILES_INFO:
    # geojson only supports epsg:4326
    tiles_4326 = tiles_in_restricted_aoi.to_crs(epsg=4326)
    filepath = os.path.join(path_json, 'tiles_aoi.geojson')
    tiles_4326.to_file(filepath, driver='GeoJSON')
    written_files.append(filepath)

if GENERATE_LABELS:
    filepath = os.path.join(path_json, 'ground_truth_labels.geojson')
    GT_labels_gdf.to_file(filepath, driver='GeoJSON')
    written_files.append(filepath)

    if not OTH_labels_gdf.empty:
        filepath = os.path.join(path_json, f'other_labels.geojson')
        OTH_labels_gdf.to_file(filepath, driver='GeoJSON')
        written_files.append(filepath)

logger.info('All done!')
logger.info('Written files:')
for file in written_files:
    logger.info(file)

toc = time()	
logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")