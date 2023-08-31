#!/bin/python
# -*- coding: utf-8 -*-

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import inspect
import time
import argparse
import yaml
from loguru import logger

import geopandas as gpd
import pandas as pd
import rasterio
from sklearn.cluster import KMeans


# the following allows us to import modules from within this file's parent folder
sys.path.insert(0, '.')

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}", level="INFO")


if __name__ == "__main__":

    # Chronometer
    tic = time.time()
    logger.info('Starting...')

    # argument parser
    parser = argparse.ArgumentParser(description="The script filters the detection of potential Mineral Extraction Sites obtained with the object-detector scripts")
    parser.add_argument('config_file', type=str, help='input geojson path')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    YEAR = cfg['year']
    INPUT = cfg['input']
    LABELS_SHPFILE = cfg['labels_shapefile']
    DEM = cfg['dem']
    SCORE = cfg['score']
    AREA = cfg['area']
    ELEVATION = cfg['elevation']
    DISTANCE = cfg['distance']
    OUTPUT = cfg['output']

    written_files = [] 

    # Convert input detection to a geo dataframe 
    aoi = gpd.read_file(LABELS_SHPFILE)
    aoi = aoi.to_crs(epsg=2056)
    
    input = gpd.read_file(INPUT)
    input = input.to_crs(2056)
    total = len(input)
    logger.info(f"Total input = {total}")

    # Discard polygons detected above the threshold elevalation and 0 m 
    r = rasterio.open(DEM)
    row, col = r.index(input.centroid.x, input.centroid.y)
    values = r.read(1)[row, col]
    input['elev'] = values   
    input = input[input.elev < ELEVATION]
    row, col = r.index(input.centroid.x, input.centroid.y)
    values = r.read(1)[row, col]
    input['elev'] = values  

    input = input[input.elev != 0]
    te = len(input)
    logger.info(f"{str(total - te)} predictions removed by elevation threshold: {str(ELEVATION)} m")

    # Centroid of every prediction polygon
    centroids = gpd.GeoDataFrame()
    centroids.geometry = input.representative_point()

    # KMeans Unsupervised Learning
    centroids = pd.DataFrame({'x': centroids.geometry.x, 'y': centroids.geometry.y})
    k = int((len(input)/3) + 1)
    cluster = KMeans(n_clusters=k, algorithm='auto', random_state=1)
    model = cluster.fit(centroids)
    labels = model.predict(centroids)
    logger.info(f"KMeans algorithm computed with k = {str(k)}")

    # Dissolve and Aggregate (keep the max value of aggregate attributes)
    input['cluster'] = labels

    input = input.dissolve(by='cluster', aggfunc='max')
    total = len(input)

    # Filter dataframe by score value
    input = input[input['score'] > SCORE]
    sc = len(input)
    logger.info(f"{str(total - sc)} predictions removed by score threshold: {str(SCORE)}")

    # Clip prediction to AOI
    input = gpd.clip(input, aoi)

    # Create empty data frame
    geo_merge = gpd.GeoDataFrame()
    # Merge close labels using buffer and unions
    geo_merge = input.buffer(+DISTANCE, resolution = 2)
    geo_merge = geo_merge.geometry.unary_union
    geo_merge = gpd.GeoDataFrame(geometry=[geo_merge], crs = input.crs)  
    geo_merge = geo_merge.explode(index_parts=True).reset_index(drop=True)
    geo_merge = geo_merge.buffer(-DISTANCE, resolution=2)

    td = len(geo_merge)
    logger.info(f"{str(sc - td)} difference to clustered predictions after union (distance {str(DISTANCE)})")

    # Discard polygons with area under the threshold 
    geo_merge = geo_merge[geo_merge.area > AREA]
    ta = len(geo_merge)
    logger.info(f"{str(td - ta)} difference to clustered predictions after union (distance {str(AREA)})")

    # Preparation of a geo df 
    data = {'id': geo_merge.index,'area': geo_merge.area, 'centroid_x': geo_merge.centroid.x, 'centroid_y': geo_merge.centroid.y, 'geometry': geo_merge}
    geo_tmp = gpd.GeoDataFrame(data, crs=input.crs)

    # Get the averaged prediction score of the merged polygons  
    intersection = gpd.sjoin(geo_tmp, input, how='inner')
    intersection['id'] = intersection.index
    score_final=intersection.groupby(['id']).mean(numeric_only=True)
    # Formatting the final geo df 
    data = {'id_feature': geo_merge.index,'score': score_final['score'] , 'area': geo_merge.area, 'centroid_x': geo_merge.centroid.x, 'centroid_y': geo_merge.centroid.y, 'geometry': geo_merge}
    geo_final = gpd.GeoDataFrame(data, crs=input.crs)
    logger.info(f"{len(geo_final)} predictions remaining")

    # Format the ooutput name of the filtered prediction  
    feature = OUTPUT.replace('{score}', str(SCORE)).replace('0.', '0dot') \
        .replace('{year}', str(int(YEAR)))\
        .replace('{area}', str(int(AREA)))\
        .replace('{elevation}', str(int(ELEVATION))) \
        .replace('{distance}', str(int(DISTANCE)))
    geo_final.to_file(feature, driver='GeoJSON')

    written_files.append(feature)
    logger.info(f"...done. A file was written: {feature}")  

    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()