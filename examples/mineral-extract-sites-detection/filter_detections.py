#!/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import time
import argparse
import yaml

import geopandas as gpd
import pandas as pd
import rasterio
from sklearn.cluster import KMeans

sys.path.insert(0, '.')
from helpers import misc
from helpers.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


if __name__ == "__main__":

    # Chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script filters the detection of potential Mineral Extraction Sites obtained with the object-detector scripts")
    parser.add_argument('config_file', type=str, help='input geojson path')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    YEAR = cfg['year']
    DETECTIONS = cfg['detections']
    SHPFILE = cfg['shapefile']
    DEM = cfg['dem']
    SCORE = cfg['score']
    AREA = cfg['area']
    ELEVATION = cfg['elevation']
    DISTANCE = cfg['distance']
    OUTPUT = cfg['output']

    written_files = [] 

    # Convert input detection to a geo dataframe 
    aoi = gpd.read_file(SHPFILE)
    aoi = aoi.to_crs(epsg=2056)
    
    detections = gpd.read_file(DETECTIONS)
    detections = detections.to_crs(2056)
    total = len(detections)
    logger.info(f"{total} input shapes")

    # Discard polygons detected above the threshold elevalation and 0 m 
    r = rasterio.open(DEM)
    row, col = r.index(detections.centroid.x, detections.centroid.y)
    values = r.read(1)[row, col]
    detections['elevation'] = values   
    detections = detections[detections.elevation < ELEVATION]
    row, col = r.index(detections.centroid.x, detections.centroid.y)
    values = r.read(1)[row, col]
    detections['elevation'] = values  

    detections = detections[detections.elevation != 0]
    te = len(detections)
    logger.info(f"{total - te} detections were removed by elevation threshold: {ELEVATION} m")

    # Centroid of every detection polygon
    centroids = gpd.GeoDataFrame()
    centroids.geometry = detections.representative_point()

    # KMeans Unsupervised Learning
    centroids = pd.DataFrame({'x': centroids.geometry.x, 'y': centroids.geometry.y})
    k = int((len(detections)/3) + 1)
    cluster = KMeans(n_clusters=k, algorithm='auto', random_state=1)
    model = cluster.fit(centroids)
    labels = model.predict(centroids)
    logger.info(f"KMeans algorithm computed with k = {k}")

    # Dissolve and aggregate (keep the max value of aggregate attributes)
    detections['cluster'] = labels

    detections = detections.dissolve(by='cluster', aggfunc='max')
    total = len(detections)

    # Filter dataframe by score value
    detections = detections[detections['score'] > SCORE]
    sc = len(detections)
    logger.info(f"{total - sc} detections were removed by score threshold: {SCORE}")

    # Clip detection to AoI
    detections = gpd.clip(detections, aoi)

    # Merge close labels using buffer and unions
    detections_merge = gpd.GeoDataFrame()
    detections_merge = detections.buffer(+DISTANCE, resolution=2)
    detections_merge = detections_merge.geometry.unary_union
    detections_merge = gpd.GeoDataFrame(geometry=[detections_merge], crs=detections.crs)  
    detections_merge = detections_merge.explode(index_parts=True).reset_index(drop=True)
    detections_merge = detections_merge.buffer(-DISTANCE, resolution=2)

    td = len(detections_merge)
    logger.info(f"{td} clustered detections remains after shape union (distance {DISTANCE})")

    # Discard polygons with area under the threshold 
    detections_merge = detections_merge[detections_merge.area > AREA]
    ta = len(detections_merge)
    logger.info(f"{td - ta} detections were removed to after union (distance {AREA})")

    # Preparation of a geo df 
    data = {'id': detections_merge.index,'area': detections_merge.area, 'centroid_x': detections_merge.centroid.x, 'centroid_y': detections_merge.centroid.y, 'geometry': detections_merge}
    geo_tmp = gpd.GeoDataFrame(data, crs=detections.crs)

    # Get the averaged detection score of the merged polygons  
    intersection = gpd.sjoin(geo_tmp, detections, how='inner')
    intersection['id'] = intersection.index
    score_final = intersection.groupby(['id']).mean(numeric_only=True)

    # Formatting the final geo df 
    data = {'id_feature': detections_merge.index,'score': score_final['score'] , 'area': detections_merge.area, 'centroid_x': detections_merge.centroid.x, 'centroid_y': detections_merge.centroid.y, 'geometry': detections_merge}
    detections_final = gpd.GeoDataFrame(data, crs=detections.crs)
    logger.info(f"{len(detections_final)} detections remaining after filtering")

    # Formatting the output name of the filtered detection  
    feature = OUTPUT.replace('{score}', str(SCORE)).replace('0.', '0dot') \
        .replace('{year}', str(int(YEAR)))\
        .replace('{area}', str(int(AREA)))\
        .replace('{elevation}', str(int(ELEVATION))) \
        .replace('{distance}', str(int(DISTANCE)))
    detections_final.to_file(feature, driver='GeoJSON')

    written_files.append(feature)
    logger.success(f"{DONE_MSG} A file was written: {feature}")  

    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()