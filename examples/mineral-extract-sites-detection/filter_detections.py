import argparse
import os
import sys
import time
import yaml

import geopandas as gpd
import numpy as np
import rasterio as rio

from rasterio.mask import mask

sys.path.insert(0, '../..')
import helpers.misc as misc
from helpers.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


def check_gdf_len(gdf):
    """Check if the GeoDataFrame is empty. If True, exit the script

    Args:
        gdf (GeoDataFrame): detection polygons
    """

    if len(gdf) == 0:
        logger.error("No detections left in the dataframe. Exit script.")
        sys.exit(1)


def get_mean_slope_for_detection(detection_geometry, slope_raster):
    """
    Calculate the mean slope for the detection polygon.

    Args:
        detection_geometry (Shape): polygon shape of the detections.
        slope_raster (raster): raster slope.

    Returns:
        mean_slope: float value of the polygon mean slope
        """

    # Mask the slope raster with the detection geometry, crop to the detection area
    out_image, out_transform = mask(slope_raster, [detection_geometry], crop=True)
    # Mask the slope values that are outside the detection area (i.e., no data or zeros)
    out_image = np.ma.masked_where(out_image == slope_raster.nodata, out_image)
    # Calculate the mean slope for the masked area
    mean_slope = out_image.mean()  # Mean of the valid, non-masked slope values
   
    return mean_slope


def none_if_undefined(cfg, key):
    
    return cfg[key] if key in cfg.keys() else None


if __name__ == "__main__":

    # Chronometer
    tic = time.time()
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser(description="The script post-processes the detections obtained with the object-detector")
    parser.add_argument('config_file', type=str, help='input geojson path')
    args = parser.parse_args()

    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    WORKING_DIR = cfg['working_directory']
    AOI = cfg['aoi']
    DETECTIONS = cfg['detections']
    DEM = misc.none_if_undefined(cfg, 'dem')
    SLOPE = misc.none_if_undefined(cfg, 'slope')
    SCORE_THD = cfg['score_threshold']
    AREA_THD = cfg['area_threshold']
    ELEVATION_THD = misc.none_if_undefined(cfg, 'elevation_threshold')
    MIN_SLOPE_THD = misc.none_if_undefined(cfg, 'min_slope_threshold')
    MAX_SLOPE_THD = misc.none_if_undefined(cfg, 'max_slope_threshold')

    os.chdir(WORKING_DIR)
    logger.info(f'Working directory set to {WORKING_DIR}')

    written_files = [] 

    # Convert input detections to a geodataframe 
    detections_gdf = gpd.read_file(DETECTIONS)
    detections_gdf = detections_gdf.to_crs(2056)
    if 'tag' in detections_gdf.keys():
        detections_gdf = detections_gdf[detections_gdf['tag']!='FN']
    detections_gdf['area'] = detections_gdf.geometry.area 
    detections_gdf['det_id'] = detections_gdf.index
    total = len(detections_gdf)
    logger.info(f"{total} detections")

    detections_gdf = misc.check_validity(detections_gdf, correct=True)

    aoi_gdf = gpd.read_file(AOI)
    aoi_gdf = aoi_gdf.to_crs(2056)

    # Discard polygons detected at/below 0 m and above the threshold elevation and above a given slope
    if DEM:
        dem_raster = rio.open(DEM)

        row, col = dem_raster.index(detections_gdf.centroid.x, detections_gdf.centroid.y)
        elevation = dem_raster.read(1)[row, col]
        detections_gdf['elevation'] = elevation 
        detections_gdf['centroid_x'] = detections_gdf.centroid.x
        detections_gdf['centroid_y'] = detections_gdf.centroid.y
        check_gdf_len(detections_gdf)
        detections_gdf = detections_gdf[(detections_gdf.elevation != 0) & (detections_gdf.elevation < ELEVATION_THD)]
        total = total - len(detections_gdf)
        logger.info(f"{total} detections were removed by elevation threshold: {ELEVATION_THD} m")

    # Discard polygons detected above the slope threshold
    if SLOPE:
        check_gdf_len(detections_gdf)
        
        slope_raster = rio.open(SLOPE)

        detections_gdf['mean_slope'] = detections_gdf.geometry.apply(lambda geom: get_mean_slope_for_detection(geom, slope_raster))
        detections_gdf = detections_gdf[(detections_gdf['mean_slope'] > MIN_SLOPE_THD) & (detections_gdf['mean_slope'] < MAX_SLOPE_THD)]  
        total = total - len(detections_gdf)
        logger.info(f"{total} detections were removed by slope threshold: [{MIN_SLOPE_THD} - {MAX_SLOPE_THD}] degrees")

    # Filter dataframe by score value
    check_gdf_len(detections_gdf)
    detections_score_gdf = detections_gdf[detections_gdf.score > SCORE_THD]
    total = len(detections_gdf) - len(detections_score_gdf)
    logger.info(f"{total} detections were removed by score filtering (score threshold = {SCORE_THD})")

    # Discard polygons with area under a given threshold 
    check_gdf_len(detections_gdf)
    detections_area_gdf = detections_score_gdf.copy()
    detections_area_gdf['area'] = detections_score_gdf.area
    nb_sjoin = len(detections_area_gdf)
    detections_area_gdf = detections_area_gdf[detections_area_gdf.area > AREA_THD]
    nb_area = len(detections_area_gdf)
    logger.info(f"{nb_sjoin - nb_area} detections were removed by area filtering (area threshold = {AREA_THD} m2)")
    check_gdf_len(detections_area_gdf)

    # Final gdf
    detections_gdf = detections_area_gdf.reset_index(drop=True)
    logger.info(f"{len(detections_gdf)} detections remaining after filtering")

    # Formatting the output name of the filtered detection  
    feature = f'{DETECTIONS[:-5]}_score-{SCORE_THD}_area-{str(AREA_THD)}_elevation-{str(ELEVATION_THD)}_slope-{str(MIN_SLOPE_THD)}-{str(MAX_SLOPE_THD)}'.replace('0.', '0dot') + '.gpkg'
    detections_gdf.to_file(feature)

    written_files.append(feature)
    logger.success(f"{DONE_MSG} A file was written: {feature}")  

    logger.info("The following files were written. Let's check them out!")
    for written_file in written_files:
        logger.info(written_file)

    # Stop chronometer  
    toc = time.time()
    logger.info(f"Nothing left to be done: exiting. Elapsed time: {(toc-tic):.2f} seconds")

    sys.stderr.flush()