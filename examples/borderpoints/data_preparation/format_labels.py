import os
import sys
from loguru import logger

import geopandas as gpd

from helpers.misc import format_logger

logger = format_logger(logger)


def main(path_points_poly, output_dir='outputs'):
    """Format the labels based on the GT

    Args:
        path_points_poly (str): path to the GT file
        output_dir (str, optional): path to the output dir. Defaults to 'outputs'.

    Returns:
        pts_gdf: GeoDataFrame with the formatted labels
        written_files: list with the path of the written files
    """

    written_files = [] 

    os.makedirs(output_dir, exist_ok=True)

    logger.info('Format the labels...')
    all_pts_gdf = gpd.read_file(path_points_poly)

    all_pts_gdf.drop(columns=['Shape_Leng', 'Shape_Area'], inplace=True, errors='ignore')
    all_pts_gdf['CATEGORY'] = [str(int(code)) + color if color else 'undetermined' for code, color in zip(all_pts_gdf.Code_type_, all_pts_gdf.Couleur)] 
    all_pts_gdf['SUPERCATEGORY'] = 'border points'
    pts_gdf = all_pts_gdf[all_pts_gdf.CATEGORY!='3n'].copy()

    logger.info('Export the labels...')
    filepath = os.path.join(output_dir, 'ground_truth_labels.gpkg')
    pts_gdf.to_file(filepath)
    written_files.append(filepath)

    logger.success('Done formatting the labels!')
    return pts_gdf, written_files