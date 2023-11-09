#!/bin/python
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import geopandas as gpd
import json
import pandas as pd
from loguru import logger

from shapely.affinity import scale
from rasterio.transform import from_bounds


class BadFileExtensionException(Exception):
    "Raised when the file extension is different from the expected one"
    pass



def bounds_to_bbox(bounds):
    
    xmin = bounds[0]
    ymin = bounds[1]
    xmax = bounds[2]
    ymax = bounds[3]
    
    bbox = f"{xmin},{ymin},{xmax},{ymax}"
    
    return bbox



def clip_labels(labels_gdf, tiles_gdf, fact=0.99):

    tiles_gdf['tile_geometry'] = tiles_gdf['geometry']
        
    assert(labels_gdf.crs == tiles_gdf.crs)
    
    labels_tiles_sjoined_gdf = gpd.sjoin(labels_gdf, tiles_gdf, how='inner', predicate='intersects')
    
    def clip_row(row, fact=fact):
        
        old_geo = row.geometry
        scaled_tile_geo = scale(row.tile_geometry, xfact=fact, yfact=fact)
        new_geo = old_geo.intersection(scaled_tile_geo)
        row['geometry'] = new_geo

        return row

    clipped_labels_gdf = labels_tiles_sjoined_gdf.apply(lambda row: clip_row(row, fact), axis=1)
    clipped_labels_gdf.crs = labels_gdf.crs

    clipped_labels_gdf.drop(columns=['tile_geometry', 'index_right'], inplace=True)
    clipped_labels_gdf.rename(columns={'id': 'tile_id'}, inplace=True)

    return clipped_labels_gdf



def format_logger(logger):

    logger.remove()
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
            level="INFO", filter=lambda record: record["level"].no < 25)
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - <green>{level}</green> - {message}",
            level="SUCCESS", filter=lambda record: record["level"].no < 30)
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} - <yellow>{level}</yellow> - {message}",
            level="WARNING", filter=lambda record: record["level"].no < 40)
    logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss} - <red>{level}</red> - <level>{message}</level>",
            level="ERROR")
    
    return logger


def find_category(df, cfg = {}):

    if  'category' in df.columns:
        df['CATEGORY'] = df.category
    elif 'CATEGORY' not in df.columns:
        try:
            df['CATEGORY'] = cfg['COCO_metadata']['category']['name']
        except KeyError:
            logger.error('The GT labels have no category.')
            logger.warning('Setting a fake category')
            df['CATEGORY'] = 'foo'

    if  'supercategory' in df.columns:
        df['SUPERCATEGORY'] = df.supercategory
    elif 'SUPERCATEGORY' not in df.columns:
        try:
            df['SUPERCATEGORY'] = cfg['COCO_metadata']['category']['supercategory']
        except KeyError:
            logger.error('The GT labels have no supercategory.')
            logger.warning('Setting a fake supercategory')
            df['SUPERCATEGORY'] = 'bar'
    
    return df


def get_fractional_sets(dets_gdf, labels_gdf):
    """Find the intersecting detections and labels.
    Control their class to get the TP.
    Labels non-intersection detections and labels as FP and FN respectively.
    Save the intersetions with mismatched class ids in a separate geodataframe.

    Args:
        preds_gdf (geodataframe): geodataframe of the prediction with the id "ID_DET".
        labels_gdf (geodataframe): threshold to apply on the IoU to determine TP and FP.

    Raises:
        Exception: CRS mismatch

    Returns:
        tuple:
        - geodataframe: true positive intersections between a detection and a label;
        - geodataframe: false postive detection;
        - geodataframe: false negative labels;
        - geodataframe: intersections between a detection and a label with a mismatched class id.
    """

    _dets_gdf = dets_gdf.copy()
    _labels_gdf = labels_gdf.copy()
    
    if len(_labels_gdf) == 0:
        fp_gdf = _dets_gdf.copy()
        tp_gdf = gpd.GeoDataFrame()
        fn_gdf = gpd.GeoDataFrame()
        fp_fn_tmp_gdf = gpd.GeoDataFrame()
        return tp_gdf, fp_gdf, fn_gdf, fp_fn_tmp_gdf
    
    assert(_dets_gdf.crs == _labels_gdf.crs), f"CRS Mismatch: detections' CRS = {_dets_gdf.crs}, labels' CRS = {_labels_gdf.crs}"

    # we add a dummy column to the labels dataset, which should not exist in detections too;
    # this allows us to distinguish matching from non-matching detections
    _labels_gdf['dummy_id'] = _labels_gdf.index
    
    # TRUE POSITIVES
    left_join = gpd.sjoin(_dets_gdf, _labels_gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
    
    # Test that something is detected
    candidates_tp_gdf = left_join[left_join.dummy_id.notnull()].copy()
    candidates_tp_gdf.drop_duplicates(subset=['dummy_id', 'tile_id'], inplace=True)
    candidates_tp_gdf.drop(columns=['dummy_id'], inplace=True)

    # Test that it has the right class (id starting at 1 and predicted class at 0)
    tp_gdf = candidates_tp_gdf[candidates_tp_gdf.label_class == candidates_tp_gdf.det_class+1].copy()
    fp_fn_tmp_gdf = candidates_tp_gdf[candidates_tp_gdf.label_class != candidates_tp_gdf.det_class+1].copy()

    # FALSE POSITIVES
    fp_gdf = left_join[left_join.dummy_id.isna()].copy()
    assert(len(fp_gdf[fp_gdf.duplicated()]) == 0)
    fp_gdf.drop(columns=['dummy_id'], inplace=True)
    
    # FALSE NEGATIVES
    right_join = gpd.sjoin(_dets_gdf, _labels_gdf, how='right', predicate='intersects', lsuffix='left', rsuffix='right')
    fn_gdf = right_join[right_join.score.isna()].copy()
    fn_gdf.drop_duplicates(subset=['dummy_id', 'tile_id'], inplace=True)
    
    return tp_gdf, fp_gdf, fn_gdf, fp_fn_tmp_gdf



def get_metrics(tp_gdf, fp_gdf, fn_gdf, mismatch_gdf, id_classes=0):
    """Determine the metrics based on the TP, FP and FN

    Args:
        tp_gdf (geodataframe): true positive detections
        fp_gdf (geodataframe): false positive detections
        fn_gdf (geodataframe): false negative labels
        mismatch_gdf (geodataframe): labels and detections intersecting with a mismatched class id
        id_classes (list): list of the possible class ids. Defaults to 0.
    
    Returns:
        tuple: 
            - dict: precision for each class
            - dict: recall for each class
            - float: precision;
            - float: recall;
            - float: f1 score.
    """
    
    p_k={key: None for key in id_classes}
    r_k={key: None for key in id_classes}
    
    for id_cl in id_classes:

        if tp_gdf.empty:
            TP = 0
        else:
            TP = len(tp_gdf[tp_gdf.det_class==id_cl])
            FP = len(fp_gdf[fp_gdf.det_class==id_cl]) + len(mismatch_gdf[mismatch_gdf.det_class == id_cl])
            FN = len(fn_gdf[fn_gdf.label_class==id_cl-1]) + len(mismatch_gdf[mismatch_gdf.label_class == id_cl-1])
    
        if TP == 0:
            p_k[id_cl]=0
            r_k[id_cl]=0
            continue            

        p_k[id_cl] = TP / (TP + FP)
        r_k[id_cl] = TP / (TP + FN)
        
    precision=sum(p_k.values())/len(id_classes)
    recall=sum(r_k.values())/len(id_classes)
    
    if precision==0 and recall==0:
        return p_k, r_k, 0, 0, 0
    
    f1 = 2*precision*recall/(precision+recall)
    
    return p_k, r_k, precision, recall, f1



def get_number_of_classes(coco_files_dict):

    # get the number of classes
    classes = {"file":[coco_files_dict['trn'], coco_files_dict['tst'], coco_files_dict['val']], "num_classes":[]}

    for filepath in classes["file"]:
        file_content = open(filepath)
        coco_json = json.load(file_content)
        classes["num_classes"].append(len(coco_json["categories"]))
        file_content.close()

    # test if it is the same number of classes in all datasets
    try:
        assert classes["num_classes"][0]==classes["num_classes"][1] and classes["num_classes"][0]==classes["num_classes"][2]
    except AssertionError:
        logger.critical(f"The number of classes is not equal in the training ({classes['num_classes'][0]}), testing ({classes['num_classes'][1]}), ",
                    f"and validation ({classes['num_classes'][2]}) datasets.")
        sys.exit(1)

   # set the number of classes to detect 
    num_classes=classes["num_classes"][0]
    logger.info(f"Working with {num_classes} classe{'s' if num_classes>1 else ''}.")

    return num_classes



def image_metadata_to_affine_transform(image_metadata):
    """
    This uses rasterio.
    cf. https://gdal.org/drivers/raster/wld.html#wld-esri-world-file
    """
    
    xmin = image_metadata['extent']['xmin']
    xmax = image_metadata['extent']['xmax']
    ymin = image_metadata['extent']['ymin']
    ymax = image_metadata['extent']['ymax']
    width  = image_metadata['width']
    height = image_metadata['height']
    
    affine = from_bounds(xmin, ymin, xmax, ymax, width, height)

    return affine



def image_metadata_to_world_file(image_metadata):
    """
    This uses rasterio.
    cf. https://www.perrygeo.com/python-affine-transforms.html
    """
    
    xmin = image_metadata['extent']['xmin']
    xmax = image_metadata['extent']['xmax']
    ymin = image_metadata['extent']['ymin']
    ymax = image_metadata['extent']['ymax']
    width  = image_metadata['width']
    height = image_metadata['height']
    
    affine = from_bounds(xmin, ymin, xmax, ymax, width, height)

    a = affine.a
    b = affine.b
    c = affine.c
    d = affine.d
    e = affine.e
    f = affine.f
    
    c += a/2.0 # <- IMPORTANT
    f += e/2.0 # <- IMPORTANT

    return "\n".join([str(a), str(d), str(b), str(e), str(c), str(f)+"\n"])



def img_md_record_to_tile_id(img_md_record):
    
        filename = os.path.split(img_md_record.img_file)[-1]
        
        z_x_y = filename.split('.')[0]
        z, x, y = z_x_y.split('_')
        
        return f'({x}, {y}, {z})'



def make_hard_link(row):

        if not os.path.isfile(row.img_file):
            raise FileNotFoundError(row.img_file)

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



def scale_point(x, y, xmin, ymin, xmax, ymax, width, height):

    return (x-xmin)/(xmax-xmin)*(width), (ymax-y)/(ymax-ymin)*(height)



def scale_polygon(shapely_polygon, xmin, ymin, xmax, ymax, width, height):
    
    xx, yy = shapely_polygon.exterior.coords.xy

    scaled_polygon = [scale_point(x, y, xmin, ymin, xmax, ymax, width, height) for x, y in zip(xx, yy)]
    
    return scaled_polygon




logger = format_logger(logger)
