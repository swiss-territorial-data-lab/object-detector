#!/bin/python
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os, sys
import pandas as pd
import geopandas as gpd
import numpy as np

from shapely.affinity import affine_transform, scale
from shapely.geometry import box
from rasterio import rasterio, features
from rasterio.transform import from_bounds


def scale_point(x, y, xmin, ymin, xmax, ymax, width, height):

    return (x-xmin)/(xmax-xmin)*(width), (ymax-y)/(ymax-ymin)*(height)


def scale_polygon(shapely_polygon, xmin, ymin, xmax, ymax, width, height):
    
    xx, yy = shapely_polygon.exterior.coords.xy

    # TODO: vectorize!
    scaled_polygon = [scale_point(x, y, xmin, ymin, xmax, ymax, width, height) for x, y in zip(xx, yy)]
    
    return scaled_polygon


def my_unpack(list_of_tuples):
    # cf. https://www.geeksforgeeks.org/python-convert-list-of-tuples-into-list/
    
    return [item for t in list_of_tuples for item in t]


def img_md_record_to_tile_id(img_md_record):
    
        filename = os.path.split(img_md_record.img_file)[-1]
        
        z_x_y = filename.split('.')[0]
        z, x, y = z_x_y.split('_')
        
        return f'({x}, {y}, {z})'


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


def get_metrics(tp_gdf, fp_gdf, fn_gdf, non_diag_gdf, id_classes):
    
    p_k={key: None for key in id_classes}
    r_k={key: None for key in id_classes}
    for id_cl in id_classes:
        TP = len(tp_gdf[tp_gdf['pred_class']==id_cl])
        FP = len(fp_gdf[fp_gdf['pred_class']==id_cl]) + len(non_diag_gdf[non_diag_gdf['pred_class']==id_cl])
        FN = len(fn_gdf[fn_gdf['contig_id']==id_cl]) + len(non_diag_gdf[non_diag_gdf['contig_id']==id_cl])
        #print(TP, FP, FN)

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


def get_fractional_sets(the_preds_gdf, the_labels_gdf):

    preds_gdf = the_preds_gdf.copy()
    labels_gdf = the_labels_gdf.copy()
    
    if len(labels_gdf) == 0:
        fp_gdf = preds_gdf.copy()
        tp_gdf = gpd.GeoDataFrame(columns=['pred_class', 'contig_id'])
        fn_gdf = gpd.GeoDataFrame(columns=['pred_class', 'contig_id'])
        non_diag_gdf=gpd.GeoDataFrame(columns=['pred_class', 'contig_id'])
        return tp_gdf, fp_gdf, fn_gdf, non_diag_gdf
    
    try:
        assert(preds_gdf.crs == labels_gdf.crs), f"CRS Mismatch: predictions' CRS = {preds_gdf.crs}, labels' CRS = {labels_gdf.crs}"
    except Exception as e:
        raise Exception(e)
        

    # we add a dummy column to the labels dataset, which should not exist in predictions too;
    # this allows us to distinguish matching from non-matching predictions
    labels_gdf['dummy_id'] = labels_gdf.index
    
    # TRUE POSITIVES -> detected something & it has the right ID
    left_join = gpd.sjoin(preds_gdf, labels_gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
    
    detections_w_label = left_join[left_join.dummy_id.notnull()].copy()    
    detections_w_label.drop_duplicates(subset=['dummy_id', 'tile_id'], inplace=True)
    detections_w_label.drop(columns=['dummy_id'], inplace=True)
    
    tp_gdf=detections_w_label[detections_w_label['contig_id']==detections_w_label['pred_class']]
    
    # Elements not on the diagonal -> detected somehting & it has the wrong ID
    non_diag_gdf = detections_w_label[detections_w_label['contig_id']!=detections_w_label['pred_class']]
    
    # FALSE POSITIVES -> detected something where there is nothing
    fp_gdf = left_join[left_join.dummy_id.isna()].copy()
    assert(len(fp_gdf[fp_gdf.duplicated()]) == 0)
    fp_gdf.drop(columns=['dummy_id'], inplace=True)
    
    # FALSE NEGATIVES -> detected nothing where there is something
    right_join = gpd.sjoin(preds_gdf, labels_gdf, how='right', predicate='intersects', lsuffix='left', rsuffix='right')
    fn_gdf = right_join[right_join.score.isna()].copy()
    fn_gdf.drop_duplicates(subset=['dummy_id', 'tile_id'], inplace=True)
    fn_gdf.drop(columns=['dummy_id'], inplace=True)
    
    return tp_gdf, fp_gdf, fn_gdf, non_diag_gdf


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


def reformat_xyz(row):
    """
    convert 'id' string to list of ints for z,x,y
    """
    x, y, z = row['id'].lstrip('(,)').rstrip('(,)').split(',')
    
    # check whether x, y, z are ints
    assert str(int(x)) == str(x).strip(' ')
    assert str(int(y)) == str(y).strip(' ')
    assert str(int(z)) == str(z).strip(' ')

    row['xyz'] = [int(x), int(y), int(z)]
    
    return row


def bounds_to_bbox(bounds):
    
    xmin = bounds[0]
    ymin = bounds[1]
    xmax = bounds[2]
    ymax = bounds[3]
    
    bbox = f"{xmin},{ymin},{xmax},{ymax}"
    
    return bbox


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