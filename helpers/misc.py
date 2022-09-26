#!/bin/python
# -*- coding: utf-8 -*-
import sys
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


# cf. https://gis.stackexchange.com/questions/187877/how-to-polygonize-raster-to-shapely-polygons
def predictions_to_features(predictions_dict, img_path):
    """
        predictions_dict = {"<image_filename>': [<prediction>]
        <prediction> = {'score': ..., 'pred_class': ..., 'pred_mask': ..., 'pred_box': ...}
    """

    feats = []

    for k, v in predictions_dict.items():
        # N.B.: src images are only used for georeferencing (src.crs, src.transform)
        with rasterio.open(os.path.join(img_path, k)) as src:

            for pred in v:

                pred_mask_int = pred['pred_mask'].astype(int)

                feats += [{'type': 'Feature', 
                            'properties': {'raster_val': v, 'score': pred['score'], 'crs': src.crs}, 
                            'geometry': s
                    } for (s, v) in features.shapes(pred_mask_int, mask=None, transform=src.transform)
                ]

    return feats


def fast_predictions_to_features(predictions_dict, img_metadata_dict):
    """
        predictions_dict = {"<image_filename>': [<prediction>]
        <prediction> = {'score': ..., 'pred_class': ..., 'pred_mask': ..., 'pred_box': ...}

        img_metadata_dict's values includes the metadata issued by ArcGIS Server; keys are equal to filenames
    """
    
    feats = []

    for k, v in predictions_dict.items():

        # k is like "images/val-images-256/18_135617_92947.tif"
        # img_metadata_dict keys are like "18_135617_92947.tif"

        kk = k.split('/')[-1]
        this_img_metadata = img_metadata_dict[kk]
        #print(this_img_metadata)
        
        crs = f"EPSG:{this_img_metadata['extent']['spatialReference']['latestWkid']}"
        transform = image_metadata_to_affine_transform(this_img_metadata)
        #print(transform)
        for pred in v:
            #print(pred)
            if 'pred_mask' in pred.keys():

                pred_mask_int = pred['pred_mask'].astype(np.uint8)
                feats += [{'type': 'Feature', 
                            'properties': {'raster_val': v, 'score': pred['score'], 'crs': crs}, 
                            'geometry': s
                    } for (s, v) in features.shapes(pred_mask_int, mask=None, transform=transform)
                ]

            else:

                geom = affine_transform(box(*pred['pred_box']), [transform.a, transform.b, transform.d, transform.e, transform.xoff, transform.yoff])
                feats += [{'type': 'Feature', 
                            'properties': {'raster_val': 1.0, 'score': pred['score'], 'crs': crs}, 
                            'geometry': geom}]

    return feats


def img_md_record_to_tile_id(img_md_record):
    
        filename = os.path.split(img_md_record.img_file)[-1]
        
        z_x_y = filename.split('.')[0]
        z, x, y = z_x_y.split('_')
        
        return f'({x}, {y}, {z})'


def create_hard_link(row):

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
    
    labels_tiles_sjoined_gdf = gpd.sjoin(labels_gdf, tiles_gdf, how='inner', op='intersects')
    
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


def get_metrics(tp_gdf, fp_gdf, fn_gdf):
    
    TP = len(tp_gdf)
    FP = len(fp_gdf)
    FN = len(fn_gdf)
    #print(TP, FP, FN)
    
    if TP == 0:
        return 0, 0, 0

    precision = TP / (TP + FP)
    recall    = TP / (TP + FN)
    f1 = 2*precision*recall/(precision+recall)
    
    return precision, recall, f1


def get_fractional_sets(the_preds_gdf, the_labels_gdf):

    preds_gdf = the_preds_gdf.copy()
    labels_gdf = the_labels_gdf.copy()
    
    if len(labels_gdf) == 0:
        fp_gdf = preds_gdf.copy()
        tp_gdf = gpd.GeoDataFrame()
        fn_gdf = gpd.GeoDataFrame()       
        return tp_gdf, fp_gdf, fn_gdf
    
    try:
        assert(preds_gdf.crs == labels_gdf.crs), f"CRS Mismatch: predictions' CRS = {preds_gdf.crs}, labels' CRS = {labels_gdf.crs}"
    except Exception as e:
        raise Exception(e)
        

    # we add a dummy column to the labels dataset, which should not exist in predictions too;
    # this allows us to distinguish matching from non-matching predictions
    labels_gdf['dummy_id'] = labels_gdf.index
    
    # TRUE POSITIVES
    left_join = gpd.sjoin(preds_gdf, labels_gdf, how='left', op='intersects', lsuffix='left', rsuffix='right')
    
    tp_gdf = left_join[left_join.dummy_id.notnull()].copy()
    tp_gdf.drop_duplicates(subset=['dummy_id', 'tile_id'], inplace=True)
    tp_gdf.drop(columns=['dummy_id'], inplace=True)
    
    # FALSE POSITIVES -> potentially "new" swimming pools
    fp_gdf = left_join[left_join.dummy_id.isna()].copy()
    assert(len(fp_gdf[fp_gdf.duplicated()]) == 0)
    fp_gdf.drop(columns=['dummy_id'], inplace=True)
    
    # FALSE NEGATIVES -> potentially, objects that are not actual swimming pools!
    right_join = gpd.sjoin(preds_gdf, labels_gdf, how='right', op='intersects', lsuffix='left', rsuffix='right')
    fn_gdf = right_join[right_join.score.isna()].copy()
    fn_gdf.drop_duplicates(subset=['dummy_id', 'tile_id'], inplace=True)
    
    return tp_gdf, fp_gdf, fn_gdf


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


def image_metadata_to_affine_transform(image_metadata):
    """
    This uses rasterio.
    """
    
    xmin = image_metadata['extent']['xmin']
    xmax = image_metadata['extent']['xmax']
    ymin = image_metadata['extent']['ymin']
    ymax = image_metadata['extent']['ymax']
    width  = image_metadata['width']
    height = image_metadata['height']
    
    affine = from_bounds(xmin, ymin, xmax, ymax, width, height)

    return affine
