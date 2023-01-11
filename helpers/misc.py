#!/bin/python
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os, sys
import gdal
import pandas as pd
import geopandas as gpd
import numpy as np

from cv2 import imwrite

from shapely.affinity import affine_transform, scale
from shapely.geometry import box
from rasterio import rasterio, features
from rasterio.transform import from_bounds

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

# Geometrical operations -------------------------------------

def scale_point(x, y, xmin, ymin, xmax, ymax, width, height):

    return (x-xmin)/(xmax-xmin)*(width), (ymax-y)/(ymax-ymin)*(height)


def scale_polygon(shapely_polygon, xmin, ymin, xmax, ymax, width, height):
    
    xx, yy = shapely_polygon.exterior.coords.xy

    # TODO: vectorize!
    scaled_polygon = [scale_point(x, y, xmin, ymin, xmax, ymax, width, height) for x, y in zip(xx, yy)]
    
    return scaled_polygon

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
        tp_gdf = gpd.GeoDataFrame()
        fn_gdf = gpd.GeoDataFrame()
        non_diag_gdf=gpd.GeoDataFrame()
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

# Assessing predictions --------------------------------------------

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
        predictions_dict = {"<image_filename>': [<prediction>]}
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
                            'properties': {'raster_val': v, 'pred_class':pred['pred_class'], 'score': pred['score'], 'crs': crs}, 
                            'geometry': s
                    } for (s, v) in features.shapes(pred_mask_int, mask=None, transform=transform)
                ]

            else:

                geom = affine_transform(box(*pred['pred_box']), [transform.a, transform.b, transform.d, transform.e, transform.xoff, transform.yoff])
                feats += [{'type': 'Feature', 
                            'properties': {'raster_val': 1.0, 'pred_class':pred['pred_class'], 'score': pred['score'], 'crs': crs}, 
                            'geometry': geom}]

    return feats

def visualize_predictions(dataset, predictor, input_format='RGB',
                        WORKING_DIR='object_detector', SAMPLE_TAGGED_IMG_SUBDIR='sample_tagged_images'):
    '''Use the predictor to do inferences on the dataset and tag some images with them.
    
    - dataset: registered COCO dataset
    - predictor: detectron2 preditor
    - input_format: input format as configured in the config file
    - WORKING_DIR: working directory
    - SAMPLE_TAGGED_IMG_SUBDIR: subdirectory where to store the tagged images
    
    return: a list of the written files
    '''

    written_files=[]
    for d in DatasetCatalog.get(dataset)[0:min(len(DatasetCatalog.get(dataset)), 10)]:
        output_filename = f'{dataset}_pred_{d["file_name"].split("/")[-1]}'
        output_filename = output_filename.replace('tif', 'png')

        ds = gdal.Open(d["file_name"])
        im_cwh = ds.ReadAsArray()
        im = np.transpose(im_cwh, (1, 2, 0))
        outputs = predictor(im)
        if input_format=='BGR':
            im_rgb=im[:, :, ::-1]
        elif input_format=='RGB':
            im_rgb=im
        elif input_format.startswith('BGR'):
            im=im[:,:,0:3]
            im_rgb=im[:, :, ::-1]
        elif input_format.startswith('RGB'):
            im_rgb=im[:,:,0:3]
        else:
            sys.exit(1)

        v = Visualizer(im_rgb, 
                    metadata=MetadataCatalog.get(dataset), 
                    scale=1.0, 
                    instance_mode=ColorMode.IMAGE_BW # remove the colors of unsegmented pixels
        )   
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        imwrite(os.path.join(SAMPLE_TAGGED_IMG_SUBDIR, output_filename),
                v.get_image()[:, :, ::-1]) # [:, :, ::-1] is for RGB -> BGR conversion, cf. https://stackoverflow.com/questions/14556545/why-opencv-using-bgr-colour-space-instead-of-rgb
        written_files.append( os.path.join(WORKING_DIR, os.path.join(SAMPLE_TAGGED_IMG_SUBDIR, output_filename)) )

    return written_files


# Path manipulations ----------------------------------------

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

# Raster manipulations ----------------------------

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

# Miscellanous ----------------------------------------------------

def my_unpack(list_of_tuples):
    # cf. https://www.geeksforgeeks.org/python-convert-list-of-tuples-into-list/
    
    return [item for t in list_of_tuples for item in t]