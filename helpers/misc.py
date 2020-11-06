#!/bin/python
# -*- coding: utf-8 -*-
import geopandas as gpd

from shapely.affinity import affine_transform, scale
from shapely.geometry import box
from rasterio import rasterio, features

from helpers.MIL import image_metadata_to_affine_transform

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

        crs = f"EPSG:{this_img_metadata['extent']['spatialReference']['latestWkid']}"
        transform = image_metadata_to_affine_transform(this_img_metadata)
        
        for pred in v:

            if 'pred_mask' in pred.keys():

                pred_mask_int = pred['pred_mask'].astype(int)

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


def clip_labels(the_labels_gdf, the_tile, fact=0.990):
    
    # the following prevents labels crossing the tile borders to be mis-classified
    scaled_tile_geometry = scale(the_tile.geometry, xfact=fact, yfact=fact)
  
    tmp_gdf = gpd.clip(the_labels_gdf, scaled_tile_geometry, keep_geom_type=True)
   
    if len(tmp_gdf[tmp_gdf.geometry.notnull()]) == 0:
        return gpd.GeoDataFrame()
    
    clipped_labels_gdf = tmp_gdf.explode()
    clipped_labels_gdf['dataset'] = the_tile.dataset
    clipped_labels_gdf['tile_id'] = the_tile.id
    
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
        tp_gdf = pd.DataFrame()
        fn_gdf = pd.DataFrame()       
        return tp_gdf, fp_gdf, fn_gdf
    
    assert(preds_gdf.crs == labels_gdf.crs)

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

