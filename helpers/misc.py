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
