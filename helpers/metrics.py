import geopandas as gpd
import pandas as pd



def get_fractional_sets(dets_gdf, labels_gdf, iou_threshold=0.25, area_threshold=None):
    """
    Find the intersecting detections and labels.
    Control their IoU and class to get the TP.
    Tag detections and labels not intersecting or not intersecting enough as FP and FN respectively.
    Save the intersections with mismatched class ids in a separate geodataframe.

    Args:
        dets_gdf (geodataframe): geodataframe of the detections.
        labels_gdf (geodataframe): geodataframe of the labels.
        iou_threshold (float): threshold to apply on the IoU to determine if detections and labels can be matched. Defaults to 0.25.
        area_threshold (float): threshold applied on clipped label and detection polygons to discard the smallest ones. Default None
    Raises:
        Exception: CRS mismatch

    Returns:
        dict:
        - geodataframe: true positive intersections between a detection and a label;
        - geodataframe: false postive detections;
        - geodataframe: false negative labels;
        - geodataframe: intersections between a detection and a label with a mismatched class id.
        - geodataframe: label and detection polygons with an area smaller than the threshold.
        """

    _dets_gdf = dets_gdf.reset_index(drop=True)
    _labels_gdf = labels_gdf.reset_index(drop=True)

    small_poly_gdf = gpd.GeoDataFrame() 
       
    if len(_labels_gdf) == 0:
        columns_list = ['area', 'geometry', 'dataset', 'label_id', 'label_class']
        fp_gdf = _dets_gdf.copy()
        tp_gdf = gpd.GeoDataFrame(columns=columns_list + ['det_id', 'det_class', 'score', 'IOU'])
        fn_gdf = gpd.GeoDataFrame(columns=columns_list)
        mismatched_classes_gdf = gpd.GeoDataFrame(columns=columns_list + ['det_id', 'det_class', 'score', 'IOU'])
        return {'tp_gdf': tp_gdf, 'fp_gdf': fp_gdf, 'fn_gdf': fn_gdf, 'mismatched_classes_gdf': mismatched_classes_gdf, 'small_poly_gdf': small_poly_gdf}
    
    assert(_dets_gdf.crs == _labels_gdf.crs), f"CRS Mismatch: detections' CRS = {_dets_gdf.crs}, labels' CRS = {_labels_gdf.crs}"

    # we add a id column to the labels dataset, which should not exist in detections too;
    # this allows us to distinguish matching from non-matching detections
    _labels_gdf['label_id'] = _labels_gdf.index
    _dets_gdf['det_id'] = _dets_gdf.index
    # We need to keep both geometries after sjoin to check the best intersection over union
    _labels_gdf['label_geom'] = _labels_gdf.geometry
    
    # Filter detections and labels with area less than thd value 
    if area_threshold:
        _dets_gdf['area'] = _dets_gdf.area
        filter_dets_gdf = _dets_gdf[_dets_gdf['area']<area_threshold]
        _dets_gdf = _dets_gdf[_dets_gdf['area']>=area_threshold]
        _labels_gdf['area'] = _labels_gdf.area
        filter_labels_gdf = _labels_gdf[_labels_gdf['area']<area_threshold]
        _labels_gdf = _labels_gdf[_labels_gdf['area']>=area_threshold]
        small_poly_gdf = pd.concat([filter_dets_gdf, filter_labels_gdf])

    # TRUE POSITIVES
    left_join = gpd.sjoin(_dets_gdf, _labels_gdf, how='left', predicate='intersects', lsuffix='dets', rsuffix='labels')

    # Test that something is detected
    candidates_tp_gdf = left_join[left_join.label_id.notnull()].copy()

    # IoU computation between labels and detections
    geom1 = candidates_tp_gdf['geometry'].to_numpy().tolist()
    geom2 = candidates_tp_gdf['label_geom'].to_numpy().tolist()    
    candidates_tp_gdf.loc[:, ['IOU']] = [intersection_over_union(i, ii) for (i, ii) in zip(geom1, geom2)]
  
    # Filter detections based on IoU value
    best_matches_gdf = candidates_tp_gdf.groupby(['det_id'], group_keys=False).apply(lambda g:g[g.IOU==g.IOU.max()])
    best_matches_gdf.drop_duplicates(subset=['det_id'], inplace=True) # Case two IoU are equal on the same det, keep the first one.

    # Detection, resp labels, with IOU lower than threshold value are considered as FP, resp FN, and saved as such
    actual_matches_gdf = best_matches_gdf[best_matches_gdf['IOU'] >= iou_threshold].copy()
    actual_matches_gdf = actual_matches_gdf.sort_values(by=['IOU'], ascending=False).drop_duplicates(subset=['label_id', 'tile_id'])
    actual_matches_gdf['IOU'] = actual_matches_gdf.IOU.round(3)

    matched_det_ids = actual_matches_gdf['det_id'].unique().tolist()
    matched_label_ids = actual_matches_gdf['label_id'].unique().tolist()
    fp_gdf_temp = candidates_tp_gdf[~candidates_tp_gdf.det_id.isin(matched_det_ids)].drop_duplicates(subset=['det_id'], ignore_index=True)
    fn_gdf_temp = candidates_tp_gdf[~candidates_tp_gdf.label_id.isin(matched_label_ids)].drop_duplicates(subset=['label_id'], ignore_index=True)
    fn_gdf_temp.loc[:, 'geometry'] = fn_gdf_temp.label_geom

    # Test that labels and detections share the same class (id starting at 1 for labels and at 0 for detections)
    condition = actual_matches_gdf.label_class == actual_matches_gdf.det_class + 1
    tp_gdf = actual_matches_gdf[condition].reset_index(drop=True)

    mismatched_classes_gdf = actual_matches_gdf[~condition].reset_index(drop=True)
    mismatched_classes_gdf.drop(columns=['x', 'y', 'z', 'dataset_labels', 'label_geom'], errors='ignore', inplace=True)
    mismatched_classes_gdf.rename(columns={'dataset_dets': 'dataset'}, inplace=True)
  
    # FALSE POSITIVES
    fp_gdf = left_join[left_join.label_id.isna()]
    assert(len(fp_gdf[fp_gdf.duplicated()]) == 0)
    fp_gdf = pd.concat([fp_gdf_temp, fp_gdf], ignore_index=True)
    fp_gdf.drop(
        columns=_labels_gdf.drop(columns='geometry').columns.to_list() + ['index_labels', 'dataset_labels', 'label_geom', 'IOU'], 
        errors='ignore', 
        inplace=True
    )
    fp_gdf.rename(columns={'dataset_dets': 'dataset'}, inplace=True)

    # FALSE NEGATIVES
    right_join = gpd.sjoin(_dets_gdf, _labels_gdf, how='right', predicate='intersects', lsuffix='dets', rsuffix='labels')
    fn_gdf = right_join[right_join.score.isna()].copy()
    fn_gdf.drop_duplicates(subset=['label_id', 'tile_id'], inplace=True)
    fn_gdf = pd.concat([fn_gdf_temp, fn_gdf], ignore_index=True)
    fn_gdf.drop(
        columns=_dets_gdf.drop(columns='geometry').columns.to_list() + ['dataset_dets', 'index_labels', 'x', 'y', 'z', 'label_geom', 'IOU', 'index_dets'], 
        errors='ignore', 
        inplace=True
    )
    fn_gdf.rename(columns={'dataset_dets': 'dataset'}, inplace=True)
 

    return {'tp_gdf': tp_gdf, 'fp_gdf': fp_gdf, 'fn_gdf': fn_gdf, 'mismatched_classes_gdf': mismatched_classes_gdf, 'small_poly_gdf': small_poly_gdf}


def get_metrics(tp_gdf, fp_gdf, fn_gdf, mismatched_classes_gdf, id_classes=0, method='macro-average'):
    """Determine the metrics based on the TP, FP and FN

    Args:
        tp_gdf (geodataframe): true positive detections
        fp_gdf (geodataframe): false positive detections
        fn_gdf (geodataframe): false negative labels
        mismatched_classes_gdf (geodataframe): labels and detections intersecting with a mismatched class id
        id_classes (list): list of the possible class ids. Defaults to 0.
        method (str): method used to compute multi-class metrics. Default to macro-average
    
    Returns:
        tuple: 
            - dict: TP count for each class
            - dict: FP count for each class
            - dict: FN count for each class
            - dict: precision for each class
            - dict: recall for each class
            - dict: f1-score for each class
            - float: accuracy
            - float: precision;
            - float: recall;
            - float: f1 score.
    """

    by_class_dict = {key: 0 for key in id_classes}
    tp_k = by_class_dict.copy()
    fp_k = by_class_dict.copy()
    fn_k = by_class_dict.copy()
    p_k = by_class_dict.copy()
    r_k = by_class_dict.copy()
    count_k = by_class_dict.copy()
    pw_k = by_class_dict.copy()
    rw_k = by_class_dict.copy()

    total_labels = len(tp_gdf) + len(fn_gdf) + len(mismatched_classes_gdf)

    for id_cl in id_classes:

        pure_fp_count = len(fp_gdf[fp_gdf.det_class==id_cl])
        pure_fn_count = len(fn_gdf[fn_gdf.label_class==id_cl+1])  # label class starting at 1 and id class at 0

        mismatched_fp_count = len(mismatched_classes_gdf[mismatched_classes_gdf.det_class==id_cl])
        mismatched_fn_count = len(mismatched_classes_gdf[mismatched_classes_gdf.label_class==id_cl+1])

        fp_count = pure_fp_count + mismatched_fp_count
        fn_count = pure_fn_count + mismatched_fn_count
        tp_count = len(tp_gdf[tp_gdf.det_class==id_cl])

        fp_k[id_cl] = fp_count
        fn_k[id_cl] = fn_count
        tp_k[id_cl] = tp_count

        count_k[id_cl] = tp_count + fn_count
        if tp_count > 0:
            p_k[id_cl] = tp_count / (tp_count + fp_count)
            r_k[id_cl] = tp_count / (tp_count + fn_count)

        if (method == 'macro-weighted-average') & (total_labels > 0):
            pw_k[id_cl] = (count_k[id_cl] / total_labels) * p_k[id_cl]
            rw_k[id_cl] = (count_k[id_cl] / total_labels) * r_k[id_cl] 

    if method == 'macro-average':   
        precision = sum(p_k.values()) / len(id_classes)
        recall = sum(r_k.values()) / len(id_classes)
    elif method == 'macro-weighted-average':  
        precision = sum(pw_k.values()) / len(id_classes)
        recall = sum(rw_k.values()) / len(id_classes)
    elif method == 'micro-average':  
        if sum(tp_k.values()) == 0:
            precision = 0
            recall = 0
        else:
            precision = sum(tp_k.values()) / (sum(tp_k.values()) + sum(fp_k.values()))
            recall = sum(tp_k.values()) / (sum(tp_k.values()) + sum(fn_k.values()))

    if precision==0 and recall==0:
        return tp_k, fp_k, fn_k, p_k, r_k, 0, 0, 0
    
    f1 = 2 * precision * recall / (precision + recall)
    
    return tp_k, fp_k, fn_k, p_k, r_k, precision, recall, f1


def intersection_over_union(polygon1_shape, polygon2_shape):
    """Determine the intersection area over union area (IOU) of two polygons

    Args:
        polygon1_shape (geometry): first polygon
        polygon2_shape (geometry): second polygon

    Returns:
        int: Unrounded ratio between the intersection and union area
    """

    # Calculate intersection and union, and the IOU
    polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
    polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection

    if polygon_union != 0:
        iou = polygon_intersection / polygon_union
    else:
        iou = 0

    return iou