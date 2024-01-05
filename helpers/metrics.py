import geopandas as gpd
import pandas as pd



def get_fractional_sets(dets_gdf, labels_gdf, iou_threshold=0.25):
    """Find the intersecting detections and labels.
    Control their class to get the TP.
    Labels non-intersection detections and labels as FP and FN respectively.
    Save the intersetions with mismatched class ids in a separate geodataframe.

    Args:
        dets_gdf (geodataframe): geodataframe of the detections.
        labels_gdf (geodataframe): geodataframe of the labels.
        iou_threshold (float): threshold to apply on the IoU to determine if detections and labels can be matched. Defaults to 0.25.

    Raises:
        Exception: CRS mismatch

    Returns:
        tuple:
        - geodataframe: true positive intersections between a detection and a label;
        - geodataframe: false postive detection;
        - geodataframe: false negative labels;
        - geodataframe: intersections between a detection and a label with a mismatched class id.
    """

    _dets_gdf = dets_gdf.reset_index(drop=True)
    _labels_gdf = labels_gdf.reset_index(drop=True)
    
    if len(_labels_gdf) == 0:
        fp_gdf = _dets_gdf.copy()
        tp_gdf = gpd.GeoDataFrame()
        fn_gdf = gpd.GeoDataFrame()
        mismatched_class_gdf = gpd.GeoDataFrame()
        return tp_gdf, fp_gdf, fn_gdf, mismatched_class_gdf
    
    assert(_dets_gdf.crs == _labels_gdf.crs), f"CRS Mismatch: detections' CRS = {_dets_gdf.crs}, labels' CRS = {_labels_gdf.crs}"

    # we add a id column to the labels dataset, which should not exist in detections too;
    # this allows us to distinguish matching from non-matching detections
    _labels_gdf['label_id'] = _labels_gdf.index
    _dets_gdf['det_id'] = _dets_gdf.index
    # We need to keep both geometries after sjoin to check the best intersection
    _labels_gdf['label_geom'] = _labels_gdf.geometry
    
    # TRUE POSITIVES
    left_join = gpd.sjoin(_dets_gdf, _labels_gdf, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
    
    # Test that something is detected
    candidates_tp_gdf = left_join[left_join.label_id.notnull()].copy()

    # IoU computation between labels and detections
    geom1 = candidates_tp_gdf['geometry'].to_numpy().tolist()
    geom2 = candidates_tp_gdf['label_geom'].to_numpy().tolist()
    iou = []
    for (i, ii) in zip(geom1, geom2):
        iou.append(intersection_over_union(i, ii))
    candidates_tp_gdf['IOU'] = iou
    
    # Filter detections based on IoU value
    best_matches_gdf = candidates_tp_gdf.groupby(['det_id'], group_keys=False).apply(lambda g:g[g.IOU==g.IOU.max()])
    best_matches_gdf.drop_duplicates(subset=['det_id'], inplace=True) # <- this line could change the results depending and which is dropped 

    # Detection, resp labels, with IOU lower than threshold value are considered as FP, resp FN, and saved as such
    actual_matches_gdf = best_matches_gdf[best_matches_gdf['IOU'] >= iou_threshold].copy()
    actual_matches_gdf = actual_matches_gdf.sort_values(by=['IOU'], ascending=False).drop_duplicates(subset=['label_id', 'tile_id'])

    matched_det_ids = actual_matches_gdf['det_id'].unique().tolist()
    matched_label_ids = actual_matches_gdf['label_id'].unique().tolist()
    fp_gdf_temp = candidates_tp_gdf[~candidates_tp_gdf.det_id.isin(matched_det_ids)].drop_duplicates(subset=['det_id'], ignore_index=True)
    fn_gdf_temp = candidates_tp_gdf[~candidates_tp_gdf.label_id.isin(matched_label_ids)].drop_duplicates(subset=['label_id'], ignore_index=True)
    fn_gdf_temp.loc[:, 'geometry'] = fn_gdf_temp.label_geom

    actual_matches_gdf['IOU'] = actual_matches_gdf.IOU.round(3)

    # Test that it has the right class (id starting at 1 for labels and at 0 for detections)
    condition = actual_matches_gdf.label_class == actual_matches_gdf.det_class+1
    tp_gdf = actual_matches_gdf[condition].reset_index(drop=True)
    mismatched_class_gdf = actual_matches_gdf[~condition].reset_index(drop=True)
    mismatched_class_gdf.drop(columns=['x', 'y', 'z', 'dataset_right', 'label_geom'], errors='ignore', inplace=True)
    mismatched_class_gdf.rename(columns={'dataset_left': 'dataset'}, inplace=True)


    # FALSE POSITIVES
    fp_gdf = left_join[left_join.label_id.isna()].copy()
    assert(len(fp_gdf[fp_gdf.duplicated()]) == 0)
    fp_gdf = pd.concat([fp_gdf_temp, fp_gdf], ignore_index=True)
    fp_gdf.drop(columns=_labels_gdf.columns.to_list() + ['index_right', 'dataset_right', 'label_geom', 'IOU'], errors='ignore', inplace=True)
    fp_gdf.rename(columns={'dataset_left': 'dataset'}, inplace=True)
    
    # FALSE NEGATIVES
    right_join = gpd.sjoin(_dets_gdf, _labels_gdf, how='right', predicate='intersects', lsuffix='left', rsuffix='right')
    fn_gdf = right_join[right_join.score.isna()].copy()
    fn_gdf.drop_duplicates(subset=['label_id', 'tile_id'], inplace=True)
    fn_gdf = pd.concat([fn_gdf_temp, fn_gdf], ignore_index=True)
    fn_gdf.drop(columns=_dets_gdf.columns.to_list() + ['dataset_left', 'index_right', 'x', 'y', 'z', 'label_geom', 'IOU', 'index_left'], errors='ignore', inplace=True)
    fn_gdf.rename(columns={'dataset_right': 'dataset'}, inplace=True)
    
    return tp_gdf, fp_gdf, fn_gdf, mismatched_class_gdf


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
            FN = len(fn_gdf[fn_gdf.label_class==id_cl+1]) + len(mismatch_gdf[mismatch_gdf.label_class == id_cl+1])
    
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

    return polygon_intersection / polygon_union