import geopandas as gpd



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