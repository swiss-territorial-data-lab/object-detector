#!/bin/python
# -*- coding: utf-8 -*-

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import geopandas as gpd
import pandas as pd

from tqdm import tqdm

# the following lines allow us to import modules from within this file's parent folder
from inspect import getsourcefile
current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)

from helpers import misc
from helpers.constants import DONE_MSG

from loguru import logger
logger = misc.format_logger(logger)


def split_additional_tiles(tiles_gdf, gt_tiles_gdf, trn_tiles_ids, val_tiles_ids, tst_tiles_ids, tile_type, frac_trn, seed):
        _tiles_gdf = tiles_gdf.copy()
        _gt_tiles_gdf = gt_tiles_gdf.copy()

        logger.info(f'Add {int(frac_trn * 100)}% of {tile_type} tiles to the trn, val and tst datasets')
        trn_fp_tiles_ids, val_fp_tiles_ids, tst_fp_tiles_ids = split_dataset(_tiles_gdf, frac_trn=frac_trn, seed=seed)

        # Add the FP tiles to the GT gdf 
        trn_tiles_ids.extend(trn_fp_tiles_ids)
        val_tiles_ids.extend(val_fp_tiles_ids)
        tst_tiles_ids.extend(tst_fp_tiles_ids)

        _gt_tiles_gdf = pd.concat([_gt_tiles_gdf, _tiles_gdf])

        return trn_tiles_ids, val_tiles_ids, tst_tiles_ids, _gt_tiles_gdf


def split_dataset(tiles_df, frac_trn=0.7, frac_left_val=0.5, seed=1):
    """Split the dataframe in the traning, validation and test set.

    Args:
        tiles_df (DataFrame): Dataset of the tiles
        frac_trn (float, optional): Fraction of the dataset to put in the training set. Defaults to 0.7.
        frac_left_val (float, optional): Fration of the leftover dataset to be in the validation set. Defaults to 0.5.
        seed (int, optional): random seed. Defaults to 1.

    Returns:
        tuple: 
            - list: tile ids going to the training set
            - list: tile ids going to the validation set
            - list: tile ids going to the test set
    """

    trn_tiles_ids = tiles_df\
        .sample(frac=frac_trn, random_state=seed)\
        .id.astype(str).to_numpy().tolist()

    val_tiles_ids = tiles_df[~tiles_df.id.astype(str).isin(trn_tiles_ids)]\
        .sample(frac=frac_left_val, random_state=seed)\
        .id.astype(str).to_numpy().tolist()

    tst_tiles_ids = tiles_df[~tiles_df.id.astype(str).isin(trn_tiles_ids + val_tiles_ids)]\
        .id.astype(str).to_numpy().tolist()
    
    return trn_tiles_ids, val_tiles_ids, tst_tiles_ids


def split_tiles(aoi_tiles_gdf, gt_labels_gdf, oth_labels_gdf, fp_labels_gdf, fp_frac_trn, empty_tiles_dict, id_list_ept_tiles, img_metadata_dict, tile_size, seed, 
                output_dir, debug_mode):

    written_files = []

    if gt_labels_gdf.empty:
        split_aoi_tiles_gdf = aoi_tiles_gdf.copy()
        split_aoi_tiles_gdf['dataset'] = 'oth'
    else:
        assert(aoi_tiles_gdf.crs == gt_labels_gdf.crs ), "CRS Mismatch between AoI tiles and labels."

        gt_tiles_gdf = gpd.sjoin(aoi_tiles_gdf, gt_labels_gdf, how='inner', predicate='intersects')
    
        # get the number of labels per class
        labels_per_class_dict = {}
        for category in gt_tiles_gdf.CATEGORY.unique():
            labels_per_class_dict[category] = gt_tiles_gdf[gt_tiles_gdf.CATEGORY == category].shape[0]
        # Get the number of labels per tile
        labels_per_tiles_gdf = gt_tiles_gdf.groupby(['id', 'CATEGORY'], as_index=False).size()

        gt_tiles_gdf.drop_duplicates(subset=aoi_tiles_gdf.columns, inplace=True)
        gt_tiles_gdf = gt_tiles_gdf[aoi_tiles_gdf.columns]

        # Get the tiles containing at least one "FP" label but no "GT" label (if applicable)
        if fp_labels_gdf.empty:
            fp_tiles_gdf = gpd.GeoDataFrame(columns=['id'])
        else:
            tmp_fp_tiles_gdf, _ = misc.intersect_labels_with_aoi(aoi_tiles_gdf, fp_labels_gdf)
            fp_tiles_gdf = tmp_fp_tiles_gdf[~tmp_fp_tiles_gdf.id.astype(str).isin(gt_tiles_gdf.id.astype(str))].copy()
            del tmp_fp_tiles_gdf            

        # remove tiles including at least one "oth" label (if applicable)
        if not oth_labels_gdf.empty:
            oth_tiles_to_remove_gdf, _ = misc.intersect_labels_with_aoi(gt_tiles_gdf, oth_labels_gdf)
            gt_tiles_gdf = gt_tiles_gdf[~gt_tiles_gdf.id.astype(str).isin(oth_tiles_to_remove_gdf.id.astype(str))].copy()
            del oth_tiles_to_remove_gdf

        # add ramdom tiles not intersecting labels to the dataset 
        oth_tiles_gdf = aoi_tiles_gdf[~aoi_tiles_gdf.id.astype(str).isin(gt_tiles_gdf.id.astype(str))].copy()
        oth_tiles_gdf = oth_tiles_gdf[~oth_tiles_gdf.id.astype(str).isin(fp_tiles_gdf.id.astype(str))].copy()

        # OTH tiles = AoI tiles with labels, but which are not GT
        if empty_tiles_dict:           
            empty_tiles_gdf = aoi_tiles_gdf[aoi_tiles_gdf.id.astype(str).isin(id_list_ept_tiles)].copy()

            if debug_mode:
                assert(len(empty_tiles_gdf != 0)), "No empty tiles could be added. Increase the number of tiles sampled in debug mode"
            
            oth_tiles_gdf = oth_tiles_gdf[~oth_tiles_gdf.id.astype(str).isin(empty_tiles_gdf.id.astype(str))].copy()
            oth_tiles_gdf['dataset'] = 'oth'
            assert( len(aoi_tiles_gdf) == len(gt_tiles_gdf) + len(fp_tiles_gdf) + len(empty_tiles_gdf) + len(oth_tiles_gdf) )
        else: 
            oth_tiles_gdf['dataset'] = 'oth'
            assert( len(aoi_tiles_gdf) == len(gt_tiles_gdf) + len(fp_tiles_gdf) + len(oth_tiles_gdf) )
        
        # 70%, 15%, 15% split
        categories_arr = labels_per_tiles_gdf.CATEGORY.unique()
        categories_arr.sort()
        if not seed:
            max_seed = 50
            best_split = 0
            for test_seed in tqdm(range(max_seed), desc='Test seeds for splitting tiles between datasets'):
                ok_split = 0
                trn_tiles_ids, val_tiles_ids, tst_tiles_ids = split_dataset(gt_tiles_gdf, seed=test_seed)
                
                for category in categories_arr:
                    
                    ratio_trn = labels_per_tiles_gdf.loc[
                        (labels_per_tiles_gdf.CATEGORY == category) & labels_per_tiles_gdf.id.astype(str).isin(trn_tiles_ids), 'size'
                    ].sum() / labels_per_class_dict[category]
                    ratio_val = labels_per_tiles_gdf.loc[
                        (labels_per_tiles_gdf.CATEGORY == category) & labels_per_tiles_gdf.id.astype(str).isin(val_tiles_ids), 'size'
                    ].sum() / labels_per_class_dict[category]
                    ratio_tst = labels_per_tiles_gdf.loc[
                        (labels_per_tiles_gdf.CATEGORY == category) & labels_per_tiles_gdf.id.astype(str).isin(tst_tiles_ids), 'size'
                    ].sum() / labels_per_class_dict[category]

                    ok_split = ok_split + 1 if ratio_trn >= 0.60 else ok_split
                    ok_split = ok_split + 1 if ratio_val >= 0.12 else ok_split
                    ok_split = ok_split + 1 if ratio_tst >= 0.12 else ok_split

                    ok_split = ok_split - 1 if 0 in [ratio_trn, ratio_val, ratio_tst] else ok_split
                
                if ok_split == len(categories_arr)*3:
                    logger.info(f'A seed of {test_seed} produces a good repartition of the labels.')
                    seed = test_seed
                    break
                elif ok_split > best_split:
                    seed = test_seed
                    best_split = ok_split
                
                if test_seed == max_seed-1:
                    logger.warning(f'No satisfying seed found between 0 and {max_seed}.')
                    logger.info(f'The best seed was {seed} with ~{best_split} class subsets containing the correct proportion (trn~0.7, val~0.15, tst~0.15).')
                    logger.info('The user should set a seed manually if not satisfied.')

        else:
            trn_tiles_ids, val_tiles_ids, tst_tiles_ids = split_dataset(gt_tiles_gdf, seed=seed)
        
        if not fp_tiles_gdf.empty:
            trn_tiles_ids, val_tiles_ids, tst_tiles_ids, gt_tiles_gdf = split_additional_tiles(
                fp_tiles_gdf, gt_tiles_gdf, trn_tiles_ids, val_tiles_ids, tst_tiles_ids, 'FP', fp_frac_trn, seed
            )
            del fp_tiles_gdf
        if empty_tiles_dict:
            EPT_FRAC_TRN = empty_tiles_dict['frac_trn'] if 'frac_trn' in empty_tiles_dict.keys() else 0.7
            trn_tiles_ids, val_tiles_ids, tst_tiles_ids, gt_tiles_gdf = split_additional_tiles(
                empty_tiles_gdf, gt_tiles_gdf, trn_tiles_ids, val_tiles_ids, tst_tiles_ids, 'empty', EPT_FRAC_TRN, seed
            )
            del empty_tiles_gdf

        for df in [gt_tiles_gdf, labels_per_tiles_gdf]:
            df.loc[df.id.astype(str).isin(trn_tiles_ids), 'dataset'] = 'trn'
            df.loc[df.id.astype(str).isin(val_tiles_ids), 'dataset'] = 'val'
            df.loc[df.id.astype(str).isin(tst_tiles_ids), 'dataset'] = 'tst'

        logger.info('Repartition in the datasets by category:')
        for dst in ['trn', 'val', 'tst']:
            for category in categories_arr:
                row_ids = labels_per_tiles_gdf.index[(labels_per_tiles_gdf.dataset==dst) & (labels_per_tiles_gdf.CATEGORY==category)]
                logger.info(f'   {category} labels in {dst} dataset: {labels_per_tiles_gdf.loc[labels_per_tiles_gdf.index.isin(row_ids), "size"].sum()}')

        # remove columns generated by the Spatial Join
        gt_tiles_gdf = gt_tiles_gdf[aoi_tiles_gdf.columns.tolist() + ['dataset']].copy()

        assert( len(gt_tiles_gdf) == len(trn_tiles_ids) + len(val_tiles_ids) + len(tst_tiles_ids) ), \
            'Tiles were lost in the split between training, validation and test sets.'
        
        split_aoi_tiles_gdf = pd.concat(
            [
                gt_tiles_gdf,
                oth_tiles_gdf
            ]
        )

        # let's free up some memory
        del gt_tiles_gdf
        del oth_tiles_gdf
        
        
    assert( len(split_aoi_tiles_gdf) == len(aoi_tiles_gdf) ) # it means that all the tiles were actually used
    
    SPLIT_AOI_TILES = os.path.join(output_dir, 'split_aoi_tiles.geojson')

    split_aoi_tiles_gdf.to_file(SPLIT_AOI_TILES, driver='GeoJSON')
    written_files.append(SPLIT_AOI_TILES)
    logger.success(f'{DONE_MSG} A file was written {SPLIT_AOI_TILES}')

    img_md_df = pd.DataFrame.from_dict(img_metadata_dict, orient='index')
    img_md_df.reset_index(inplace=True)
    img_md_df.rename(columns={"index": "img_file"}, inplace=True)

    img_md_df['id'] = img_md_df.apply(misc.img_md_record_to_tile_id, axis=1)

    split_aoi_tiles_with_img_md_gdf = split_aoi_tiles_gdf.merge(img_md_df, on='id', how='left')
    for dst in split_aoi_tiles_with_img_md_gdf.dataset.to_numpy():
        os.makedirs(os.path.join(output_dir, f'{dst}-images{f"-{tile_size}" if tile_size else ""}'), exist_ok=True)

    split_aoi_tiles_with_img_md_gdf['dst_file'] = [
        src_file.replace('all', dataset) 
        for src_file, dataset in zip(split_aoi_tiles_with_img_md_gdf.img_file, split_aoi_tiles_with_img_md_gdf.dataset)
    ]
    for src_file, dst_file in zip(split_aoi_tiles_with_img_md_gdf.img_file, split_aoi_tiles_with_img_md_gdf.dst_file):
        misc.make_hard_link(src_file, dst_file)

    return split_aoi_tiles_with_img_md_gdf, written_files