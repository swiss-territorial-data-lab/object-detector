import sys
import os

import geopandas as gpd
import pandas as pd

import numpy as np

def test_crs(crs1, crs2 = "EPSG:2056"):
    '''
    Take the crs of two dataframes and compare them. If they are not the same, stop the script.
    '''
    if isinstance(crs1, gpd.GeoDataFrame):
        crs1=crs1.crs
    if isinstance(crs2, gpd.GeoDataFrame):
        crs2=crs2.crs

    try:
        assert(crs1 == crs2), f"CRS mismatch between the two files ({crs1} vs {crs2})."
    except Exception as e:
        print(e)
        sys.exit(1)

def ensure_dir_exists(dirpath):
    '''
    Test if a directory exists. If not, make it.

    return: the path to the verified directory.
    '''

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
        print(f"The directory {dirpath} was created.")

    return dirpath



def polygons_diff_without_artifacts(polygons, p1_idx, p2_idx, keep_everything=False):
    '''
    Make the difference of the geometry at row p2_idx with the one at the row p1_idx
    
    - polygons: dataframe of polygons
    - p1_idx: index of the "obstacle" polygon in the dataset
    - p2_idx: index of the final polygon
    - keep_everything: boolean indicating if we should keep large parts that would be eliminated otherwise

    return: a dataframe of the polygons where the part of p1_idx overlapping with p2_idx has been erased. The parts of
    multipolygons can be all kept or just the largest one (longer process).
    '''
    
    # Store intermediary results back to poly
    diff=polygons.loc[p2_idx,'geometry']-polygons.loc[p1_idx,'geometry']

    if diff.geom_type == 'Polygon':
        polygons.loc[p2_idx,'geometry'] -= polygons.loc[p1_idx,'geometry']

    elif diff.geom_type == 'MultiPolygon':
        # if a multipolygone is created, only keep the largest part to avoid the following error: https://github.com/geopandas/geopandas/issues/992
        polygons.loc[p2_idx,'geometry'] = max((polygons.loc[p2_idx,'geometry']-polygons.loc[p1_idx,'geometry']).geoms, key=lambda a: a.area)

        # The threshold to which we consider that subparts are still important is hard-coded at 10 units.
        limit=10
        parts_geom=[poly for poly in diff.geoms if poly.area>limit]
        if len(parts_geom)>1 and keep_everything:
            parts_area=[poly.area for poly in diff.geoms if poly.area>limit]
            parts=pd.DataFrame({'geometry':parts_geom,'area':parts_area})
            parts.sort_values(by='area', ascending=False, inplace=True)
            
            new_row_serie=polygons.loc[p2_idx].copy()
            new_row_dict={'OBJECTID': [], 'OBJEKTART': [], 'KUNSTBAUTE': [], 'BELAGSART': [], 'geometry': [], 
                        'GDB-Code': [], 'Width': [], 'saved_geom': []}
            new_poly=0
            for elem_geom in parts['geometry'].values[1:]:
                
                new_row_dict['OBJECTID'].append(int(str(int(new_row_serie.OBJECTID))+str(new_poly)))
                new_row_dict['geometry'].append(elem_geom)
                new_row_dict['OBJEKTART'].append(new_row_serie.OBJEKTART)
                new_row_dict['KUNSTBAUTE'].append(new_row_serie.KUNSTBAUTE)
                new_row_dict['BELAGSART'].append(new_row_serie.BELAGSART)
                new_row_dict['GDB-Code'].append(new_row_serie['GDB-Code'])
                new_row_dict['Width'].append(new_row_serie.Width)
                new_row_dict['saved_geom'].append(new_row_serie.saved_geom)

                new_poly+=1

            polygons=pd.concat([polygons, pd.DataFrame(new_row_dict)], ignore_index=True)

    return polygons


def test_valid_geom(poly_gdf, correct=False, gdf_obj_name=None):
    '''
    Test if all the geometry of a dataset are valid. When it is not the case, correct the geometries with a buffer of 0 m
    if correct != False and stop with an error otherwise.

    - poly_gdf: dataframe of geometries to check
    - correct: boolean indicating if the invalid geometries should be corrected with a buffer of 0 m
    - gdf_boj_name: name of the dataframe of the object in it to print with the error message

    return: a dataframe with only valid geometries.
    '''

    try:
        assert(poly_gdf[poly_gdf.is_valid==False].shape[0]==0), \
            f"{poly_gdf[poly_gdf.is_valid==False].shape[0]} geometries are invalid {f' among the {gdf_obj_name}' if gdf_obj_name else ''}."
    except Exception as e:
        print(e)
        if correct:
            print("Correction of the invalid geometries with a buffer of 0 m...")
            corrected_poly=poly_gdf.copy()
            corrected_poly.loc[corrected_poly.is_valid==False,'geometry']= \
                            corrected_poly[corrected_poly.is_valid==False]['geometry'].buffer(0)

            return corrected_poly
        else:
            sys.exit(1)

    print(f"There aren't any invalid geometries{f' among the {gdf_obj_name}' if gdf_obj_name else ''}.")

    return poly_gdf
