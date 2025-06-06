import os
import sys
from loguru import logger
from tqdm import tqdm

import geopandas as gpd
import pandas as pd
import rasterio as rio
from glob import glob
from rasterio.features import shapes
from shapely.geometry import box, Point, Polygon, shape

from math import ceil

from helpers.functions_for_examples import get_bbox_origin
from helpers.misc import format_logger


logger = format_logger(logger)


def control_overlap(gdf1, gdf2, threshold=0.5, op='larger'):
    """Test the overlap between the geometries of two GeoDataFrames and return the ids of the 1st gdf passing the test.

    Args:
        gdf1 (GeoDataFrame): first GeoDataFrame
        gdf2 (GeoDataFrame): second GeoDataFrame
        threshold (float, optional): limit value. Defaults to 0.5.
        op (str, optional): operator to use in the test. Possible values are 'larger' and "lte" (larger than or equal to). Defaults to 'larger'.

    Returns:
        list: ids of the 1st gdf passing the test
    """
    
    gdf1['total_area'] = gdf1.area

    intersection_gdf = gpd.overlay(gdf1, gdf2, how="difference", keep_geom_type=True)
    intersection_gdf = intersection_gdf.dissolve('id', as_index=False)
    intersection_gdf['percentage_area_left'] = intersection_gdf.area / intersection_gdf.total_area
    if op=='larger':
        id_to_keep = intersection_gdf.loc[intersection_gdf.percentage_area_left > threshold, 'id'].unique().tolist()
    elif op=='lte':
        id_to_keep = intersection_gdf.loc[intersection_gdf.percentage_area_left <= threshold, 'id'].unique().tolist()
    else:
        logger.critical('Passed operator is unknow. Please pass "larger" or "lte" (= less than or equal to) as operator.')
        sys.exit(1)

    return id_to_keep


def get_grid_size(tile_size, grid_width=256, grid_height=256, max_dx=0, max_dy=0):
    """Determine the number of grid cells based on the tile size, grid dimension and overlap between tiles.
    All values are in pixels.

    Args:
        tile_size (tuple): tile width and height
        grid_width (int, optional): width of a grid cell. Defaults to 256.
        grid_height (int, optional): height of a grid cell. Defaults to 256.
        max_dx (int, optional): overlap on the width. Defaults to 0.
        max_dy (int, optional): overlap on the height. Defaults to 0.

    Returns:
        number_cells_x: number of grid cells on the width
        number_cells_y: number of grid cells on the height
    """

    tile_width, tile_height = tile_size
    number_cells_x = ceil((tile_width - max_dx)/(grid_width - max_dx))
    number_cells_y = ceil((tile_height - max_dy)/(grid_height - max_dy))

    return number_cells_x, number_cells_y


def grid_over_tile(tile_size, tile_origin, pixel_size_x, pixel_size_y=None, max_dx=0, max_dy=0, grid_width=256, grid_height=256, crs='EPSG:2056', test_shape = None):
    """Create a grid over a tile and save it in a GeoDataFrame with each row representing a grid cell.

    Args:
        tile_size (tuple): tile width and height
        tile_origin (tuple): tile minimum coordinates
        pixel_size_x (float): size of the pixel in the x direction
        pixel_size_y (float, optional): size of the pixels in the y drection. If None, equals to pixel_size_x. Defaults to None.
        max_dx (int, optional): overlap in the x direction. Defaults to 0.
        max_dy (int, optional): overlap in the y direction. Defaults to 0.
        grid_width (int, optional): number of pixels in the width of one grid cell. Defaults to 256.
        grid_height (int, optional): number of pixels in the height of one grid cell. Defaults to 256.
        crs (str, optional): coordinate reference system. Defaults to 'EPSG:2056'.

    Returns:
        GeoDataFrame: grid cells and their attributes
    """

    min_x, min_y = tile_origin

    number_cells_x, number_cells_y = get_grid_size(tile_size, grid_width, grid_height, max_dx, max_dy)

    # Convert dimensions from pixels to meters
    pixel_size_y = pixel_size_y if pixel_size_y else pixel_size_x
    grid_x_dim = grid_width * pixel_size_x
    grid_y_dim = grid_height * pixel_size_y
    max_dx_dim = max_dx * pixel_size_x
    max_dy_dim = max_dy * pixel_size_y

    # Create grid polygons
    polygons = []
    for x in range(number_cells_x):
        for y in range(number_cells_y):
            
            down_left = (min_x + x * (grid_x_dim - max_dx_dim), min_y + y * (grid_y_dim - max_dy_dim))

            # Fasten the process by not producing every single polygon
            if test_shape and not (test_shape.intersects(Point(down_left))):
                continue

            # Define the coordinates of the polygon vertices
            vertices = [down_left,
                        (min_x + (x + 1) * grid_x_dim - x * max_dx_dim, min_y + y * (grid_y_dim - max_dy_dim)),
                        (min_x + (x + 1) * grid_x_dim - x * max_dx_dim, min_y + (y + 1) * grid_y_dim - y * max_dy_dim),
                        (min_x + x * (grid_x_dim - max_dx_dim), min_y + (y + 1) * grid_y_dim - y * max_dy_dim)]

            # Create a Polygon object
            polygon = Polygon(vertices)
            polygons.append(polygon)

    # Create a GeoDataFrame from the polygons
    grid_gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    grid_gdf['id'] = [f'{round(min_x)}, {round(min_y)}' for min_x, min_y in [get_bbox_origin(poly) for poly in grid_gdf.geometry]]

    return grid_gdf


def no_data_to_polygons(image_band, transform, nodata_value, crs="EPSG:2056"):
    """Convert nodata values in raster (numpy array) to polygons
    cf. https://gis.stackexchange.com/questions/295362/polygonize-raster-file-according-to-band-values

    Args:
        images (DataFrame): image dataframe with an attribute named path

    Returns:
        GeoDataFrame: the polygons of the area with nodata values on the read rasters.
    """

    nodata_polygons = []

    nodata_shapes = list(shapes(image_band, mask=image_band == nodata_value, transform=transform))
    nodata_polygons.extend([shape(geom) for geom, value in nodata_shapes])

    nodata_gdf = gpd.GeoDataFrame({'id_nodata_poly': [i for i in range(len(nodata_polygons))], 'geometry': nodata_polygons}, crs=crs)
    # Remove isolated pixels with the same value as nodata
    nodata_gdf = nodata_gdf[nodata_gdf.area > 10].copy()

    return nodata_gdf


def pad_geodataframe(gdf, tile_bounds, tile_size, pixel_size, grid_width=256, grid_height=256, max_dx=0, max_dy=0):
    """Extend the GeoDataFrame of the tile, definded by its bounding box, to match with a specified grid, 
    defined by its cell width, height, and overlapp, as well as the pixel size.
    Save the result in a GeoDataFrame.

    Args:
        gdf (GeoDataFrame): GeoDataFrame in which the result is saved
        tile_bounds (bounds): bounds of the tile
        tile_size (tuple): dimensions of the tile in pixels
        pixel_size (float): size of the pixel in cm
        grid_width (int, optional): number of pixels along the width of one grid cell. Defaults to 256.
        grid_height (int, optional): number of pixels along the hight of one grid cell. Defaults to 256.
        max_dx (int, optional): overlap in pixels along the width. Defaults to 0.
        max_dy (int, optional): overlap in pixels along the height. Defaults to 0.

    Returns:
        gdf: GeoDataFrame with two additional geometries corresponding to the padding on the top and on the right
    """

    min_x, min_y, max_x, max_y = tile_bounds
    tile_width, tile_height = tile_size
    number_cells_x, number_cells_y = get_grid_size(tile_size, grid_width, grid_height, max_dx, max_dy)

    # Get difference between grid size and tile size
    pad_width_px_x = number_cells_x * (grid_width - max_dx) + max_dx - tile_width
    pad_width_px_y = number_cells_y * (grid_height - max_dy) + max_dy - tile_height

    # Convert dimensions from pixels to meters
    pad_width_m_x = pad_width_px_x * pixel_size
    pad_width_m_y = pad_width_px_y * pixel_size

    # Pad on the top
    vertices = [(min_x, max_y),
                (max_x + pad_width_m_x, max_y),
                (max_x + pad_width_m_x, max_y + pad_width_m_y),
                (min_x, max_y + pad_width_m_y)]
    polygon_top = Polygon(vertices)
    
    # Pad on the right
    vertices = [(max_x, min_y),
                (max_x + pad_width_m_x, min_y),
                (max_x + pad_width_m_x, max_y ),
                (max_x, max_y)]
    polygon_right = Polygon(vertices)

    gdf = pd.concat([gdf, gpd.GeoDataFrame({'id_nodata_poly': [10001, 10002], 'geometry': [polygon_top, polygon_right]}, crs="EPSG:2056")], ignore_index=True)

    return gdf


def main(tile_dir, tile_suffix='.tif', output_dir='outputs', subtiles=False, overwrite=False):
    """Get the delimitation of the tiles in a directory

    Args:
        tile_dir (str): path to the directory containing the tiles
        tile_suffix (str, optional): suffix of the filename, which is the part coming after the tile number or id. Defaults to '.tif'.
        output_dir (str, optional): path to the output directory. Defaults to 'outputs'.
        subtiles (bool, optional): whether to generate the subtiles over each tile or not. Defaults to False.

    Returns:
        tiles_gdf: GeoDataFrame with the bounding box and the info of each tile
        nodata_gdf: GeoDataFrame with the nodata areas in the tile bounding boxes
        subtiles_gdf: GeoDataFrame with the bounding box and the info of each subtiles. If `subtiles`==None, returns None
        written_files: list of the written files
    """

    os.makedirs(output_dir, exist_ok=True)
    written_files = [] 

    output_path_tiles = os.path.join(output_dir, 'tiles.gpkg')
    output_path_nodata = os.path.join(output_dir, 'nodata_areas.gpkg')

    if not overwrite and os.path.exists(output_path_tiles) and os.path.exists(output_path_nodata):
        tiles_gdf = gpd.read_file(output_path_tiles)
        nodata_gdf=gpd.read_file(output_path_nodata)
        logger.info('Files for tiles already exist. Reading from disk...')

    else:
        logger.info('Read info for tiles...')
        tile_list = glob(os.path.join(tile_dir, '*.tif'))

        if len(tile_list) == 0:
            logger.critical('No tile in the tile directory.')
            sys.exit(1)

        logger.info('Create a geodataframe with tile info...')
        tiles_dict = {'id': [], 'name': [], 'number': [], 'scale': [], 'geometry': [],
                      'pixel_size_x': [], 'pixel_size_y': [], 'dimension': [], 'origin': []}
        nodata_gdf = gpd.GeoDataFrame()
        for tile in tqdm(tile_list, desc='Read tile info'):

            # Get name and id of the tile
            tile_name = os.path.basename(tile).rstrip(tile_suffix)
            tiles_dict['name'].append(tile_name)
            nbr, x, y = tile_name.split('_')
            tiles_dict['id'].append(f"({x}, {y}, {nbr})")
            tiles_dict['number'].append(nbr)

            with rio.open(tile) as src:
                bounds = src.bounds
                first_band = src.read(1)
                meta = src.meta

            # Set tile geometry
            geom = box(*bounds)
            tiles_dict['geometry'].append(geom)
            tiles_dict['origin'].append(str(get_bbox_origin(geom)))
            tile_size = (meta['width'], meta['height'])
            tiles_dict['dimension'].append(str(tile_size))

            # Guess tile scale
            perimeter = geom.length
            if perimeter <= 2575:
                tile_scale = 500
            elif perimeter <= 4450:
                tile_scale = 1000
            elif perimeter <= 9680:
                tile_scale = 2000
            else:
                tile_scale = 4000
            tiles_dict['scale'].append(tile_scale)
            
            # Set pixel size
            pixel_size_x = abs(meta['transform'][0])
            pixel_size_y = abs(meta['transform'][4])

            try:
                assert round(pixel_size_x, 5) == round(pixel_size_y, 5), f'The pixels are not square on tile {tile_name}: {round(pixel_size_x, 5)} x {round(pixel_size_y, 5)} m.'
            except AssertionError as e:
                print()
                logger.warning(e)

            tiles_dict['pixel_size_x'].append(pixel_size_x)
            tiles_dict['pixel_size_y'].append(pixel_size_y)

            # Transform nodata area into polygons
            temp_gdf = no_data_to_polygons(first_band, meta['transform'], meta['nodata'])
            temp_gdf = pad_geodataframe(temp_gdf, bounds, tile_size, max(pixel_size_x, pixel_size_y), 512, 512)
            temp_gdf = temp_gdf.assign(tile_name=tile_name, scale=tile_scale)
            nodata_gdf = pd.concat([nodata_gdf, temp_gdf], ignore_index=True)

        tiles_gdf = gpd.GeoDataFrame(tiles_dict, crs='EPSG:2056')

        tiles_gdf.to_file(output_path_tiles)
        written_files.append(output_path_tiles)

        nodata_gdf.to_file(output_path_nodata)
        written_files.append(output_path_nodata)

    subtiles_gdf = None
    if subtiles:
       
        logger.info('Determine subtiles...')
        subtiles_gdf = gpd.GeoDataFrame()
        for tile in tqdm(tiles_gdf.itertuples(), desc='Define a grid to subdivide tiles', total=tiles_gdf.shape[0]):
            tile_infos = {
                'tile_size': tuple(map(int, tile.dimension.strip('()').split(', '))), 
                'tile_origin': tuple(map(float, tile.origin.strip('()').split(', '))), 
                'pixel_size_x': tile.pixel_size_x,
                'pixel_size_y': tile.pixel_size_y
            }
            nodata_subset_gdf = nodata_gdf[nodata_gdf.tile_name==tile.name].copy()

            # Make a large tiling grid to cover the image
            temp_gdf = grid_over_tile(grid_width=512, grid_height=512, **tile_infos)

            # Only keep tiles that do not overlap too much the nodata zone
            large_id_on_image = control_overlap(temp_gdf[['id', 'geometry']].copy(), nodata_subset_gdf, threshold=0.5)
            large_subtiles_gdf = temp_gdf[temp_gdf.id.isin(large_id_on_image)].copy()
            large_subtiles_gdf.loc[:, 'id'] = [f'({subtile_id}, {str(tile.number)})' for subtile_id in large_subtiles_gdf.id] 
            large_subtiles_gdf['initial_tile'] = tile.name

            # Make a smaller tiling grid to not lose too much data
            temp_gdf = grid_over_tile(grid_width=256, grid_height=256, **tile_infos)
            # Only keep smal subtiles not under a large one
            small_subtiles_gdf = gpd.overlay(temp_gdf, large_subtiles_gdf, how='difference', keep_geom_type=True)
            small_subtiles_gdf = small_subtiles_gdf[small_subtiles_gdf.area > 10].copy()
            
            if not small_subtiles_gdf.empty:
                # Only keep tiles that do not overlap too much the nodata zone
                small_id_on_image = control_overlap(small_subtiles_gdf[['id', 'geometry']].copy(), nodata_subset_gdf, threshold=0.25)
                small_subtiles_gdf = small_subtiles_gdf[small_subtiles_gdf.id.isin(small_id_on_image)].copy()
                small_subtiles_gdf.loc[:, 'id'] = [f'({subtile_id}, {str(tile.number)})' for subtile_id in small_subtiles_gdf.id]
                small_subtiles_gdf['initial_tile'] = tile.name

                subtiles_gdf = pd.concat([subtiles_gdf, small_subtiles_gdf], ignore_index=True)
            
            subtiles_gdf = pd.concat([subtiles_gdf, large_subtiles_gdf], ignore_index=True)

        logger.info('The tiles are clipped to the image border.')
        subtiles_gdf = gpd.overlay(
            subtiles_gdf, tiles_gdf[['name', 'geometry']], 
            how="intersection", keep_geom_type=True
        )
        subtiles_gdf = subtiles_gdf.loc[subtiles_gdf.initial_tile == subtiles_gdf.name, ['id', 'initial_tile', 'geometry']]

        filepath = os.path.join(output_dir, 'subtiles.gpkg')
        subtiles_gdf.to_file(filepath)
        written_files.append(filepath)

    logger.success('Done determining the tiling!')
    return tiles_gdf, nodata_gdf, subtiles_gdf, written_files
