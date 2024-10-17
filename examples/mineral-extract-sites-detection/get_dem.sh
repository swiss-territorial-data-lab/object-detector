#!/bin/bash

mkdir -p ./data/DEM/
wget https://github.com/lukasmartinelli/swissdem/releases/download/v1.0/switzerland_dem.tif -O ./data/DEM/switzerland_dem.tif
gdalwarp -t_srs "EPSG:2056" ./data/DEM/switzerland_dem.tif ./data/DEM/switzerland_dem_EPSG2056.tif
rm ./data/DEM/switzerland_dem.tif