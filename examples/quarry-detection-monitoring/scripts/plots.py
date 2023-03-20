
#!/bin/python
# -*- coding: utf-8 -*-

#  Proj quarry detection and time machine
#
#      Clemence Herny 
#      Shanci Li
#      Alessandro Cerioni 
#      Nils Hamel - nils.hamel@alumni.epfl.ch
#      Huriel Reichel
#      Copyright (c) 2020 Republic and Canton of Geneva
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# 
################################################################
#  Script used to plot and visualize quarry data trough times 
#  Inputs are defined in config-dm.yaml
 

import os, sys
import logging
import logging.config
import argparse
import yaml
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# the following allows us to import modules from within this file's parent folder
sys.path.insert(0, '.')

logging.config.fileConfig('logging.conf')
logger = logging.getLogger('root')

if __name__ == "__main__":

    # Start chronometer
    logger.info('Starting...')

    # Argument and parameter specification
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='Framework configuration file')
    args = parser.parse_args()
    logger.info(f"Using {args.config_file} as config file.")

    with open(args.config_file) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)[os.path.basename(__file__)]

    # Load input parameters
    DETECTION = cfg['datasets']['detection']
    QUARRIES = cfg['object_id']
    OUTPUT_DIR = cfg['output_folder']
    PLOTS = cfg['plots']


    # Create an output directory in case it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load prediction 
    gdf = gpd.read_file(DETECTION)
    gdf = gdf.sort_values(by=['year'], ascending=False)

    for PLOT in PLOTS:
        
        # Plot the quarry area vs time 
        if PLOT == 'area-year':
            logger.info(f"Plot {PLOT}")
            fig, ax = plt.subplots(figsize=(8,5))
            for QUARRY in QUARRIES:
                x = gdf.loc[gdf["id_object"] == QUARRY,["year"]]
                y = gdf.loc[gdf["id_object"] == QUARRY,["area"]]
                id = QUARRY
                ax.scatter(x, y, label=id)
                ax.plot(x, y, linestyle="-")
                ax.set_xlabel("Year", fontweight='bold')
                ax.set_ylabel(r"Area (m$^2$)", fontweight='bold')
                ax.ticklabel_format(axis='y', style='sci',scilimits=(0,0))
                ax.legend(title='Object ID', loc=[1.05,0.5] )
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            plot_path = os.path.join(OUTPUT_DIR, 'quarry_area-year.png')
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.show()