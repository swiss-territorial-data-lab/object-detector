# Example: quarry automatic detection and monitoring over Switzerland

## Overview

This example project provides data and scripts, allowing the end-user to detect and identified quarries for a given Area of Interest (AOI) and year in Switzerland. This example is illustrating automatic object detection and tracking in different years image datasets and the option to add random empty tiles to the training process.

The procedure is defined in three distinct workflows:
1. **Training and Evaluation** workflow allowing to train and evaluate the detection model with a customed dataset reviewed by domain experts and constituing the ground truth. The detector is initially trained on _SWISSIMAGE 10 cm_ mosaic of 2020 ([swisstopo](https://www.swisstopo.admin.ch/fr/geodata/images/ortho/swissimage10.html)), using the _TLM_ data of _swisstopo_ as Ground Truth.
2. **Prediction** workflow performing inference detection of quarries in a given image dataset (_SWISSIMAGE_ acquisition year) thanks to the previously trained model.
3. **Detection monitoring** workflow tracking quarry evolution over years.

A global documentation of the project can be found [here](https://github.com/swiss-territorial-data-lab/stdl-tech-website/tree/master/docs/PROJ-DQRY). 

**TOC**
- [Requirements](#requirements)
- [Provided assets](#provided-assets)
- [Workflow](#workflow)
- [Disclaimer](#disclaimer)
- [Copyright and License](#copyright-and-license)


## Requirements

The minimium hardware and software requirements are the following:

- Ubuntu 20.04
- 15 GiB RAM machine 
- 15 GiB GPU

## Provided assets

- the read-to-use configuration files in `config` subfolder: 
    - `config_dm.yaml`
    - `config_trne.yaml`
    - `config_prd.yaml`
    - `detectron2_config_dqry.yaml`,
- the input data in the `input` subfolders:
    - For **Training and Evaluation**, `input-trne`: 
        - labeled quarries (_i.e._ Ground Truth) from the product swissTLM3D
        - border shape of Switzerland
    - For **Prediction**, `input-prd`: 
        - _SWISSIMAGE_ footprint image acquisition (`swissimage_footprint_YEAR.*`) for 2017 and 2020 (overlapping footprints) delineating the AOI
        - border shape of Switzerland
        - Swiss DEM raster can be dowloaded from this [link](https://github.com/lukasmartinelli/swissdem) with coordinate reference system EPSG:4326 - WGS 84. The raster should first be reproject to EPSG:2056 - CH1903+ / LV95 named `switzerland_dem_EPSG2056.tif`and located to this subfolder.
- the pre-processing and post-processing scripts in `scripts` subfolder:
    - data preparation script (`prepare_data.py`) producing the files to be used as input to the `generate_tilesets.py`script
    - filtering script (`prediction_filter`) producing the final prediction files after performing instance prediction 
    - the object tracking script (`detection_monitoring.py`) identifying overlapping object detections in several dataset and attributing them an unique object ID.
    - Detection area as function of time for a given unique object ID can be plotted with script `plot.py`. 


## Workflow
    
First create and activate a new virtual environment in python 3.8: 

    $ sudo apt-get install -y python3-gdal gdal-bin libgdal-dev gcc g++ python3.8-dev
    $ python3 -m venv <dir_path>/[name of the virtual environment]
    $ source <dir_path>/[name of the virtual environment]/bin/activate

Following the end-to-end workflow can be run by issuing the following list of actions and commands:

    $ cd config/
    $ pip install -r ../../../requirements.txt

**Training and evaluation**:

    $ python3 ../scripts/prepare_data.py config-trne.yaml
    $ python3 ../../../scripts/generate_tilesets.py config-trne.yaml
    $ python3 ../../../scripts/train_model.py config-trne.yaml
    $ tensorboard --logdir ../output/output-trne/logs

Open the following link with a web browser: `http://localhost:6006` and identified the iteration minimizing the validation loss curve and the selected model name (**pth_file**) in `config-trne` to run `make_predictions.py`. 

    $ python3 ../../../scripts/make_predictions.py config-trne.yaml
    $ python3 ../../../scripts/assess_predictions.py config-trne.yaml

**Predictions**: 

    $ python3 ../scripts/prepare_data.py config-prd.yaml
    $ python3 ../../../scripts/generate_tilesets.py config-prd.yaml
    $ python3 ../../../scripts/make_predictions.py config-prd.yaml
    $ python3 ../scripts/prediction_filter.py config-prd.yaml 

Run **Prediction** workflow for year 2017 and 2020 (to be changed in `config-prd.yaml`).  

**Object Monitoring**: 

    $ mkdir ../input/input-dm     
    $ cp ../output/output-prd/2017/oth_prediction_at_0dot3_threshold_year-2017_score-0dot95_area-5000_elevation-1200_distance-10.geojson ../input/input-dm
    $ cp ../output/output-prd/2020/oth_prediction_at_0dot3_threshold_year-2020_score-0dot95_area-5000_elevation-1200_distance-10.geojson ../input/input-dm
    $ python3 ../scripts/detection_monitoring.py config-dm.yaml

In `config-dm.yaml` indicate the quarry unique ID (**object_id**) to track.  

    $ python3 ../scripts/plots.py config-dm.yaml

We strongly encourage the end-user to review the provided config_*_.yaml files as well as the various output files, a list of which is printed by each script before exiting.


## Disclaimer

The results provided by the _quarry-detection-monitoring_ example are resulting from numerical implementation providing segmentation of **potential** quarry sites. False positive and false negative detection, inherent to deep learning automatic methods, are present in the final detection dataset. A **manual inspection** of the detection must be performed prior to data exploitation and interpretation.

## Copyright and License
 
**proj-dqry** - Nils Hamel, Adrian Meyer, Huriel Reichel, Clémence Herny, Shanci Li, Alessandro Cerioni, Roxane Pott <br >
Copyright (c) 2020-2022 Republic and Canton of Geneva

This program is licensed under the terms of the GNU GPLv3. Documentation and illustrations are licensed under the terms of the CC BY 4.0.