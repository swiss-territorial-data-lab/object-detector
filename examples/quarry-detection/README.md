# Automatic detection and monitoring of mineral extraction sites in Switzerland

Clémence Herny<sup>1,2</sup>, Shanci Li<sup>1,3</sup>, Alessandro Cerioni<sup>1,4</sup>, Roxane Pott<sup>1,5</sup>

<sup>1</sup> STDL  <br/>
<sup>2</sup> ExoLabs  <br/>
<sup>3</sup> Uzufly  <br/>
<sup>4</sup> Etat de Genève  <br/>
<sup>5</sup> Swisstopo  <br/>


Proposed by swisstopo - PROJ-DQRY <br />
October 2022 to February 2023 - Published on ..., 2023

## Abstract

_Mineral extraction site (MES) monitoring is primordial for mineral resources management and environmental impact assessment. Within this scope, swisstopo has solicited the STDL to automatize the vectorization of MES over the years. This tedious task was previously performed manually and was not regularly updated. Automatic object detection has been done with Deep Learning methods on open data SWISSIMAGE RGB orthophotos (spatial resolution of 1.6 m px<sup>-1</sup>). The model proved its ability to detect MES accurately (achieving an F1 score of 82%). Inference prediction of potential MES was performed in images from 1999 to 2021, allowing us to follow the MES evolution through several years. Although the results are satisfactory, a careful review of detection must be performed by experts to validate it as actual MES. Despite this remaining manual work, the process is sped up compared to manual vectorization and can be used in the future to keep up-to-date with the MES information._

## Overview

This example project provides data and scripts, allowing the end-user to detect and identify mineral extraction sites (latter referred to as quarries) for a given Area of Interest (AOI) and year in Switzerland. This example illustrates automatic object detection and tracking in different years' image datasets and the option to add random empty tiles to the training process.

The procedure is defined in three distinct workflows:
1. **Training and Evaluation** workflow allows to train and evaluate the detection model with a customed dataset reviewed by domain experts and constituting the ground truth. The detector is initially trained on _SWISSIMAGE 10 cm_ mosaic of 2021 ([swisstopo](https://www.swisstopo.admin.ch/fr/geodata/images/ortho/swissimage10.html)), using the _TLM_ data of _swisstopo_ as Ground Truth.
2. **Prediction** workflow performing inference detection of quarries in a given image dataset (_SWISSIMAGE_ acquisition year) thanks to the previously trained model.
3. **Detection monitoring** workflow tracking quarry evolution over the years.

Global documentation of the project can be found [here](https://github.com/swiss-territorial-data-lab/stdl-tech-website/tree/master/docs/PROJ-DQRY). 

**TOC**
- [Requirements](#requirements)
- [Provided assets](#provided-assets)
- [Workflow](#workflow)
- [Disclaimer](#disclaimer)
- [Copyright and License](#copyright-and-license)


## Requirements

The minimium hardware and software requirements are the following:

- 15 GiB RAM machine 
- 16 GiB GPU
- Ubuntu 20.04
- CUDA version 11.3
- PyTorch version 1.10
- Python 3.8


## Provided assets

- the read-to-use configuration files in the `config` subfolder: 
    - `config_dm.yaml`
    - `config_trne.yaml`
    - `config_prd.yaml`
    - `detectron2_config_dqry.yaml`,
- the input data in the `input` subfolders:
    - For **Training and Evaluation**, `input-trne`: 
        - labeled quarries (_i.e._ Ground Truth) from the product swissTLM3D
        - border shape of Switzerland
    - For **Prediction**, `input-prd`: 
        - _SWISSIMAGE_ footprint image acquisition (`swissimage_footprint_YEAR.*`) for 2018 and 2021 (overlapping footprints) delineating the AOI
        - border shape of Switzerland
        - the swiss DEM raster can be downloaded from this [link](https://github.com/lukasmartinelli/swissdem) with coordinate reference system EPSG:4326 - WGS 84. The raster should be first reprojected to EPSG:2056 - CH1903+ / LV95 named `switzerland_dem_EPSG2056.tif`and located in this subfolder.
- the pre-processing and post-processing scripts in the `scripts` subfolder:
    - data preparation script (`prepare_data.py`) producing the files to be used as input to the `generate_tilesets.py`script.
    - filtering script (`prediction_filter`) producing the final prediction files after performing instance prediction. 
    - the object tracking script (`detection_monitoring.py`) identifying overlapping object detections in several datasets and attributing them to a unique object ID.
    - Detection area as a function of time for a given unique object ID can be plotted with script `plot.py`. 


## Workflow

<p align="center">
<img src="./images/dqry_workflow_graph.png?raw=true" width="100%">
<br />
<i>Workflow scheme.</i>
</p>


First, create and activate a new virtual environment in python 3.8: 

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

Run **Prediction** workflow for the year 2018 and 2021 (to be changed in `config-prd.yaml`).  

**Object Monitoring**: 

    $ mkdir ../input/input-dm     
    $ cp ../output/output-prd/2018/oth_prediction_at_0dot3_threshold_year-2018_score-0dot95_area-5000_elevation-1200_distance-10.geojson ../input/input-dm
    $ cp ../output/output-prd/2021/oth_prediction_at_0dot3_threshold_year-2021_score-0dot95_area-5000_elevation-1200_distance-10.geojson ../input/input-dm
    $ python3 ../scripts/detection_monitoring.py config-dm.yaml

In `config-dm.yaml` indicate the quarry unique ID (**object_id**) to track.  

    $ python3 ../scripts/plots.py config-dm.yaml

We strongly encourage the end-user to review the provided config_*_.yaml files as well as the various output files, a list of which is printed by each script before exiting.


## Disclaimer

Depending on the end purpose, we strongly recommend users not to take for granted the detections obtained through this code. Indeed, results can exhibit false positives and false negatives, as is the case in all Machine Learning-based approaches.

## Copyright and License
 
**proj-dqry** - Nils Hamel, Adrian Meyer, Huriel Reichel, Clémence Herny, Shanci Li, Alessandro Cerioni, Roxane Pott <br >
Copyright (c) 2020-2022 Republic and Canton of Geneva

This program is licensed under the terms of the GNU GPLv3. Documentation and illustrations are licensed under the terms of the CC BY 4.0.