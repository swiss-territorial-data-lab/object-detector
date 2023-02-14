## Overview

This document provides a detailed description of the procedure to run `proj-dqry` and perform quarries automatic detections. For object detection, scripts developed in the git repository `object-detector` are used. The description of the scripts are presented here: https://github.com/swiss-territorial-data-lab/object-detector

The procedure is defined in three distinct workflows:
1. **Training and Evaluation** workflow allowing to train and evaluate the detection model with a customed dataset reviewed by domain experts and constituing the ground truth. The detector is initially trained on _SWISSIMAGE 10 cm_ mosaic of 2020 ([swisstopo](https://www.swisstopo.admin.ch/fr/geodata/images/ortho/swissimage10.html)) from using the _TLM_ data of _swisstopo_ as Ground Truth.
2. **Prediction** workflow performing inference detection of quarries in a given image dataset (year) thanks to the previously trained model.
3. **Detection monitoring** workflow tracking quarry evolution over years.

Configuration files are used to set the variables parameters. They must be adapted (_i.e._ path to file) if required. In **config** folder, config files relative to `proj-dqry` and `object-detector` are present, one for each defined workflow:

1. **Training and evaluation**: `config-trne.yaml`
2. **Predictions**: `config-prd.yaml`, `config-prd.template.yaml` is used to run prediction batch process for different year datasets
3. **Detection monitoring**: `config-dm.yaml`

A config file dedicated to set the parameters of the detectron2 algorithm is also provided: `detectron2_config_dqry.yaml`

 A synthetic list of command lines to run the whole project can be found at the end of the document.


## Python libraries

The scripts have been developed with Python 3.8 by importing libraries that are listed in `requirements.in` and `requirements.txt`. Before starting to run scripts make sure the required Python libraries that have been used during the code development are installed, otherwise incompatibilities and errors could poltentially occur. A clean method to install Python libraries is to work with a virtual environment preserving the package dependencies.

* create a dedicated Python virtual environment:
	    
      python3 -m venv <dir_path>/[name of the virtual environment]

* activate the virtual environment:

      source <dir_path>/[name of the virtual environment]/bin/activate

* install the required Python packages into the virtual environment:

      pip install -r requirements.txt

The `requirements.txt` file used for the quarries detection can be found in the `proj-dqry` repository. 

* deactivate the virtual environment if not used anymore

      deactivate


## Input data

The input data for the **Training and Evaluation** and **Prediction** workflows for the quarry detection project are stored on the STDL S3 server (with the following access path: /s3/proj-quarries/02_Data/)

In this main folder you can find subfolders:

* DEM
	-  DEMs (spatial resolution about 25 m/px) of Switzerland produced from _SRTM_ instrument ([USGS - SRTM](https://doi.org/10.5066/F7PR7TFT)). The processed product `switzerland_dem.tif` has been downloaded [here](https://github.com/lukasmartinelli/swissdem) with coordinate reference system EPSG:4326 - WGS 84 and then reprojected with QGIS to EPSG:2056 - CH1903+ / LV95 to the used raster `switzerland_dem_EPSG:2056.tif`. The raster is used to filter the quarry detection according to elevation values. Another raster can be found `swiss_srtm.tif` (crs: EPSG:2056 - CH1903+ / LV95) which can be used alternatively (slight differences with `switzerland_dem_EPSG:2056.tif`). `swiss_srtm.tif` raster was used in the parent quarry detection project but the file source is untracked. This raster has been used as elevation input to obtained the final **Prediction** in **debug_mode** (usually 2000 tiles for z16) during test phase.

* Learning models
	- `z*/logs`: folders containing trained detection models obtained during the **Training and Evaluation** workflow using the Ground Truth data. The learning characteristics of the algorithm can be visualized using tensorboard (see below in Processing/Run scripts). Models at several iterations have been saved. The optimum model minimizes the validation loss curve as function of iteration. This model is selected to perform object detection. The algorithm has been trained on _SWISSIMAGE 10 cm_ 2020 mosaic for zoom levels 15 (3.2 m/px) to 18 (0.4 m/px). For each zoom level subfolders, a file `metrics_ite-*.txt` is provided summing-up the metrics values (_precision_, _recall_ and _f1-score_) obtained for the optimized model for which the iteration value corresponds to the value in the file name. The user can either use the already trained model or train his own model by running the **Training and Evaluation** workflow and use the produced model to detect quarries. It is important to note that the training procedure display some random components and therefore the training metrics and results of a new trained model might differ from the ones provided.

* Shapefiles
	- `quarry_label_tlm_revised`: polygons shapefile of the quarries labels (_TLM_ data) reviewed by the domain experts. The data of this file have been used as Ground Truth data to train and assess the automatic detection algorithm.
	- `swissimage_footprints_shape_year_per_year`: original _SWISSIMAGE 10 cm_ footprints and processed polygons border shapefiles for every acquisition year.
	- `switzerland_border`: polygon shapefile of the Switzerland border.

* SWISSIMAGE
	- `Explanation.txt`: file explaining the main characteristics of _SWISSIMAGE_ and the reference links.


## Workflows

This section detail the procedure and the command lines to execute in order to run the **Training and Evaluation** and **Prediction** workflows.

### Training and Evaluation

- Working directory

The working directory can be modified but by default is:

    $ cd proj-dqry/config/

- Config and input data

Configuration files are required:

    [logging_config] = logging.conf

The logging format file can be used as provided. 

    [config_yaml] = config-trne.yaml 

The _yaml_ configuration file has been set for the object detector workflow by reading the dedicated sections. Verify and adapt, if necessary, the input and output paths. 

- Run scripts

The training and detection of objects requires the use of `object-detector` scripts. The workflow is processed in following way:

    $ python3 ../scripts/prepare_data.py [config_yaml]
    $ python3 <path to object-detector>/scripts/generate_tilesets.py [config_yaml]
    $ python3 <path to object-detector>/scripts/train_model.py [config_yaml]
    $ python3 <path to object-detector>/scripts/make_predictions.py [config_yaml]
    $ python3 <path to object-detector>/scripts/assess_predictions.py [config_yaml]

The fisrt script to run is [`prepare_data.py`](/../scripts/README.md) in order to create tiles and labels files that will be then used by the object detector scripts. The `prepare_data.py` section of the _yaml_ configuration file is expected as follow:

    prepare_data.py:
        srs: "EPSG:2056"
        datasets:
            labels_shapefile: ../input/input-trne/[Label_Shapefile]
      output_folder: ../output/output-trne
      zoom_level: [z]

Set the path to the desired label shapefile (AOI) (create a new folder: /proj-dqry/input/input-trne/ to the `proj-dqry` project). The **labels_shapefile** corresponds to polygons of quarries defined in the _TLM_ and manually reviewed by experts. It constitutes the ground truth.

For the quarries example:

	[Label_Shapefile] = tlm-hr-trn-topo.shp

Specify the **zoom level** _z_. The zoom level will act on the tiles size (and so tiles number) and on the pixel resolution. We advise to use zoom level between 16 (1.6 m/px) and 17 (0.8 m/px) to perform quarries detection.
The **srs** key provides the working geographical frame to ensure all the input data are compatible together.

Then, by running `generate_tilesets.py` the images will be downloaded from _WMTS_ server according to the tiles characteristics defined previously. A _XYZ_ connector is used to access _SWISSIMAGE_ for a given year. Be careful to set the desired year **[YEAR]** in the **url** present in the config file:

      https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage-product/default/[YEAR]/3857/{z}/{x}/{y}.jpeg

A **debug mode** can be activated in order to run the script on a sub set of images to perform some test. The number of images sampled is hard coded in the script `generate_tileset.py` of `object-detector`. The images will be split in three datasets: _trn_ (70%), _tst_ (15%) and _val_ (15%) to perform the training. The ground truth with reviewed labels is provided as input.

The algorithm is trained with the script `train_model.py` calling _detectron2_ algorithm. For object detection, instance segmentation is used `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml`. The model characteristics can be tuned by modifying parameters in `detectron2_config_dqry.yaml`.
Especially pay attention to the **MAX_ITER** and **STEPS** values (Last **STEPS** < **MAX_ITER**). Choose a number of iterations to ensure robust model training. The model training evolution can be monitored with a `tensorboard`. The library `PyTorch` must be installed (installed and activated in the virtual environment). 

    tensorboard --logdir <logs folder path>/logs
Open the following link with a web browser: `http://localhost:6006`

The validation loss curve can be visualized and the optimum iteration of the model, _i.e._ the iteration minimizing the validation loss curve can be identified. This iteration corresponds to the optimum and must be set as input trained model for `make_predictions.py` (**model_weights**: pth_file:./logs/[chosen model iteration].pth). The optimum is dependent on the zoom level and the number of tiles involved. The file `model_final.pth` corresponds to the last iteration recorded during the training procedure. Trained model at different iterations are saved in folder **logs**.

Prediction polygons are produced with high resolution (one point by vertices) causing for some configuration RAM saturation and heavy prediction files. A Ramer algorithm can be activated (**enabled**: true otherwise false) in order to simplify the polygon geomtry by deleting non essential points to respect the polygon geometry according to an **epsilon** factor. By default the value is set to 0.5. Increasing **epsilon** filter more points.

Predictions come with a confidence score [0 to 1]. Predictions with low score can be discarded by setting a threshold value for `make_predictions.py` script **score_lower_threshold** (by default set to 0.05). 

Finally, the detection capabilities of the model are evaluated with `assess_prediction.py` by computed metrics, _precision_, _recall_ and _f1_ score based on the inventory of true positive (TP), false positive (FP) and false negative (FN) detections obtained with the _trn_, _tst_ and _val_ datasets. The predictions obtained are compared to the Ground Truth labels.

- Output

Finally we obtained the following results in the folder /proj-dqry/output/output-trne/:

- `*_images`: folders containing downloaded images splitted to the different datasets (all, train, test, validation)
- `sample_tagged_images`: folder containing sample of tagged images and bounding boxes
- `logs`: folder containing the files relative to the model training
- `tiles.geojson`: tiles polygons obtained for the given AOI
- `labels.geojson`: labels polygons obtained for the given GT
- `*_predictions_at_0dot*_threshold.gpkg`: detection polygons obtained for the different datasets (train, test, validation) at a given **score_lower_threshold** (in previous version of `object_detector`: `*_predictions_at_0dot*_threshold.pkl` and `*_predictions.geojson`).
- `img_metadata.json`: images metadata file
- `clipped_labels.geojson`: labels polygons clipped according to tiles
- `split_aoi_tiles.geojson`: tiles polygons sorted according to the dataset where it belongs  
- `COCO_*.json`: COCO annotations for each image of the datasets
-`*.html`: plots representing the prediction results with metrics values (precision, recall, f1) and TP, FP and FN values

### Prediction

- Working directory

The working directory can be modified but by default is:

	$ cd /proj-dqry/config/

- Config and input data

Configuration files are required:

    [logging_config] = logging.conf

The logging format file can be used as provided. 

    [config_yaml] = config-prd.yaml 

The _yaml_ configuration file has been set for the object detector workflow by reading the dedicated sections. Verify and adapt, if necessary, the input and output paths. 

- Run scripts

The prediction of objects requires the use of `object-detector` scripts. The workflow is processed in following way:

    $ python3 ../scripts/prepare_data.py [config_yaml]
    $ python3 <path to object-detector>/scripts/generate_tilesets.py [config_yaml]
    $ python3 <path to object-detector>/scripts/make_predictions.py [config_yaml]

The first script to run is [`prepare_data.py`](/../scripts/README.md) in order to create tiles and labels files that will be then used by the object detector scripts. The `prepare_data.py` section of the _yaml_ configuration file is expected as follow:

    prepare_data.py:
        srs: "EPSG:2056"
        datasets:
            labels_shapefile: ../input/input-trne/[AOI_Shapefile]
      output_folder: ../output/output-prd
      zoom_level: [z]

Set the path to the desired label shapefile (AOI) (create a new folder: /proj-dqry/input/input-prd/ to the project). For prediction, the **labels_shapefile** corresponds to the AOI on which the object detection must be performed. It can be the whole Switzerland or part of it such as the footprint where _SWISSIMAGE_ has been acquired for a given year.  

For the quarries example:

	[Label_Shapefile] = swissimage_footprint_[YEAR].shp

Specify the **zoom level** _z_. The zoom level will act on the tiles size (and so tiles number) and on the pixel resolution. We advise using zoom levels between 16 (1.6 m/px) and 17 (0.8 m/px). The zoom level should be the same as the used model has been trained.
The **srs** key provides the working geographical frame to ensure all the input data are compatible together.

Then, by running `generate_tilesets.py` the images will be downloaded from a _WMTS_ server according to the tiles characteristics defined previously. A _XYZ_ connector is used to access _SWISSIMAGE_ for a given year. Be careful to set the desired **[YEAR]** in the url:

      https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage-product/default/[YEAR]/3857/{z}/{x}/{y}.jpeg

A **debug mode** can be activated in order to run the script on a sub set of images to perform some test. The number of images sampled is hard coded in the script `generate_tileset.py` of `object-detector`. Inference predictions are performed (no Ground Truth data provided). One dataset will be defined (_oth_).

The object predictions are computed with a previously trained model. Copy the desired `logs_*` folder obtained during the **Training and Evaluation** workflow into the folder proj-dqry/input/input-prd/.

  	model_weights: pth_file: '../../input/input-prd/logs/model_*.pth'

Choose the relevant `model_*.pth` file, _i.e._ the one minimizing the validation loss curve (see above Training and Evaluation/Run scripts).

Prediction polygons are produced with high resolution (one point by vertices) causing for some configuration RAM saturation and heavy prediction files. A Ramer algorithm can be activated (**enabled**: true otherwise false) in order to simplify the polygon geomtry by deleting non essential points to respect the polygon geometry according to an **epsilon** factor. By default the value is set to 2.0. Increasing **epsilon** filter more points.

Predictions come with a confidence score [0 to 1]. Predictions with low score can be discarded by setting a threshold value for `make_predictions.py` script **score_lower_threshold** (by default set to 0.3). 


- Batch process

The **Prediction** workflow has been automatized and can be run for batch of years using `batch_process.sh` in **scripts** folder along with `config.prd.template.yaml` in **config** folder.

The list of years to process must be specified as input of the shell script 

    for year in YEAR1 YEAR2 YEAR3 ... 

By executing the command:

    $ ../scripts/batch_process.sh

`config-prd_[YEAR].yaml` will be generated for a given year and the command list in `batch_process.sh` will be executed for the provided list of years. 

The paths and value of _yaml_ configuration file template must be adapted, if necessary.

- Output:

Finally we obtained the following results stored in the folder /proj-dqry/output/output-prd/:

- `*_images`: folders containing downloaded images of an AOI to perform prediction inference
- `sample_tagged_images`: folder containing sample of tagged images and bounding boxes
- `tiles.geojson`: tiles polygons obtained for the given AOI
- `oth_predictions_at_0dot*_threshold.gpkg`: detection polygons obtained for the AOI at a given **score_lower_threshold** (in previous version of `object_detector`: `oth_predictions_at_0dot*_threshold.pkl` and `oth_predictions.geojson`).
- `img_metadata.json`: images metadata file
- `clipped_labels.geojson`: labels polygons clipped according to tiles
- `split_aoi_tiles.geojson`: tiles polygons sorted according to the dataset where it belongs  
- `COCO_oth.json`: COCO annotations for each image

- Post-processing

The object detection output (`oth_predictions_at_0dot*_threshold.gpkg`or `oth_predictions_at_0dot*_threshold.geojson` (previous version of `make_predictions.py`)) obtained via the `object-detector` scripts needs to be filtered to discard false detections and improve the aesthetic of the polygons (merge polygons belonging to a single quarry). The script [`prediction_filter.py`](/../script/README.md) allows to extract the prediction out of the detector based on a series of provided threshold values.

The `prediction_filter.py` is run as follow:

- Working directory

The working directory can be modified but is by default:

	$ cd /proj-dqry/config/
    
- Config and input data

	[config_yaml] = config-prd.yaml

The script expects a prediction file (`oth_predictions_at_0dot*_threshold.gpkg`) obtained with all polygons geometry and a _score_ value normalized [0,1]. The `prediction_filter.py` section of the `config-prd.yaml` file is expected as follow. Paths and threshold values must be adapted:

    prediction_filter.py:
        year:[YEAR] 
        input: ../output/output-prd/oth_predictions_at_0dot*_threshold.gpkg
        labels_shapefile: ../input/input-prd/[AOI_Shapefile] 
        dem: ../input/input-prd/[DEM.tif] 
        elevation: [THRESHOLD VALUE]   
        score: [THRESHOLD VALUE]
        distance: [THRESHOLD VALUE] 
        area: [THRESHOLD VALUE] 
        output: ../output/output-prd/oth_prediction_at_0dot*_threshold_year-[YEAR]_score-{score}_area-{area}_elevation-{elevation}_distance-{distance}.geojson


-**year**: year of the dataset used as input for filtering

-**input**: indicate path to the input file that needs to be filtered, _i.e._ `oth_predictions_at_0dot*_threshold.gpkg`

-**labels_shapefile**: AOI of interest is used to remove polygons that are located partially outside the AOI. For the quarry project we used the _SWISSIMAGE_ acquisition footprint for a given year. The shapefiles can be found in the STDL S3 storage (s3/proj-quarries/02_Data/Shapefiles/swissimage_footprints_shape_per_year/swissimage_footprints_border/).

-**dem**: indicate the path to a DEM of Switzerland. SRTM derived product is used and can be found in the STDL S3 storage(s3/proj-quarries/02_Data/DEM/`switzerland_dem_EPSG:2056.tif`). A threshold elevation is used to discard detection above the given value.

-**elevation**: altitude above which predictions are discarded. Indeed 1st tests have shown numerous false detection due to snow cover area (reflectance value close to bedrock reflectance) or mountain bedrock exposure that are mainly observed in altitude.. By default the threshold elevation has been set to 1200.0 m.

-**score**: each polygon comes with a confidence score given by the prediction algorithm. Polygons with low scores can be discarded. By default the value is set to 0.95.

-**distance**: two polygons that are close to each other can be considered to belong to the same quarry. Those polygons can be merged into a single one. By default the buffer value is set to 10 m.

-**area**: small area polygons can be discarded assuming a quarry has a minimal area. The default value is set to 5000 m2.

-**output**: provide the path of the filtered polygons shapefile with prediction score preserved. The output file name will be formated as: `oth_prediction_at_0dot*_threshold_year-{year}_score-{score}_elevation-{elevation}_distance-{distance}_area-{area}.geojson`.


The script `prediction_filter.py` is run as follow:

    $ python3 ../scripts/prediction-filter.py [config_yaml]


It has to be noted that different versions of the `prediction_filter.py` have been used to produce the results. The predictions obtained during the test phase (**debug_mode**) and provided were produced by taking `oth_predictions_at_0dot*_threshold.geojson` as input. In this case, the elevation filtering was processed at the end with DEM `swiss_srtm.tif`.

The final predictions for years from 1999 to 2000 are stored in the STDL S3 server with the following access path: /s3/proj-quarries/03_Results/Prediction/. 

### Detection monitoring

The **Prediction** workflow computes object detections on images acquired at different years. In order to monitor the detected object over years, the script [`detection_monitoring.py`](/../scripts/README.md) has been developed.

The `detection_monitoring.py` is run as follow:

- Working directory

The working directory can be modified but is by default:

    $ cd /proj-dqry/config/
    
- Config and input data

    [config_yaml] = config-dm.yaml 

The `detection_monitoring.py` section of the _yaml_ configuration file is expected as follow:

    detection_monitoring.py:  
    years: [YEAR1, YEAR2, YEAR3,...]       
    datasets:
        detection: ../input/input-dm/oth_prediction_at_0dot*_threshold_year-{year}_score-[SCORE]_elevation-[elevation]_distance-[distance]_area-[area].geojson 
    output_folder: ../output/output-dm
  
Paths must be adapted if necessary (create a new folder: /proj-dqry/input/input-dm/ to the project to copy the input files of different years in it). The script takes as input a _geojson_ file (`oth_prediction_at_0dot*_threshold_year-{year}_[filters_list].geojson`) obtained previously with the script `prediction_filter.py` for different years. The list of years required for the object monitoring must be specified in **years**.

-Run scripts

The prediction of objects requires the use of `object-detector` scripts. The workflow is processed in following way:

    $ python3 ../scripts/detection_monitoring.py [config_yaml]


The outputs are a _geojson_ and _csv_ (`quarry_time`) files saving predictions over years with their caracteristics (_ID_object_, _ID_feature_, _year_, _score_, _area_, _geometry_). The prediction files computed for previous years and `quarry_times` files can be found on the STDL S3 storage with the following access path: /s3/proj-quarries/03_Results/Detection_monitoring/.

### Plots

Script to draw basic plots is provided with [`plots.py`](/../scripts/README.md).

- Working directory

The working directory can be modified but is by default:

	$ cd /proj-dqry/config/

- Config and input data

	[config_yaml] = config-dm.yaml

The `plots.py` section of the _yaml_ configuration file is expected as follow:

	plots.py:  
	object_id: [ID_OBJECT1, ID_OBJECT2, ID_OBJECT3,...]
	plots: ['area-year']
	datasets:
    	detection: ../output/output-dm/quarry_times.geojson
	output_folder: ../output/output-dm/plots

Paths must be adapted if necessary. The script takes as input a `quarry_times.geojson` file produced with the script `detection_monitoring.py`. The list of object_id is required and the type of plot as well. So far only 'area-year' plot is available. Additional types of plots can be added in the future.

-Run scripts

	$ python3 ../scripts/prediction-filter.py [config_yaml]

## Global workflow
    
Following the end to end workflow can be run by issuing the following list of actions and commands:

Copy `proj-dqry` and `object-detector` repository in a same folder.  

    $ cd proj-dqry/
    $ python3 -m venv <dir_path>/[name of the virtual environment]
    $ source <dir_path>/[name of the virtual environment]/bin/activate
    $ pip install -r requirements.txt

    $ mkdir input
    $ mkdir input-trne
    $ mkdir input-prd
    $ mkdir input-dm
    $ cd proj-dqry/config/

Adapt the paths and input value of the configuration files accordingly.

**Training and evaluation**: copy the required input files (labels shapefile (tlm-hr-trn-topo.shp) and trained model is necessary (`z*/logs`)) to **input-trne** folder.

    $ python3 ../scripts/prepare_data.py config-trne.yaml
    $ python3 ../../object-detector/scripts/generate_tilesets.py config-trne.yaml
    $ python3 ../../object-detector/scripts/train_model.py config-trne.yaml
    $ tensorboard --logdir ../output/output-trne/logs

Open the following link with a web browser: `http://localhost:6006` and identified the iteration minimizing the validation loss curve and the selected model name (**pth_file**) in `config-trne` to run `make_predictions.py`. 

    $ python3 ../../object-detector/scripts/make_predictions.py config-trne.yaml
    $ python3 ../../object-detector/scripts/assess_predictions.py config-trne.yaml

**Predictions**: copy the required input files (AOI shapefile (`swissimage_footprint_[YEAR].shp`), trained model (`/z*/logs`) and DEM (`switzerland_dem_EPSG:2056.tif`)) to **input-prd** folder.

    $ python3 ../scripts/prepare_data.py config-prd.yaml
    $ python3 ../../object-detector/scripts/generate_tilesets.py config-prd.yaml
    $ python3 ../../object-detector/scripts/make_predictions.py config-prd.yaml
    $ python3 ../scripts/prediction-filter.py config-prd.yaml 

The workflow has been automatized and can be run for batch of years by running this command:

    $ ../scripts/batch_process.sh

**Object Monitoring**: copy the required input files (filtered prediction files (`oth_prediction_filter_year-{year}_[filters_list].geojson`)) to **input-dm** folder.

    $ python3 ../scripts/detection_monitoring.py config-dm.yaml
    $ python3 ../scripts/plots.py config-dm.yaml
