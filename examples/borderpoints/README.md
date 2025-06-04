# Example: classification of the border points based on the cadastral plans

A working setup is provided here to test the multi-class classification and the use of images from another folder.
It consists of the following elements:

* ready-to-use configuration files:
* input data
* scripts for data preparation and post-processing

The historical plans can be large files. In our case, 32 GB of RAM were required to transform the color of the image from color map to RGB space. In case only 16 GB of RAM are available, excluding the plan XXX solves the bottleneck.


### Installation

The installation is performed from this folder with the following steps:

* Clone the [STDL object detector](https://github.com/swiss-territorial-data-lab/object-detector),
* Get into the `object-detector` folder,
* The dockerfile of this project supposes the existence on the machine of an image called `object-detector-stdl-objdet`. 
    * You can control the image existence by listing the available images with `docker images ls`.
    * If it is not available, build it from the folder of the object detector with `docker compose build`.
    * You can control the installation by running `docker compose run --rm stdl-objdet stdl-objdet -h`.
* Go back to the folder `proj-borderpoints`,
* Build docker,
* Run docker,
* Go to `proj-borderpoints` directory in docker.


The corresponding command lines are

```
git clone https://github.com/swiss-territorial-data-lab/object-detector.git
docker images ls
cd object-detector
docker compose build
cd -
docker compose build
docker compose run --rm borderpoints
cd proj-borderpoints            # Command to run in the docker bash
```

**All workflow commands are supposed to be launched in Docker from the `proj-borderpoints` directory.**


## Data

The data used for the proof of concept are available in the `data` folder, except for the plans, which are downloadable [here](https://map.geo.fr.ch/STDL_Plans_georeferences/STDL_Plans_georeferences.zip).

The following data are necessary for the segmentation and the post-processing:

* Plans: 
    * RGB images or images with a color map in EPSG:2056
    * provided by the Canton of Fribourg
    * to be [downloaded](https://map.geo.fr.ch/STDL_Plans_georeferences/STDL_Plans_georeferences.zip),  placed in the `data` folder, and unzipped. The expected path is `data/STDL_Plans_georeferences`.
* Approximated missing points:
    * vector layer with the missing points of cadastral surveying with their position approximated from the digitized lines
    * provided by the Canton of Fribourg
    * path: `data/ground_truth/Shapefile PL_manquants/PL_manquant_BDMO2.gpkg`
* land cover:
    * subset of the swissTLM3D layer with the land cover of the area of interest to classify missed non-materialized points
    * [metadata of the original dataset](https://www.swisstopo.admin.ch/en/landscape-model-swisstlm3d)
    * path: `data/land_cover.gpkg`
* settlement areas:
    * subset of the vector layer with the settlement areas to improve matching between points and segmented polygons in those areas
    * [metadata of the original dataset](https://www.geocat.ch/datahub/dataset/4229c353-e780-42d8-9f8c-298c83920a3a)
    * path: `data/siedlung_subset_2024_2056_FR.gpkg`

When working with the ground truth (GT), the following files are required in addition:

* Bounding boxes:
    * vector layer of the areas were all the cadastral points were digitized
    * provided by the Canton of Fribourg
    * path: `data/ground_truth/Realite_terrain_Box/PL_realite_terrain_box.shp`
* Polygon GT:
    * vector layer with the delineation and class of all the cadastral points in the bounding boxes
    * provided by the Canton of Fribourg
    * path: `data/data/ground_truth/Realite_terrain_Polygone/PL_realite_terrain_polygones.shp`
* Point GT:
    * point vector layer with class of all the approximately known cadastral points in the bounding boxes
    * provided by the Canton of Fribourg
    * path: `data/data/ground_truth/Realite_terrain_Polygone/PL_realite_terrain_points.shp`
* Plan scales: 
    * Excel file with the number and scale of each plan used for the GT
    * path: `data/data/plan_scales.xlsx`
* Cadastral survey data: 
    * vector layer with the approximate position of cadastral points used to match detections with existing points
    * produced based on the polygon dataset of the cadastral survey of the Canton of Fribourg at the time
    * path: `data/BDMO2_subset.gpkg`


## General workflow

The workflow is divided into three parts:

* Data preparation: call the appropriate preprocessing script, *i.e.* `prepare_ground_truth.py` to work with ground truth produced on defined bounding boxes and `prepare_entire_plans.py` to work with entire plans. More precisely, the following steps are performed:
    - Transform the plans from a color map to RGB images,
    - If ground truth is available, format the labels according to the requirements of the STDL object detector and clip the plans to the bounding box of the ground truth,
    - Generate a vector layer with the information of the subtiles dividing the plans into square tiles of 512 or 256 pixels,
    - Clip the plan to the subtiles.
* Detection of the border points with the STDL object detector: the necessary documentation is available in the [associated GitHub repository](https://github.com/swiss-territorial-data-lab/object-detector)
* Post-processing: produce one file with all the detections formatted according to the expert requirements.
    - `post_processing.py`: the detections are filtered by their confidence score and ...
    - `point_matching.py`: the detections are matched with the points of the cadastral surveying for areas where it is not fully updated yet,
    - `check_w_land_cover.py`: use the data on land cover to assign the class "non-materialized point" to undetermined points in building and stagnant waters.
    - `heatmap.py`: highlight areas with a high concentration of false positive points.

All the parameters are passed through a configuration file. Some fixed parameters are set for the whole process in `constants.py`.

**Training with GT**

```
python scripts/instance_segmentation/prepare_ground_truth.py config/config_w_gt.yaml
stdl-objdet generate_tilesets config/config_w_gt.yaml
stdl-objdet train_model config/config_w_gt.yaml
stdl-objdet make_detections config/config_w_gt.yaml
stdl-objdet assess_detections config/config_w_gt.yaml
```

The post-processing can be performed and the detections assessed again with the following commands:

```
python scripts/post_processing/post_processing.py config/config_w_gt.yaml
python scripts/instance_segmentation/assess_w_post_process.py config/config_w_gt.yaml
```

In the configuration file, the parameters `keep_datasets` must be set to `True` to preserve the split of the training, validation and test datasets.

Performing the point matching is possible with the ground truth.

```
python scripts/post_processing/point_matching.py config/config_w_gt.yaml
python scripts/post_processing/check_w_land_cover.py config/config_w_gt.yaml
python scripts/instance_segmentation/assess_point_classif.py config/config_w_gt.yaml
```

**Inference on entire plans**

```
python scripts/instance_segmentation/prepare_entire_plans.py config/config_entire_plans.yaml
stdl-objdet generate_tilesets config/config_entire_plans.yaml
stdl-objdet make_detections config/config_entire_plans.yaml
python scripts/post_processing/post_processing.py config/config_entire_plans.yaml
python scripts/post_processing/point_matching.py config/config_entire_plans.yaml
python scripts/post_processing/check_w_land_cover.py config/config_entire_plans.yaml
python scripts/post_processing/heatmap.py config/config_entire_plans.yaml
```
