# Example: classification of the border points based on the cadastral plans

A working setup is provided here to test the multi-class classification and the use a folder as the image source.
It consists of the following elements:

* ready-to-use configuration files:
* input data
* scripts for data preparation and the 1st step of post-processing

The full project is available is its [own repository](https://github.com/swiss-territorial-data-lab/proj-borderpoints).


The **installation** can be carried out by following the instructions in the main readme file. When using docker, the container must be launched before running the workflow:

```bash
$ sudo chown -R 65534:65534 examples
$ docker compose run --rm -it stdl-objdet
```

The worklfow commands are expected to be launch from this folder, with or without docker.

The docker container is exited and permissions restored with:

 ```bash
$ exit
$ sudo chmod -R a+w examples
```

## Data

The data used for the proof of concept are available in the `data` folder, except for the plans, which are downloadable [here](https://map.geo.fr.ch/STDL_Plans_georeferences/STDL_Plans_georeferences.zip).

The following data are necessary for the segmentation and the post-processing:

* Plans: 
    * RGB images or images with a color map in EPSG:2056
    * provided by the Canton of Fribourg
    * to be [downloaded](https://map.geo.fr.ch/STDL_Plans_georeferences/STDL_Plans_georeferences.zip),  placed in the `data` folder, and unzipped. The expected path is `data/STDL_Plans_georeferences`.

When working with the ground truth (GT), the following files are required in addition:

* Bounding boxes:
    * vector layer of the areas were all the cadastral points were digitized
    * provided by the Canton of Fribourg
    * path: `data/ground_truth/Realite_terrain_Box/PL_realite_terrain_box.shp`
* Polygon GT:
    * vector layer with the delineation and class of all the cadastral points in the bounding boxes
    * provided by the Canton of Fribourg
    * path: `data/ground_truth/Realite_terrain_Polygone/PL_realite_terrain_polygones.shp`
* Plan scales: 
    * Excel file with the number and scale of each plan used for the GT
    * path: `data/plan_scales.xlsx`


## General workflow

The workflow is divided into three parts:

* Data preparation with `prepare_data.py`:
    - Transform the plans from a color map to RGB images,
    - If ground truth is available, format the labels according to the requirements of the STDL object detector and clip the plans to the bounding box of the ground truth,
    - Generate a vector layer with the information of the subtiles dividing the plans into square tiles of 512 or 256 pixels,
    - Clip the plan to the subtiles.
* Detection of the border points with the STDL object detector
* Post-processing: produce one file with all the detections formatted according to the expert requirements.
    - `post_processing.py`: the detections are filtered by their confidence score and merged to their neighbors on adjacent tiles if they share the same class,
    - some additional post-processing steps were perfomed and are not provided here.

All the parameters are passed through the configuration files in the `config` folder.

**Training with GT**

```
python prepare_data.py config/config_w_gt.yaml
stdl-objdet generate_tilesets config/config_w_gt.yaml
stdl-objdet train_model config/config_w_gt.yaml
stdl-objdet make_detections config/config_w_gt.yaml
stdl-objdet assess_detections config/config_w_gt.yaml
python post_processing.py config/config_w_gt.yaml
```

**Inference on entire plans**

```
python prepare_data.py config/config_entire_plans.yaml
stdl-objdet generate_tilesets config/config_entire_plans.yaml
stdl-objdet make_detections config/config_entire_plans.yaml
python post_processing.py config/config_entire_plans.yaml
```

We strongly encourage the end-user to review the provided configuration files as well as the various output files, a list of which is printed by each script before exiting.