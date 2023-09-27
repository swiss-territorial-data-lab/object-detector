# Object Detector

This project provides a suite of Python scripts allowing the end-user to use Deep Learning to detect objects in geo-referenced raster images.

### Table of contents

- [Requirements](#requirements)
    - [Hardware](#hardware)
    - [Software](#software)
- [Installation](#installation)
- [How-to](#how-to)
- [Examples](#examples)
- [License](#license)

## Requirements

### Hardware

A CUDA-enabled GPU is required.

### Software

* CUDA driver. This code was developed and tested with CUDA 11.3 on Ubuntu 20.04.

* Although we recommend the usage of [Docker](https://www.docker.com/) (see [here](#with-docker)), this code can also be run without Docker, provided that Python 3.8 is available. Python dependencies may be installed with either `pip` or `conda`, using the provided `requirements.txt` file. We advise using a [Python virtual environment](https://docs.python.org/3/library/venv.html).

## Installation

### Without Docker

The object detector can be installed by issuing the following command (see [this page](https://setuptools.pypa.io/en/latest/userguide/development_mode.html) for more information on the "editable installs"):

```bash
$ pip install --editable .
```

In case of a successful installation, the command 

```bash
$ stdl-objdet -h
```

should display some basic usage information.

### With Docker

A Docker image can be built by issuing the following command:

```bash
$ docker compose build
```

In case of a successful build, the command 

```bash
$ docker compose run --rm stdl-objdet stdl-objdet -h
```

should display some basic usage information. Note that, for the code to run properly,

1. the version of the CUDA driver installed on the host machine must match with the version used in the [Dockerfile](Dockerfile), namely version 11.3. We let end-user adapt the Dockerfile to her/his environment.
2. The NVIDIA Container Toolkit must be installed on the host machine (see [this guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).

## How-to

### 

This project implements the workflow described [here](https://tech.stdl.ch/TASK-IDET/#workflow), which includes four stages:

| Stage no. | Stage name | CLI command | Implementation | 
| :-: | --- | --- | --- |
| 1 | Tileset generation | `generate_tilesets` | [here](scripts/generate_tilesets.py) |
| 2 | Model training | `train_model` | [here](scripts/train_model.py) |
| 3 | Detection | `make_predictions` | [here](scripts/make_predictions.py) |
| 4 | Assessment | `assess_predictions` | [here](scripts/assess_predictions.py) |

These stages/scripts can be run one after the other, by issuing the following command from a terminal:

* w/o Docker: 

  ```bash
  $ stdl-objdet <CLI command> <configuration_file>
  ```

* w/ Docker: 

  ```bash
  $ docker compose run --rm -it stdl-objdet stdl-objdet <CLI command> <configuration_file>

  ```

  Alternatively,
  
  ```bash
  $ docker compose run --rm -it stdl-objdet
  ```

  then 

  ```
  nobody@<container ID>:/app# stdl-objdet <CLI command> <configuration_file>
  ```

  For those who are less familiar with Docker, know that all output files created inside a container are not persistent, unless "volumes" or "bind mounts" are used (see [this](https://docs.docker.com/storage/)).

The same configuration file can be used for all the commands, as each of them only reads the content related to a key named after its name. More detailed information about each stage and the related configuration is provided here-below. The following terminology is used:

* **ground-truth data**: data to be used to train the Deep Learning-based detection model; such data is expected to be 100% true 

* **GT**: abbreviation of ground-truth

* **other data**: data that is not ground-truth-grade 

* **labels**: geo-referenced polygons surrounding the objects targeted by a given analysis

* **AoI**, abbreviation of "Area of Interest": geographical area over which the user intend to carry out the analysis. This area encompasses 
  * regions for which ground-truth data is available, as well as 
  * regions over which the user intends to detect potentially unknown objects

* **tiles**, or - more explicitly - "geographical map tiles": see [this link](https://wiki.openstreetmap.org/wiki/Tiles). More precisely, "Slippy Map Tiles" are used within this project, see [this link](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames).

* **COCO data format**: see [this link](https://cocodataset.org/#format-data)

* **trn**, **val**, **tst**, **oth**: abbreviations of "training", "validation", "test" and "other", respectively

### Stage 1: tileset generation

This `generate_tilesets` command generates the various tilesets concerned by a given study. Each generated tileset is made up by:

* a collection of geo-referenced raster images (in GeoTIFF format)
* a JSON file compliant with the [COCO data format](https://cocodataset.org/#format-data)

The following relations apply: 

* <img src="https://latex.codecogs.com/png.latex?\fn_cm&space;\mbox{AoI&space;tiles}&space;=&space;(\mbox{GT&space;tiles})&space;\cup&space;(\mbox{oth&space;tiles})" title="\mbox{AoI tiles} = (\mbox{GT tiles}) \cup (\mbox{oth tiles})" />

* <img src="https://latex.codecogs.com/png.latex?\fn_cm&space;\mbox{GT&space;tiles}&space;=&space;(\mbox{trn&space;tiles})&space;\cup&space;(\mbox{val&space;tiles})&space;\cup&space;(\mbox{tst&space;tiles})" title="\mbox{GT tiles} = (\mbox{trn tiles}) \cup (\mbox{val tiles}) \cup (\mbox{tst tiles})" />

where "GT tiles" are AoI tiles including GT labels and

<img src="https://latex.codecogs.com/png.latex?\fn_cm&space;A&space;\neq&space;B&space;\Rightarrow&space;A&space;\cap&space;B&space;=&space;\emptyset,&space;\quad&space;\forall&space;A,&space;B&space;\in&space;\{\mbox{trn&space;tiles},&space;\mbox{val&space;tiles},&space;\mbox{tst&space;tiles},&space;\mbox{oth&space;tiles}\}" title="A \neq B \Rightarrow A \cap B = \emptyset, \quad \forall A, B \in \{\mbox{trn tiles}, \mbox{val tiles}, \mbox{tst tiles}, \mbox{oth tiles}\}" />

In case no GT labels are provided by the user, the script will only generate `oth` tiles, covering the entire AoI.

In order to speed up some of the subsequent computations, each output image is accompanied by a small sidecar file in JSON format, carrying information about the image

* width and height in pixels;
* bounding box;
* spatial reference system.

Here's the excerpt of the configuration file relevant to this script, with values replaced by some documentation:

```yaml
generate_tilesets.py:
  debug_mode: <True or False (without quotes); if True, only a small subset of tiles is processed>
  datasets:
    aoi_tiles_geojson: <the path to the GeoJSON file including polygons of the Slippy Mappy Tiles covering the AoI>
    ground_truth_labels_geojson: <the path to the GeoJSON file including ground-truth labels (optional)>
    other_labels_geojson: <the path to the GeoJSON file including other (non ground-truth) labels (optional)>
    orthophotos_web_service:
      type: <"WMS" as Web Map Service or "MIL" as ESRI's Map Image Layer>
      url: <the URL of the web service>
      layers: <only applies to WMS endpoints>
      srs: <e.g. "EPSG:3857">
  output_folder: <the folder were output files will be written>
  tile_size: <the tile/image width and height, in pixels>
  overwrite: <True or False (without quotes); if True, the script is allowed to overwrite already existing images>
  n_jobs: <the no. of parallel jobs the script is allowed to launch, e.g. 1>
  COCO_metadata:
    year: <see https://cocodataset.org/#format-data>
    version: <see https://cocodataset.org/#format-data>
    description: <see https://cocodataset.org/#format-data>
    contributor: <see https://cocodataset.org/#format-data>
    url: <see https://cocodataset.org/#format-data>
    license:
      name: <see https://cocodataset.org/#format-data>
      url: <see https://cocodataset.org/#format-data>
    category:
        name: <the name of the category target objects belong to, e.g. "swimming pool">
        supercategory: <the supercategory target objects belong to, e.g. "facility">
```

Note that: 

* the `ground_truth_labels_geojson` and `other_labels_geojson` datasets are optional. The user should either delete or comment out the concerned YAML keys in case she/he does not intend to provide these datasets. This feature has been developed in order to support, e.g., **inference-only scenarios**.
* The framework is agnostic with respect to the tiling scheme, which the user has to provide as a GeoJSON input file, compliant with the following requirements:

  1. a field named `id` must exist;
  2. the `id` field must not contain any duplicate value;
  3. values of the `id` field must follow the following pattern: `(<integer 1>, <integer 2>, <integer 3>)`, e.g. `(135571, 92877, 18)`.

### Stage 2: model training

> **Note**
This stage can be skipped if the user wishes to perform inference only, using a pre-trained model.

The `train_model` command allows one to train a detection model based on a Convolutional Deep Neural Network, leveraging [FAIR's Detectron2](https://github.com/facebookresearch/detectron2). For further information, we refer the user to the [official documention](https://detectron2.readthedocs.io/en/latest/).

Here's the excerpt of the configuration file relevant to this script, with values replaced by textual documentation:

```yaml
train_model.py:
  working_folder: <the script will chdir into this folder>
  log_subfolder: <the subfolder of the working folder where we allow Detectron2 writing some logs>
  sample_tagged_img_subfolder: <the subfolder where some sample images will be output>
  COCO_files: # relative paths, w/ respect to the working_folder
    trn: <the COCO JSON file related to the training dataset (mandatory)>
    val: <the COCO JSON file related to the validation dataset (mandatory)>
    tst: <the COCO JSON file related to the test dataset (mandatory)>
  detectron2_config_file: <the Detectron2 configuration file (relative path w/ respect to the working_folder>
  model_weights:
    model_zoo_checkpoint_url: <e.g. "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml">
```

Detectron2 configuration files are provided in the example folders mentioned here-below. We warn the end-user about the fact that, **for the time being, no hyperparameters tuning is automatically performed**.

The evolution of the loss function over the training and validation dataset can be observed in a local server with the following command:

```bash
$ tensorboard --logdir <path to the logs folder>
```

### Stage 3: detection

The `make_predictions` command allows one to use the object detection model trained at the previous step to make detections over various input datasets:

* detections over the `trn`, `val`, `tst` datasets can be used to assess the reliability of this approach on ground-truth data;
* detections over the `oth` dataset are, in principle, the main goal of this kind of analyses.

Here's the excerpt of the configuration file relevant to this script, with values replaced by textual documentation:

```yaml
make_predictions.py:
  working_folder: <the script will chdir into this folder>
  log_subfolder: <the subfolder of the working folder where we allow Detectron2 writing some logs>
  sample_tagged_img_subfolder: <the subfolder where some sample images will be output>
  COCO_files: # relative paths, w/ respect to the working_folder
    trn: <the COCO JSON file related to the training dataset (optional)>
    val: <the COCO JSON file related to the validation dataset (optional)>
    tst: <the COCO JSON file related to the test dataset (optional)>
    oth: <the COCO JSON file related to the "other" dataset (optional)>
  detectron2_config_file: <the Detectron2 configuration file (relative path w/ respect to the working_folder>
  model_weights:
    pth_file: <e.g. "./logs/model_final.pth">
  image_metadata_json: <the path to the image metadata JSON file, generated by the `generate_tilesets` command>
  # the following section concerns the Ramer-Douglas-Peucker algorithm, which can be optionally applied to detections before they are exported
  rdp_simplification: 
    enabled: <true/false>
    epsilon: <see https://rdp.readthedocs.io/en/latest/>
  score_lower_threshold: <choose a value between 0 and 1, e.g. 0.05 - detections with a score less than this threshold would be discarded>
```

### Stage 4: assessment

The `assess_predictions` command allows one to assess the reliability of detections, comparing detections with ground-truth data. The assessment goes through the following steps:

1. Labels (GT + `oth`) geometries are clipped to the boundaries of the various AoI tiles, scaled by a factor 0.999 in order to prevent any "crosstalk" between neighboring tiles.

2. Vector features are extracted from Detectron2's detections, which are originally in a raster format (`numpy` arrays, to be more precise).

3. Spatial joins are computed between the vectorized detections and the clipped labels, in order to identify
    * True Positives (TP), *i.e.* objects that are found in both datasets, labels and detections;
    * False Positives (FP), *i.e.* objects that are only found in the detections dataset;
    * False Negatives (FN), *i.e.* objects that are only found in the labels dataset.

4. Finally, TPs, FPs and FNs are counted in order to compute the following metrics (see [this page](https://en.wikipedia.org/wiki/Precision_and_recall)) :
    * precision
    * recall
    * f1-score

Here's the excerpt of the configuration file relevant to this command, with values replaced by textual documentation:
```yaml
assess_predictions.py:
  datasets:
    ground_truth_labels_geojson: <the path to GT labels in GeoJSON format>
    other_labels_geojson: <the path to "other labels" in GeoJSON format>
    split_aoi_tiles_geojson: <the path to the GeoJSON file including split (trn, val, tst, out) AoI tiles>
    detections:
      trn: <the path to the Pickle file including detections over the trn dataset (optional)>
      val: <the path to the Pickle file including detections over the val dataset (mandatory)>
      tst: <the path to the Pickle file including detections over the tst dataset (optional)>
      oth: <the path to the Pickle file including detections over the oth dataset (optional)>
  output_folder: <the folder where we allow this command to write output files>
```

## Examples

A few examples are provided within the `examples` folder. For further details, we refer the user to the various use-case specific readme files:

* [Swimming Pool Detection over the Canton of Geneva](examples/swimming-pool-detection/GE/README.md)
* [Swimming Pool Detection over the Canton of Neuchâtel](examples/swimming-pool-detection/NE/README.md)
* [Quarry Detection over the entire Switzerland](examples/quarry-detection/README.md)
* [Determination of type of road surface in the Emmental](examples/road-surface-detection/multi-class/readme.md)

## License

The STDL Object Detector is released under the [MIT license](LICENSE.md).