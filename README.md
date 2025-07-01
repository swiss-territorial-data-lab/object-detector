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
| 3 | Detection | `make_detections` | [here](scripts/make_detections.py) |
| 4 | Assessment | `assess_detections` | [here](scripts/assess_detections.py) |

These stages can be run one after the other, by issuing the following command from a terminal:

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

* **ground truth (GT)**: data used to train the deep learning-based detection model; such data is expected to be 100% true 

* **other data**: data that is not ground-truth-grade

* **labels**: geo-referenced polygons surrounding the objects targeted by a given analysis

* **FP labels**: geo-referenced polygons surrounding the false positive objects detected by a previously trained model. They are used to select tiles that will not be annotated (fp tiles) as they do not contain any object of interest, but are still included in the training dataset, to confront the model with potentially problematic images.

* **AoI**, abbreviation of "area of interest": geographical area over which the user intends to carry out the analysis. This area encompasses 
  * regions for which ground truth is available, as well as 
  * regions over which the user intends to detect potentially unknown objects

* **tiles**, or - more explicitly - "geographical map tiles": see [this link](https://wiki.openstreetmap.org/wiki/Tiles). More precisely, "Slippy Map Tiles" are used within this project, see [this link](https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames).

* **empty tiles**, tiles not intersecting ground truth (not annotated) added to the training dataset to provide contextual tiles and improve model performance. Empty tiles can be added to the dataset and distributed to the `trn`, `tst` and `val` dataset. Remaining tiles can either be deleted or included to the `oth` dataset. 

* **COCO data format**: see [this link](https://cocodataset.org/#format-data)

* **trn**, **val**, **tst**, **oth**: abbreviations of "training", "validation", "test" and "other", respectively

### Stage 1: tileset generation

This `generate_tilesets` command generates the various tilesets concerned by a given study. It is made of three steps:

1. Tile download;
2. Tile partition in the training, validation, test and other tilesets;
3. Generation of the COCO annotations for each dataset.

Each generated tileset is made up by:

* a collection of geo-referenced raster images (in GeoTIFF format)
* a JSON file compliant with the [COCO data format](https://cocodataset.org/#format-data)

The following relations apply: 

* <img src="https://latex.codecogs.com/png.latex?\fn_cm&space;\mbox{AoI&space;tiles}&space;=&space;(\mbox{GT&space;tiles})&space;\cup&space;(\mbox{oth&space;tiles})&space;\cup&space;(\mbox{FP&space;tiles})&space;\cup&space;(\mbox{empty&space;tiles})" title="\mbox{AoI tiles} = (\mbox{GT tiles}) \cup (\mbox{oth tiles}) \cup (\mbox{FP tiles}) \cup (\mbox{empty tiles})" />

* <img src="https://latex.codecogs.com/png.latex?\fn_cm&space;(\mbox{GT&space;tiles})\cup&space;(\mbox{FP&space;tiles})\cup&space;(\mbox{empty&space;tiles})&space;=&space;(\mbox{trn&space;tiles})&space;\cup&space;(\mbox{val&space;tiles})&space;\cup&space;(\mbox{tst&space;tiles})" title="\mbox{GT tiles} \cup (\mbox{FP_tiles}= (\mbox{trn tiles}) \cup (\mbox{val tiles}) \cup (\mbox{tst tiles})" />

where "GT tiles" are AoI tiles including GT labels and

<img src="https://latex.codecogs.com/png.latex?\fn_cm&space;A&space;\neq&space;B&space;\Rightarrow&space;A&space;\cap&space;B&space;=&space;\emptyset,&space;\quad&space;\forall&space;A,&space;B&space;\in&space;\{\mbox{trn&space;tiles},&space;\mbox{val&space;tiles},&space;\mbox{tst&space;tiles},&space;\mbox{oth&space;tiles}\}" title="A \neq B \Rightarrow A \cap B = \emptyset, \quad \forall A, B \in \{\mbox{trn tiles}, \mbox{val tiles}, \mbox{tst tiles}, \mbox{oth tiles}\}" />

In case no GT labels are provided by the user, the script will only generate `oth` tiles, covering the entire AoI.

When training the model, the user can choose to add empty tiles and/or empty tiles including FP detections to improve the model performance. Empty tiles can be manually defined or selected randomly within a given AoI.

In order to speed up some of the subsequent computations, each output image is accompanied by a small sidecar file in JSON format, carrying information about the image

* width and height in pixels;
* bounding box;
* spatial reference system.

Here's the excerpt of the configuration file relevant to this script, with values replaced by some documentation:

```yaml
generate_tilesets.py:
  debug_mode: 
    enable: <True or False; if True, only a small subset of tiles is processed>
    nb_tiles_max: <number of tiles to use if the debug mode is enabled>
  working_directory: <the script will use this folder as working directory, all paths are relative to this directory>
  output_folder: <the folder were output files will be written>
  datasets:
    aoi_tiles: <the path to the file including the delineation of the Slippy Mappy Tiles covering the AoI>
    ground_truth_labels: <the path to the file including ground-truth labels (optional, defaults to None)>
    other_labels: <the path to the file including other (non ground-truth) labels (optional, defaults to None)>
    add_fp_labels: <group must be delete if not needed>
      fp_labels: <the path to the file including false positive detections from a previous model>
      frac_trn: <fraction of FP tiles to add to the trn dataset, the remaining tiles will be split in 2 and added to the tst and val datasets (optional, defaults to 0.7)>
    image_source:
      type: <"WMS" as Web Map Service or "MIL" as ESRI's Map Image Layer or "XYZ" for xyz link or "FOLDER" for tiles from an existing folder>
      location: <the URL of the web service or the path to the initial folder>
      layers: <only applies to WMS endpoints>
      year: <"multi-year" if a 'year' attribute is provided in tiles.geojson or a numeric year else (optional, defaults to None). Use only with "XYZ" and "FOLDER" connectors>
      srs: <e.g. "EPSG:3857">
  empty_tiles: <group must be deleted if not needed>
    tiles_frac: <fraction (relative to the number of tiles intersecting labels) of empty tiles to add (optional, defaults to 0.5)>
    frac_trn: <fraction of empty tiles to add to the trn dataset, the remaining tiles will be split in 2 and added to the tst and val datasets (optional, defaults to 0.7)>
    keep_oth_tiles: <True or False, if True keep tiles in oth dataset not intersecting oth labels (optional, defaults to True)>  
  tile_size: <the tile/image width and height in pixels, necessary with WMS and MIL sources, otherwise None by default>
  overwrite: <True or False; if True, the script is allowed to overwrite already existing images>
  n_jobs: <the no. of parallel jobs the script is allowed to launch, e.g. 1>
  COCO_metadata: <group can be deleted to only perform tile download and split>
    year: <see https://cocodataset.org/#format-data>
    version: <see https://cocodataset.org/#format-data>
    description: <see https://cocodataset.org/#format-data>
    contributor: <see https://cocodataset.org/#format-data>
    url: <see https://cocodataset.org/#format-data>
    license:
      name: <see https://cocodataset.org/#format-data>
      url: <see https://cocodataset.org/#format-data>
    category:     # Only for the mono-class case, otherwise classes are read in the category file or deducted from labels.
        name: <the name of the category target objects belong to, e.g. "swimming pool">
        supercategory: <the supercategory target objects belong to, e.g. "facility">
    category_file: <file output by the script based on the labels and used to pass the classes in inference mode>
```

Note that: 

* the `ground_truth_labels`, `FP_labels` and `other_labels` datasets are optional. The user should either delete or comment out the concerned YAML keys in case she/he does not intend to provide these datasets. This feature has been developed in order to support, e.g., **inference-only scenarios**
* Except for the XYZ connector which requires EPSG:3857, the framework is agnostic with respect to the tiling scheme, which the user has to provide as a input file, compliant with the following requirements:

  1. a field named `id` must exist;
  2. the `id` field must not contain any duplicate value;
  3. values of the `id` field must follow the following pattern: `(<integer 1>, <integer 2>, <integer 3>)`, e.g. `(135571, 92877, 18)` or if a 'year' field is specified from the data preparation `(<integer 1>, <integer 2>, <integer 3>, <integer 4>)`,  e.g. `(2020, 135571, 92877, 18)`

### Stage 2: model training

> **Note**
This stage can be skipped if the user wishes to perform inference only, using a pre-trained model.

The `train_model` command allows one to train a detection model based on a convolutional deep neural network, leveraging [Meta's detectron2](https://github.com/facebookresearch/detectron2). For further information, we refer the user to the [official documentation](https://detectron2.readthedocs.io/en/latest/).

Here's the excerpt of the configuration file relevant to this script, with values replaced by textual documentation:

```yaml
train_model.py:
  debug_mode: <True or False; if True, a short training will be performed without taking the configuration for detectron2 into account.>
  working_directory: <the script will use this folder as working directory, all paths are relative to this directory>
  log_subfolder: <the subfolder of the working folder where we allow detectron2 writing some logs>
  sample_tagged_img_subfolder: <the subfolder where some sample images will be output>
  COCO_files:
    trn: <the COCO JSON file related to the training dataset>
    val: <the COCO JSON file related to the validation dataset>
    tst: <the COCO JSON file related to the test dataset>
  detectron2_config_file: <the detectron2 configuration file (relative path w/ respect to the working_folder>
  model_weights:
    model_zoo_checkpoint_url: <e.g. "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml">
```

Detectron2's configuration files are provided in the example folders mentioned here-below. We warn the end-user about the fact that, **for the time being, no hyperparameters tuning is automatically performed**.

The evolution of the loss function over the training and validation dataset can be observed in a local server with the following command:

```bash
$ tensorboard --logdir <path to the logs folder>
```

### Stage 3: detection

The `make_detections` command allows one to use the object detection model trained at the previous step to make detections over various input datasets:

* detections over the `trn`, `val`, `tst` datasets can be used to assess the reliability of this approach on ground truth data;
* detections over the `oth` dataset are, in principle, the main goal of this kind of analyses.

Here's the excerpt of the configuration file relevant to this script, with values replaced by textual documentation:

```yaml
make_detections.py:
  working_directory: <the script will use this folder as working directory, all paths are relative to this directory>
  log_subfolder: <the subfolder of the working folder where we allow detectron2 writing some logs (optional)>
  sample_tagged_img_subfolder: <the subfolder where some sample images will be output (optional)>
  COCO_files:
    trn: <the COCO JSON file related to the training dataset (optional)>
    val: <the COCO JSON file related to the validation dataset (optional)>
    tst: <the COCO JSON file related to the test dataset (optional)>
    oth: <the COCO JSON file related to the "other" dataset (optional)>
  detectron2_config_file: <the detectron2 configuration file>
  model_weights:
    pth_file: <e.g. "./logs/model_final.pth">
  image_metadata_json: <the path to the file with the image metadata JSON, generated by the `generate_tilesets` command>
  # the following section concerns the Ramer-Douglas-Peucker algorithm, which can be optionally applied to detections before they are exported
  rdp_simplification: 
    enabled: <True/False>
    epsilon: <see https://rdp.readthedocs.io/en/latest/>
  score_lower_threshold: <choose a value between 0 and 1, e.g. 0.05 - detections with a score less than this threshold would be discarded>
  remove_overlap: <True/False, whether to remove the detection with a lower threshold in case of Jaccard index higher than 0.5 between two detections (optional, defaults to False)>
```

### Stage 4: assessment

The `assess_detections` command allows one to assess the reliability of detections, comparing detections with ground-truth data. The assessment goes through the following steps:

1. Label (GT + `oth`) geometries are clipped to the boundaries of the various AoI tiles, scaled by a factor 0.999 in order to prevent any "crosstalk" between neighboring tiles.

2. Spatial joins and intersection over union are computed between the detections and the clipped labels, in order to identify
    * True positives (TP), *i.e.* objects that are found in both datasets, labels and detections;
    * False positives (FP), *i.e.* objects that are only found in the detection dataset;
    * False negatives (FN), *i.e.* objects that are only found in the label dataset;
    * Wrong class, *i.e.* objects that are found in both datasets, but with different classes.
If the detection is performed over several years, the spatial comparison is made between labels and detections in the same year.

4. Finally, TP, FP and FN are counted in order to compute the following metrics (see [this page](https://en.wikipedia.org/wiki/Precision_and_recall)) :
    * precision
    * recall
    * f1-score

Here's the excerpt of the configuration file relevant to this command, with values replaced by textual documentation:
```yaml
assess_detections.py:
  working_directory: <the script will use this folder as working directory, all paths are relative to this directory>
  output_folder: <the folder where we allow this command to write output files>
  datasets:
    ground_truth_labels: <the path to GT labels in format (optional, defaults to None)>
    other_labels: <the path to "other labels" in format (optional, defaults to None)>
    split_aoi_tiles: <the path to the file including the partition (trn, val, tst, out) of the AoI tiles>
    detections:
      trn: <the path to the Pickle file including detections over the trn dataset (optional, defaults to None)>
      val: <the path to the Pickle file including detections over the val dataset, used to determine the threshold on the confidence score (optional, defaults to None)>
      tst: <the path to the Pickle file including detections over the tst dataset (optional, defaults to None)>
      oth: <the path to the Pickle file including detections over the oth dataset (optional, defaults to None)>
  confidence_threshold: <threshold on the confidence score when there is no validation dataset (optional, defaults to None)>
  area_threshold: <area under which the polygons are excluded from assessment and returned in a specific dataframe, ignored if None (optional, defaults to None)>
  iou_threshold: <minimum overlap for two objects to be considered a match (optional, defaults to 0.25)>
  metrics_method: <method to pass from by-class to global metrics, choice is macro-average, macro-weighted-average, or micro-average (optional, defaults to macro-average)>
```

## Examples

A few examples are provided within the `examples` folder. For further details, we refer the user to the various use-case specific readme files:

* [Delimitation of Anthropogenic Activities on Natural Soil over Time](examples/anthropogenic-activities/README.md): multi-class instance segmentation including empty and false positive tiles in the training phase and with images from an XYZ service,
* [Segmentation of Border Points based on Analog Cadastral Plans](examples/borderpoints/README.md): multi-class instance segmentation with images from another folder based on a custom grid,
* [Evolution of Mineral Extraction Sites over the Entire Switzerland](examples/mineral-extrac-sites-detection/README.md): object monitoring with images from an XYZ service,
* [Swimming Pool Detection over the Canton of Geneva](examples/swimming-pool-detection/GE/README.md): instance segmentation with images from a MIL service,
* [Swimming Pool Detection over the Canton of Neuchâtel](examples/swimming-pool-detection/NE/README.md): instance segmentation with images from a WMS service.service,

It is brought to the reader attention that the examples are provided with a debug parameter that can be set to `True` for quick tests.

## License

The STDL Object Detector is released under the [MIT license](LICENSE.md).
