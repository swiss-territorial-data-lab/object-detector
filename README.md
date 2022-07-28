# Object Detector

This project provides a suite of Python scripts allowing the end-user to use Deep Learning to detect objects in georeferenced raster images.

## Hardware requirements

A CUDA-capable system is required.

## Software Requirements

* Python 3.8

* Dependencies may be installed with either `pip` or `conda`, by making use of the provided `requirements.txt` file. The following method was tested successfully on a Linux system powered by CUDA 10.1: 

    ```bash
    $ conda create -n <the name of the virtual env> -c conda-forge python=3.8 gdal=3.2.1
    $ conda activate <the name of the virtual env>
    $ pip install -r requirements.txt
    ```

## Scripts

Four scripts can be found in the `scripts` subfolder: 

1. `generate_tilesets.py`
2. `train_model.py`
3. `make_predictions.py`
4. `assess_predictions.py`

which can be run one after the other following this very order, by issuing the following command from a terminal:

```bash
$ python <the_path_to_the_script>/<the_script.py> <the_configuration_file>
``` 

Note concerning **inference-only scenarios**: the execution of the `train_model.py` script can be skipped in case the user wishes to only perform inference, using a model trained in advance.

The same configuration file can be used for all the scripts, as each script only reads the content related to a key named after itself - further details on the configuration file will be provided here-below. Before terminating, each script prints the list of output files: we strongly encourage the end-user to review those files, *e.g.* by loading them into [QGIS](https://qgis.org).

The following terminology will be used throughout the rest of this document:

* **ground-truth data**: data to be used to train the Deep Learning-based predictive model; such data is expected to be 100% true 

* **GT**: abbreviation of ground-truth

* **other data**: data that is not ground-truth-grade 

* **labels**: georeferenced polygons surrounding the objects targeted by a given analysis

* **AoI**, abbreviation of "Area of Interest": geographical area over which the user intend to carry out the analysis. This area encompasses 
  * regions for which ground-truth data is available, as well as 
  * regions over which the user intends to detect potentially unknown objects

* **tiles**, or - more explicitly - "geographical map tiles": cf. [this link](https://wiki.openstreetmap.org/wiki/Tiles). More precisely, "Slippy Map Tiles" are used within this project, cf. [this link](https://developers.planet.com/tutorials/slippy-maps-101/).

* **COCO data format**: cf. [this link](https://cocodataset.org/#format-data)

* **trn**, **val**, **tst**, **oth**: abbreviations of "training", "validation", "test" and "other", respectively

### 1. `generate_tilesets.py`

This script generates the various tilesets concerned by a given study. Each generated tileset is made up by:

* a collection of georeferenced raster images (in GeoTIFF format)
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

The script can be run by issuing the following command from a terminal:

```bash
$ python <the path>/generate_tilesets.py <the configuration file (YAML format)>
```

Here's the excerpt of the configuration file relevant to this script, with values replaced by textual documentation:

```yaml
generate_tilesets.py:
  debug_mode: <True or False (without quotes); if True, only a small subset of tiles is processed>
  datasets:
    aoi_tiles_geojson: <the path to the GeoJSON file including polygons à la Slippy Mappy Tiles covering the AoI>
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
    year: <cf. https://cocodataset.org/#format-data>
    version: <cf. https://cocodataset.org/#format-data>
    description: <cf. https://cocodataset.org/#format-data>
    contributor: <cf. https://cocodataset.org/#format-data>
    url: <cf. https://cocodataset.org/#format-data>
    license:
      name: <cf. https://cocodataset.org/#format-data>
      url: <cf. https://cocodataset.org/#format-data>
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

### 2. `train_model.py`

This script allows one to train a predictive model based on a Convolutional Deep Neural Network, leveraging [FAIR's Detectron2](https://github.com/facebookresearch/detectron2). For further information, we refer the user to the [official documention](https://detectron2.readthedocs.io/en/latest/).

The script can be run by issuing the following command from a terminal:

```bash
$ python <the path>/train_model.py <the configuration file (YAML format)>
```

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

Detectron2 configuration files are provided in the example folders mentioned here-below. We warn the end-user about the fact that, **for the time being, no hyperparameters tuning is automatically performed by this suite of scripts**.

### 3. `make_predictions.py`

This script allows to use the predictive model trained at the previous step to make predictions over various input datasets:

* predictions over the `trn`, `val`, `tst` datasets can be used to assess the reliability of this approach on ground-truth data;
* predictions over the `oth` dataset are, in principle, the main goal of this kind of analyses.

The script can be run by issuing the following command from a terminal:

```bash
$ python <the path>/make_predictions.py <the configuration file (YAML format)>
```

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
```

### 4. `assess_predictions.py`

This script allows one to assess the reliability of predictions made by the previous script, comparing predictions with ground-truth data. The assessment goes through the following steps:

1. Labels (GT + `oth`) geometries are clipped to the boundaries of the various AoI tiles, scaled by a factor 0.999 in order to prevent any "crosstalk" between neighbouring tiles.

2. Vector features are extracted from Detectron2's predictions, which are originally in a raster format (`numpy` arrays, to be more precise).

3. Spatial joins are computed between the vectorized predictions and the clipped labels, in order to identify
    * True Positives (TP), *i.e.* objects that are found in both datasets, labels and predictions;
    * False Positives (FP), *i.e.* objects that are only found in the predictions dataset;
    * False Negatives (FN), *i.e.* objects that are only found in the labels dataset.

4. Finally, TPs, FPs and FNs are counted in order to compute the following metrics (cf. [this page](https://en.wikipedia.org/wiki/Precision_and_recall)) :
    * precision
    * recall
    * f1-score

The script can be run by issuing the following command from a terminal:

```bash
$ python <the path>/assess_predictions.py <the configuration file (YAML format)>
```

Here's the excerpt of the configuration file relevant to this script, with values replaced by textual documentation:
```yaml
assess_predictions.py:
  n_jobs: <the no. of parallel jobs the script is allowed to launch, e.g. 1>
  datasets:
    ground_truth_labels_geojson: <the path to GT labels in GeoJSON format>
    other_labels_geojson: <the path to "other labels" in GeoJSON format>
    image_metadata_json: <the path to the image metadata JSON file, saved by the previous script>
    split_aoi_tiles_geojson: <the path to the GeoJSON file including split (trn, val, tst, out) AoI tiles>
    predictions:
      trn: <the path to the Pickle file including predictions over the trn dataset (optional)>
      val: <the path to the Pickle file including predictions over the val dataset (mandatory)>
      tst: <the path to the Pickle file including predictions over the tst dataset (optional)>
      oth: <the path to the Pickle file including predictions over the oth dataset (optional)>
  output_folder: <the folder where we allow this script to write output files>
```

## Examples

A few examples are provided within the folder `examples`. For further details, we refer the user to the various use-case specific readme files:

* [Swimming Pool Detection over the Canton of Geneva](examples/swimming-pool-detection/GE/README.md)
* [Swimming Pool Detection over the Canton of Neuchâtel](examples/swimming-pool-detection/NE/README.md)
