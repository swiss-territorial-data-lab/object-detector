
# Example: detecting swimming pools over the Canton of Geneva

A sample working setup is here provided, allowing the end-user to detect swimming pools over the Canton of Geneva. It is made up by the following assets:

* ready-to-use configuration files, namely `config_GE.yaml` and `detectron2_config_GE.yaml`;
* supplementary data (`data/OK_z18_tile_IDs.csv`), *i.e.* a curated list of Slippy Map Tiles corresponding to zoom level 18, which seemed to include reliable "ground-truth data" when they were manually checked against [SITG's "Piscines" Open Dataset](https://ge.ch/sitg/fiche/1836), in Summer 2020. The thoughtful user should either review or regenerate this file in order to get better results.
* A data preparation script (`prepare_data.py`), producing files to be used as input to the `generate_training_sets.py` script.

The end-to-end workflow can be run by issuing the following list of commands, straight from this folder:

```bash
$ conda activate <the name of the previously created setup virtual environment>
$ python prepare_data.py config_GE.yaml
$ cd output_GE_v2 
$ cat parcels.geojson | supermercado burn 18 | mercantile shapes | fio collect > parcels_z18_tiles.geojson
$ cd -
$ python prepare_data.py config_GE.yaml
$ python ../../../scripts/generate_training_sets.py config_GE.yaml
$ python ../../../scripts/make_predictions.py config_GE.yaml
$ python ../../../scripts/assess_predictions.py config_GE.yaml
```

We strongly encourage the end-user to review the provided `config_GE.yaml` file as well as the various output files, a list of which is printed by each script before exiting. 