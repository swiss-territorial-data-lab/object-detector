# Example: detecting swimming pools over the Canton of Geneva

A sample working setup is here provided, allowing the end-user to detect swimming pools over the Canton of Geneva. It is made up by the following assets:

* ready-to-use configuration files, namely `config_GE.yaml` and `detectron2_config_GE.yaml`.
* Supplementary data (`data/OK_z18_tile_IDs.csv`), *i.e.* a curated list of Slippy Map Tiles corresponding to zoom level 18, which seemed to include reliable "ground-truth data" when they were manually checked against the [SITG's "Piscines" Open Dataset](https://ge.ch/sitg/fiche/1836), in Summer 2020. The thoughtful user should either review or regenerate this file in order to get better results.
* A data preparation script (`prepare_data.py`), producing files to be used as input to the `generate_tilesets` stage.

The workflow can be run end-to-end by issuing the following list of commands, from the root folder of this GitHub repository:

```
$ sudo chown -R 65534:65534 examples
$ docker compose run --rm -it stdl-objdet
nobody@<id>:/app# cd examples/swimming-pool-detection/GE
nobody@<id>:/app# python prepare_data.py config_GE.yaml
nobody@<id>:/app# cd output_GE && cat parcels.geojson | supermercado burn 18 | mercantile shapes | fio collect > parcels_z18_tiles.geojson && cd -
nobody@<id>:/app# python prepare_data.py config_GE.yaml
nobody@<id>:/app# stdl-objdet generate_tilesets config_GE.yaml
nobody@<id>:/app# stdl-objdet train_model config_GE.yaml
nobody@<id>:/app# stdl-objdet make_detections config_GE.yaml
nobody@<id>:/app# stdl-objdet assess_detections config_GE.yaml
nobody@<id>:/app# exit
$ sudo chmod -R a+w examples
```

We strongly encourage the end-user to review the provided `config_GE.yaml` file as well as the various output files, a list of which is printed by each script before exiting.

Due to timeout of the WMS service, the user might have to run the tileset generation several times.
