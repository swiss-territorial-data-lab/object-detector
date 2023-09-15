# Example: detecting swimming pools over the Canton of Neuchâtel

A sample working setup is here provided, allowing the end-user to detect swimming pools over the Canton of Neuchâtel. It is made up by the following assets:

* ready-to-use configuration files, namely `config_NE.yaml` and `detectron2_config_NE.yaml`.
* Supplementary data (`data/*`), *i.e.* 
    * geographical sectors covering ground-truth data;
    * other (non ground-truth) sectors;
    * ground-truth labels;
    * other labels.
* A data preparation script (`prepare_data.py`), producing files to be used as input to the `generate_tilesets` stage.

The workflow can be run end-to-end by issuing the following list of commands, from the root folder of this GitHub repository:

```
$ sudo chown -R 65534:65534 examples
$ docker compose run --rm -it stdl-objdet
nobody@<id>:/app# cd examples/swimming-pool-detection/NE
nobody@<id>:/app# python prepare_data.py config_NE.yaml
nobody@<id>:/app# cd output_NE && cat parcels.geojson | supermercado burn 18 | mercantile shapes | fio collect > parcels_z18_tiles.geojson && cd -
nobody@<id>:/app# python prepare_data.py config_NE.yaml
nobody@<id>:/app# stdl-objdet generate_tilesets config_NE.yaml
nobody@<id>:/app# stdl-objdet train_model config_NE.yaml
nobody@<id>:/app# stdl-objdet make_detections config_NE.yaml
nobody@<id>:/app# stdl-objdet assess_detections config_NE.yaml
$ sudo chmod -R a+w examples
```

We strongly encourage the end-user to review the provided `config_NE.yaml` file as well as the various output files, a list of which is printed by each script before exiting. 
