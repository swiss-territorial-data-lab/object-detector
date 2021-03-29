
# Example: detecting swimming pools over the Canton of Neuchâtel

A sample working setup is here provided, allowing the end-user to detect swimming pools over the Canton of Neuchâtel. It is made up by the following assets:

* ready-to-use configuration files, namely `config_NE.yaml` and `detectron2_config_NE.yaml`;
* supplementary data (`data/*`), *i.e.* 
    * geographical sectors covering ground-truth data;
    * other (non ground-truth) sectors;
    * ground-truth labels;
    * other labels.
* A data preparation script (`prepare_data.py`), producing files to be used as input to the `generate_training_sets.py` script.

The end-to-end workflow can be run by issuing the following list of commands, straight from this folder:

```bash
$ conda activate <the name of the previously created setup virtual environment>
$ python prepare_data.py config_NE.yaml
$ cd output_NE_v2 
$ cat aoi.geojson | supermercado burn 18 | mercantile shapes | fio collect > aoi_z18_tiles.geojson
$ cd -
$ python prepare_data.py config_NE.yaml
$ python ../../../scripts/generate_training_sets.py config_NE.yaml
$ python ../../../scripts/make_predictions.py config_NE.yaml
$ python ../../../scripts/assess_predictions.py config_NE.yaml
```

We strongly encourage the end-user to review the provided `config_GE.yaml` file as well as the various output files, a list of which is printed by each script before exiting. 