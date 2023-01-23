# Example: determination of the type of road with 4 bands images

A sample working setup is here provided, allowing the end-user to determine the type of road surface making use of images with 4 bands (R, G, B, NIR). It is made up by the following assets:

- the ready-to-use configuration files:
	- `config_nir.yaml` and `detectron2_config_4bands.yaml`,
- the initial data:
	- the roads and the forests from the product swissTLM3D,
	- the area of interest,
	- the parameters for the development of the roads,
- a data preparation script (`prepare_data.py`), producing files to be used as input to the `generate_tilesets.py` script.

If not already installed, you will need to first have the right version fo python3-gdal.

```bash
sudo apt-get install -y python3-gdal gdal-bin libgdal-dev gcc g++ python3.8-dev
```

After the creation and activation of an environment, the end-to-end workflow can be run by issuing the following list of commands, straight from this folder:

```bash
$ pip -r ../../../requirements.txt
$ python prepare_data.py config_nir.yaml
$ python ../../../scripts/generate_tilesets.py config_nir.yaml
$ python ../../../scripts/train_model.py config_nir.yaml
$ python ../../../scripts/make_predictions.py config_nir.yaml
$ python ../../../scripts/assess_predictions.py config_nir.yaml
```

We strongly encourage the end-user to review the provided `config_nir.yaml` file as well as the various output files, a list of which is printed by each script before exiting.

For more information about the determination of road surface, you can consult [the associated respository](https://github.com/swiss-territorial-data-lab/proj-roadsurf).