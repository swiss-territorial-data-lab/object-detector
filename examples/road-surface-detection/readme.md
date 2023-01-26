# Example: determination of the type or road surface

A sample working setup is provided here to allow, allowing the end-user to detect roads and determine their type of surface (natural or artificial). This example is illustrating the functioning for the multi-class case. It can be done with RGB images or with 4 bands images including the near-infrared band.<br>
It is made of the following assets:

- the read-to-use configuration files
    - RGB case (`3-bands/`):
        - `config_rs.yaml`,
        - `detectron2_config_3bands.yaml`,
    - RGBNir case (`4-bands/`):
        - `config_nir.yaml`,
        - `detectron2_config_4bands.yaml`,
- the initial data in the `data` subfolder:
    - the roads and forests from the product swissTLM3D,
    - the delimitation of the AOI,
    - an excel file with the road parameters,
- a data preparation script (`prepare_data.py`) producing the files to be used as input to the `generate_tilesets.py`script.

After creating and a new environment in python 3.8, the end-to-end workflow can be run by issuing the following list of commands, straight from this folder. For example, in the RGB case:

```bash
$ sudo apt-get install -y python3-gdal gdal-bin libgdal-dev gcc g++ python3.8-dev
$ pip install -r ../../requirements.txt
$ python prepare_data.py 3-bands/config_rs.yaml
$ python ../../scripts/generate_tilesets.py 3-bands/config_rs.yaml
$ python ../../scripts/train_model.py 3-bands/config_rs.yaml
$ python ../../scripts/make_predictions.py 3-bands/config_rs.yaml
$ python ../../scripts/assess_predictions.py 3-bands/config_rs.yaml
```

In the current example, the ground truth is focused on the roads from the class "3m Strassen" based on the definition of the STDL project on the determination of road surface. For more information about this project, you can consult [the associated repository](https://github.com/swiss-territorial-data-lab/proj-roadsurf) (not public yet).

We strongly encourage the end-user to review the provided `config_rs.yaml` file as well as the various output files, a list of which is printed by each script before exiting.