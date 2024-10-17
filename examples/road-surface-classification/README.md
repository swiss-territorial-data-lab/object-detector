# Example: determination of the type or road surface

A sample working setup is provided here to allow, allowing the end-user to detect roads and determine their type of surface (natural or artificial). This example is illustrating the functioning for the multi-class case. <br>
It is made of the following assets:

- the read-to-use configuration files
    - `config_rs.yaml`
    - `detectron2_config_3bands.yaml`,
- the initial data in the `data` subfolder:
    - the roads and forests from the product swissTLM3D
    - the delimitation of the AOI
    - an excel file with the road parameters,
- a data preparation script (`prepare_data.py`) producing the files to be used as input to the `generate_tilesets.py`script.

Installation can be carried out by following the instructions in the main readme file. When using docker, the container must be launched before running the workflow:

```bash
$ sudo chown -R 65534:65534 examples
$ docker compose run --rm -it stdl-objdet
```

The end-to-end workflow can be run by issuing the following list of commands:

```bash
$ cd examples/road-surface-classification
$ python prepare_data.py config_rs.yaml
$ stdl-objdet generate_tilesets config_rs.yaml
$ stdl-objdet train_model config_rs.yaml
$ stdl-objdet make_detections config_rs.yaml
$ stdl-objdet assess_detections config_rs.yaml
```

The docker container is existed and the permission restored with.

 ```bash
$ exit
$ sudo chmod -r a+w examples
```

This example is made up from a subset of the data used in the proj-roadsurf project. For more information about this project, you can consult [the associated repository](https://github.com/swiss-territorial-data-lab/proj-roadsurf) and [its full documentation](https://tech.stdl.ch/PROJ-ROADSURF/). <br>
The original project does not use the original script for assessment but has his own.

We strongly encourage the end-user to review the provided `config_rs.yaml` file as well as the various output files, a list of which is printed by each script before exiting.