# Example: segmentation of the border points based on the analog cadastral plans

A working setup is provided here to test the multi-class classification and the use of empty tiles, as well as false positive ones.
It consists of the following elements:

* ready-to-use configuration files
* input data
* scripts for data preparation and the first step of post-processing

The full project is available is its [own repository](https://github.com/swiss-territorial-data-lab/proj-borderpoints).


The **installation** can be carried out by following the instructions in the main readme file. When using Docker, the container must be launched from the repository root folder before running the workflow:

```bash
$ sudo chown -R 65534:65534 examples
$ docker compose run --rm -it stdl-objdet
```

The worklfow commands are expected to be launched from this folder in Docker.

The Docker container is exited and permissions are restored with:

 ```bash
$ exit
$ sudo chmod -R a+w examples
```

## Data

The following datasets are available for this example in the `data` folder:

* images: SWISSIMAGE Journey is an annual dataset of aerial images of Switzerland from 1946 to today. The images are downloaded from the geo.admin.ch server using XYZ connector.
* empty tiles: tiles without any object of interest added to the training dataset to provide more contextual tiles.
* FP labels: objects frequently present among false positives, used to include the corresponding tiles in the training.
* ground truth: labels vectorised by the domain experts.
    Disclaimer: the ground truth dataset is unofficial and has no legal value. It has been produced specifically for the purposes of the project.

## Workflow

The workflow can be executed by running the following list of actions and commands.

Prepare the data:
```
$ python prepare_data.py config_trne.yaml
$ stdl-objdet generate_tilesets config_trne.yaml
```

Train the model:
```
$ stdl-objdet train_model config_trne.yaml
$ tensorboard --logdir output/trne/logs
```

Open the following link with a web browser: `http://localhost:6006` and identify the iteration minimising the validation loss and select the model accordingly (`model_*.pth`) in `config_trne`. The path to the trained model is `output/trne/logs/model_<number of iterations>.pth`, currently `model_0002499.pth` is used. <br>

Perform and assess detections:
```
$ stdl-objdet make_detections config_trne.yaml
$ stdl-objdet assess_detections config_trne.yaml
```

The detections obtained by tiles can be merged when adjacent:
```
$ python merge_detections.py config/config_trne.yaml
```