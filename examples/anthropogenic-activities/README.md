# Example: segmentation of anthropic soils on historic images

A working setup is provided here to test the multi-class classification and the use of empty tiles, as well as false positive ones.
It consists of the following elements:

* ready-to-use configuration files
* input data
* scripts for data preparation and the first step of post-processing

The full project is available is its [own repository](https://github.com/swiss-territorial-data-lab/proj-sda).


The **installation** can be carried out by following the instructions [here](../../README.md). When using Docker, the container must be launched from this repository root folder before running the workflow:

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

"[SWISSIMAGE Journey through time](https://www.swisstopo.admin.ch/en/timetravel-aerial-images)" is an annual dataset of aerial images of Switzerland from 1946 to today. These images are downloaded from the geo.admin.ch server using the XYZ connector.

The following datasets are available for this example in the `data` folder:

* Empty tiles: tiles without any object of interest can be added to the training dataset to provide more context.
* False positive (FP) labels: objects frequently present among false positives, used to include the corresponding tiles in the training.
* Ground truth: labels vectorised by domain experts.

> [!CAUTION]
> The ground truth dataset is unofficial and has been produced specifically for the purposes of the project.
    

## Workflow

The workflow can be executed by running the following list of actions and commands.

Prepare the data:

```bash
$ python prepare_data.py config_trne.yaml
$ stdl-objdet generate_tilesets config_trne.yaml
```

Train the model:

```bash
$ stdl-objdet train_model config_trne.yaml
$ tensorboard --logdir output/trne/logs
```

Open another shell and launch TensorBoard:

```bash
$ docker compose run --rm -p 6006:6006 stdl-objdet tensorboard --logdir /app/examples/anthropogenic-activities/output/trne/logs --bind_all
```

Open the following link with a web browser: `http://localhost:6006` and identify the iteration minimising the validation loss and select the model accordingly (`model_*.pth`) in `config_trne`. The path to the trained model is `output/trne/logs/model_<number of iterations>.pth`, currently `model_0002499.pth` is used.

Perform and assess detections:

```bash
$ stdl-objdet make_detections config_trne.yaml
$ stdl-objdet assess_detections config_trne.yaml
```

The detections obtained by tiles can be merged when adjacent:

```bash
$ python merge_detections.py config_trne.yaml
```