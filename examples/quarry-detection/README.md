# Example: detection of quarries

A sample working setup is provided here, enabling the end-user to detect quarries and mineral extraction sites in Switzerland over several years. <br>
It consists of the following elements:

- the read-to-use configuration files:
    - `config_trne.yaml`,
    - `config_prd.yaml`,
    - `detectron2_config_dqry.yaml`,
- the input data in the `data` subfolder:
    - quarries shapefile from the product [swissTLM3D](https://www.swisstopo.admin.ch/fr/geodata/landscape/tlm3d.html), revised and synchronised with the 2020 [SWISSIMAGE](https://www.swisstopo.admin.ch/fr/geodata/images/ortho/swissimage10.html) mosaic (**label**),
    - the delimitation of the AOI to perform inference predictions (**AOI**),
    - the swiss DEM raster is too large to be saved on this platform but can be downloaded from this [link](https://github.com/lukasmartinelli/swissdem) using the EPSG:4326 - WGS 84 coordinate reference system. The raster must be first reprojected to EPSG:2056 - CH1903+ / LV95, named `switzerland_dem_EPSG2056.tif`and located in the **DEM** subfolder.
- a data preparation script (`prepare_data.py`) producing the files to be used as input to the `generate_tilesets.py`script.
- a results post-processing script (`filter_prediction.py`) filtering the predictions, produced from `make_prediction.py`script, to the final shapefile 

In the provided Docker container, the end-to-end workflow can be run by issuing the following list of commands, straight from this folder:

```bash
$ python3 prepare_data.py config_trne.yaml
$ stdl-objdet generate_tilesets config_trne.yaml
$ stdl-objdet train_model config_trne.yaml
$ stdl-objdet make_predictions config_trne.yaml
$ stdl-objdet assess_predictions config_trne.yaml
$ python3 prepare_data.py config_prd.yaml
$ stdl-objdet generate_tilesets config_prd.yaml
$ stdl-objdet make_predictions config_prd.yaml
$ python3 filter_detection.py config_prd.yaml
```

We strongly encourage the end-user to review the provided `config_trne.yaml` and `config_prd.yaml` files as well as the various output files, a list of which is printed by each script before exiting.

The model is trained on the 2020 [SWISSIMAGE](https://www.swisstopo.admin.ch/fr/geodata/images/ortho/swissimage10.html) mosaic. Inference can be performed on SWISSIMAGE mosaics of the product SWISSIMAGE time travel by changing the year in `config_prd.yaml`. It should be noted that the model has been trained on RGB color images and might not perform as well on Black and White images.

For more information about this project, you can consult [the associated repository](https://github.com/swiss-territorial-data-lab/proj-dqry) (not public yet).

## Disclaimer

Depending on the end purpose, we strongly recommend users not to take for granted the detections obtained through this code. Indeed, results can exhibit false positives and false negatives, as is the case in all Machine Learning-based approaches.