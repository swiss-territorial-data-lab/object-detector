# Example: detection of quarries

A sample working setup is provided here, enabling the end-user to detect quarries and mineral extraction sites in Switzerland over several years. <br>
It consists of the following elements:

- ready-to-use configuration files:
    - `config_trne.yaml`;
    - `config_prd.yaml`;
    - `detectron2_config_dqry.yaml`.
- Input data in the `data` subfolder:
    - quarry **labels** issued from the [swissTLM3D](https://www.swisstopo.admin.ch/fr/geodata/landscape/tlm3d.html) product, revised and synchronized with the 2020 [SWISSIMAGE](https://www.swisstopo.admin.ch/fr/geodata/images/ortho/swissimage10.html) orthophotos;
    - the delimitation of the **Area of Interest (AoI)**;
    - the Swiss DEM raster is too large to be saved on this platform but can be downloaded from this [link](https://github.com/lukasmartinelli/swissdem) using the [EPSG:4326](https://epsg.io/4326) coordinate reference system. The raster must be re-projected to [EPSG:2056](https://epsg.io/2056), renamed as `switzerland_dem_EPSG2056.tif` and located in the **DEM** subfolder. This procedure is managed by running the bash script `get_dem.sh`. 
- A data preparation script (`prepare_data.py`) producing the files to be used as input to the `generate_tilesets` stage.
- A post-processing script (`filter_detections.py`) which filters detections according to their confidence score, altitude and area. The script also identifies and merges groups of nearby polygons.

The workflow can be run end-to-end by issuing the following list of commands, from the root folder of this GitHub repository:

```
$ sudo chown -R 65534:65534 examples
$ docker compose run --rm -it stdl-objdet
nobody@<id>:/app# cd examples/quarry-detection
nobody@<id>:/app# python prepare_data.py config_trne.yaml
nobody@<id>:/app# stdl-objdet generate_tilesets config_trne.yaml
nobody@<id>:/app# stdl-objdet train_model config_trne.yaml
nobody@<id>:/app# stdl-objdet make_detections config_trne.yaml
nobody@<id>:/app# stdl-objdet assess_detections config_trne.yaml
nobody@<id>:/app# python prepare_data.py config_prd.yaml
nobody@<id>:/app# stdl-objdet generate_tilesets config_prd.yaml
nobody@<id>:/app# stdl-objdet make_detections config_prd.yaml
nobody@<id>:/app# bash get_dem.sh
nobody@<id>:/app# python filter_detections.py config_prd.yaml
nobody@<id>:/app# exit
$ sudo chmod -R a+w examples
```

We strongly encourage the end-user to review the provided `config_trne.yaml` and `config_prd.yaml` files as well as the various output files, a list of which is printed by each script before exiting.

The model is trained on the 2020 [SWISSIMAGE](https://www.swisstopo.admin.ch/fr/geodata/images/ortho/swissimage10.html) mosaic. Inference can be performed on SWISSIMAGE mosaics of the product [SWISSIMAGE time travel](https://map.geo.admin.ch/?lang=en&topic=swisstopo&bgLayer=ch.swisstopo.pixelkarte-farbe&zoom=0&layers_timestamp=2004,2004,&layers=ch.swisstopo.swissimage-product,ch.swisstopo.swissimage-product.metadata,ch.swisstopo.images-swissimage-dop10.metadata&E=2594025.91&N=1221065.68&layers_opacity=1,0.7,1&time=2004&layers_visibility=true,true,false) by changing the year in `config_prd.yaml`. It should be noted that the model has been trained on RGB images and might not perform as well on B&W images.

For more information about this project, see [this repository](https://github.com/swiss-territorial-data-lab/proj-dqry) (not public yet).

## Disclaimer

Depending on the end purpose, we strongly recommend users not to take for granted the detections obtained through this code. Indeed, results can exhibit false positives and false negatives, as is the case in all Machine Learning-based approaches.