# Swimming Pool Detector

## Requirements

* Python >= 3.6 (as we need the support for "f-strings")

Dependencies may be installed with either `pip` or `conda`, by making use of the provided `requirements.txt` file.

## Scripts

### `scripts\generate_training_sets.py`

This scripts generates COCO-annotated training/validation/test datasets for the Geneva's Swimming Pools detection task, going through the following steps:

1. it downloads the required datasets:
    * cadastral parcels (CP),
    * lakes,
    * swimming pools (SP);
2. it exports the CP dataset to a GeoJSON file and invites the user to execute a Linux shell command which generates the [Slippy Map Tiles](https://developers.planet.com/tutorials/slippy-maps-101/) covering the latter dataset;
3. it computes the Area of Interest (AoI), as the geometrical difference between the tiles obtained at step #3 and the Léman lake;
4. it downloads one image per tile in the AoI, by calling a remote Raster Service;
5. it generates training, validation and test datasets out of the subset of tiles including swimming pool and - in theory - neither false positives (*i.e.* no swimming pool that is only found in aerial images) nor false negatives (*i.e.* no swimming pool that is only found in the SP dataset);
6. it generates [COCO annotations](http://cocodataset.org/#format-data) for the three datasets, by generating image segmentations according to the SP dataset.  

#### How-to

Please review the provided `config.yaml` file, then run the following:

```bash
$ python scripts\generate_training_sets.py config.yaml
```