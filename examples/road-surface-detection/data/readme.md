# Data
- AOI: We provide here the data for 3 tiles of the swiss national map at the scale 1:25'000. Two of those tiles (restricted AOI) are used for the training, validation and test of the model. The last one is only used for inferences.
- swissTLM3D:
	- The *roads* are originally in the form of lines and are transformed in polygons during the preparation thank to the parameters in `roads_parameters.xlsx`.
	- The *forests* are used to filter the roads as we can not determine the type of surface under the forest canopy.
- quarries: The quarries are used before the calculation of final metrics. We already know that roads in quarries have a natural surface. Therefore, those are not included in the calculation of final metrics.