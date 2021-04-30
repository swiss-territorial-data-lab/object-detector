# Sample Data provided by the Canton of Neuchâtel

All datasets in this folder are polygon layers stored in the ESRI Shapefile format.
Data is provided by Système d'information du territoire Neuchâtelois (SITN) and was partially modified by the STDL data science team.

The cantonal surface area of Neuchâtel is divided into two distinct non-overlapping sectors together forming the Area of Interest (AoI):
* Sectors_groundtruth.shp specifies those areas of the canton that were manually and systematically checked for the existence of swimming pools.
* Sectors_other.shp delineates all remaining cantonal surface in which swimming pools might exist and might be labelled but are not systematically verified.

The labels used in object detection are polygon segments demarcating the boundary of swimming pools based on the cantonal SITN 2019 Orthophoto.
* sp_gt_missing_manuell.shp contains manually verified polygon segments for all swimming pools present within the Sectors_groundtruth.shp. Data was mostly provided as a polygon layer export from the cadastre by SITN. 20 additional ("missing") objects were identified by a point layer. These objects were manually digitized on 2020-Dec-01 and added to the original provided dataset in order to obtain an as complete as possible dataset within the area of Sectors_groundtruth.shp.
* Swimmingpools_other.shp contains unverified polygon segments stemming from the cadastre covering the area of Sectors_other.shp.
