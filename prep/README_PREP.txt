Follow the steps listed below to prep data to reproduce our work on the Upper Yellowstone basin study. As steps are completed,
the resulting data's path should be edited in the project .toml file:

All geospatial data should be in EPSG 5071 Albers Equal Area.

1. Produce a basin delineation using either:

    Preferred: USGS Gages II Database - https://pubs.er.usgs.gov/publication/70046617
    - Use this if you have a USGS gage, select from the GAGES-II data using the STAID corresponding to the
      USGS gage identifier.
      Also available at https://drive.google.com/drive/folders/141lhxvbnr-FdcW7AJguX3abWAv-Y9k6X?usp=drive_link

    StreamStats, https://streamstats.usgs.gov/ss/
    - Delineate a basin by choosing and outlet just above the confluence of the Yellowstone and Shields rivers.
    - Export the basin to shapefile, project to EPSG 5071 (as with all data used in this project).

    Save the gage point and the basin delineation as EPSG 5071 in the project's 'domain' folder.

2. Download the statewide rasters (in EPSG:5071) from
    https://drive.google.com/file/d/1xOLfxBWLJeKRIhHn48zqAbBFvXboR_K8/view
    Move rasters to directories specified in the project .toml.

3. Download required shapefiles from
    https://drive.google.com/drive/folders/141lhxvbnr-FdcW7AJguX3abWAv-Y9k6X?usp=sharing
    Unzip and move the shapefiles to directories specified in the project .toml.

4. Raster/grid related processing takes place when the StandardBuild object is instantiated.

5. GSFlow remap directories may be found here, note keys may be missing from the directory:
    https://github.com/gsflow/gsflow-arcpy/tree/master/remaps