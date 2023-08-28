Follow the steps listed below to reproduce my work on the Upper Yellowstone basin study. As steps are completed,
the resulting data's path should be edited in the project .ini file:

1. Produce a basin delineation using either:

    StreamStats, https://streamstats.usgs.gov/ss/
    - Delineate a basin by choosing and outlet just above the confluence of the Yellowstone and Shields rivers.
    - Export the basin to shapefile, project to EPSG 5071 (as with all data used in this project).
    USGS Gages II Database - https://pubs.er.usgs.gov/publication/70046617
    - Use this if you have a USGS gage, select from the GAGES-II data using the STAID corresponding to the
      USGS gage identifier.
    - https://drive.google.com/drive/folders/1AE0b4SA2SXN-zk4W2L-nS1SeAGUtG_HO?usp=drive_link

2. Clip raster data to the basin shapfile, using MT_Rsense/gsflow_prep/raster_prep.py and the statewide rasters
    in our database. Move rasters to directories specified in the project .ini.

3. To prepare climate and streamflow input data, run batch_sequence.py through impervious_parameters().
    Then run climate and stream gage data processing functions in gsflow_prep.py. This will build shapefiles
    with a selection of stream gages and climate stations, build the precipitation zones for the model domain,
    write the record of streamflow and climate data to a PRMS datafile, and calculate monthly min/max temperature
    lapse rates.

4. Run prism_800m_parameters() and ppt_ratio_parameters(). Open the hru_params.shp in a GIS and confirm the PPT_ZONE
    field has been set correctly. NOTE: I had to double the block size parameter (to 400000) in
    gsflow-arcpy/scripts/support_functions.py zone_by_centroid_func(), line #1181 when I found the basin area PPT_ZONE
    attributes were incomplete (zeros within the domain).

5. Run batch_sequence.py through flow_parameters(). Observe initial stream network, and change DEM_ADJ according
    to instructions in the tutorial. Re-run flow_parameters().  Run crt_fill_parameters(), examine. Re-run
    these two steps if necessary. See the gsflow-arcpy readme at github.com/dgketchum/gsflow-arcpy/README.txt.