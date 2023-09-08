# MIHMS
 The Montana Integrated Hydrologic Modeling System code repository.

# Installation
 
 Use conda to install gdal and netcdf4, then run pip install -r requirements.txt

 # conda create -n mihm python=3.9
 # conda activate mihm
 # conda install gdal scipy netcdf4 rasterio
then, manually install remaining high-level packages with pip
 # pip install pygsflow pandas matplotlib dataretrieval fiona rtree shapely pyproj richdem xarray
alternatively, you can take the frozen requirements from the time of writing this doc:
 # pip install -r requirements.txt
Note: the requirements.txt file does not have the conda-installed dependencies listed.
