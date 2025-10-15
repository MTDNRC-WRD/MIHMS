# MIHMS
 The Montana Integrated Hydrologic Modeling System code repository.

# Installation
  Currently must clone and/or download to local directory and install from there.

## Re-create Environment from Prompt

 ```
 conda create -n mihms python=3.10
 conda activate mihms
 conda install xarray rasterio geopandas pandas numpy rioxarray dask zarr tomli
 ```

then, manually install remaining high-level packages with pip
 
 `pip install flopy pygsflow spotpy`

finally use pip to install from url or git:

 ```
 pip install https://github.com/jlarsen-usgs/dany/archive/refs/heads/main.zip
 pip install git+https://github.com/MTDNRC-WRD/chmdata@main
 pip install git+https://github.com/MTDNRC-WRD/GRIDtools@main
 ```
 
