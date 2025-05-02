import os
from subprocess import check_call
from typing import Union
from pathlib import Path
import geopandas as gpd
import rasterio as rio
import rioxarray

import fiona
from shapely.geometry import shape

from mihms.config import prep

warp_exe = prep.gdalwarp_exe
#PROJ4 = '+proj=aea +lat_0=23 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +ellps=GRS80' \
#        ' +towgs84=0,0,0,0,0,0,0 +units=m +no_defs'


def clip_raster(geom_pth, raster_pth, resamp='nearest', out_pth=None, buffer_extent=None):
    """
    Function to use gdalwarp to clip input raster to the bounds of an input geometry.
    :param geom_pth: str - path to the input geometry file
    :param raster_pth: str - path to the raster file to be clipped
    :param resamp: str - resample method passed to gdalwarp (default='nearest')
    :param out_pth: str - the path where the clipped raster is saved (default is same as raster_pth)
    :param buffer_extent: float - a distance to buffer the clip extent
    :return: None - output is a new .tif at the out_pth location
    """
    with fiona.open(geom_pth, 'r') as geom:
        geo = [f['geometry'] for f in geom][0]
        geo = shape(geo)
        if buffer_extent:
            geo = geo.buffer(buffer_extent)

    bnd = geo.bounds

    if out_pth is None:
        out_pth = raster_pth
    else:
        out_pth = out_pth

    _var = os.path.basename(raster_pth).split('.')[0]

    out_ras = os.path.join(out_pth, '{}_clipped.tif'.format(_var))

    cmd = [warp_exe, '-of', 'GTiff', '-r', resamp, '-overwrite',
           '-te', str(bnd[0]), str(bnd[1]), str(bnd[2]), str(bnd[3]),
           '-multi', '-wo', '-wo NUM_THREADS=8',
           raster_pth, out_ras]

    check_call(cmd)


def load_raster_window(geom: Union[gpd.GeoDataFrame, str, Path], raster_pth: Union[str, Path],
                       output: str = 'rio_meta'):
    """Returns a dict - {array, bands, raster metadata}"""
    if isinstance(geom, (str, Path)):
        G = gpd.read_file(geom)
    elif isinstance(geom, gpd.GeoDataFrame):
        G = geom
    else:
        raise ValueError(
            "Input geometry must be either string or Path object to a geometry file, or a geopandas GeoDataFrame.")

    if output == 'rio_meta':
        with rio.open(raster_pth, 'r') as src:
            if G.crs.to_epsg() == src.crs.to_epsg():
                pass
            else:
                print("Geometry and raster CRS do not match, reprojecting geometry to match raster dataset...")
                G = G.to_crs(src.crs.to_epsg())
            arr = src.read(window=rio.windows.from_bounds(G.total_bounds[0], G.total_bounds[1], G.total_bounds[2],
                                                          G.total_bounds[3], src.transform))
            bands = src.indexes
            rmeta = src.meta

            rmeta.update({'array': arr, 'bands': bands})
            return rmeta
    elif output == 'xarray':
        inrast = rioxarray.open_rasterio(raster_pth)
        if G.crs.to_epsg() == inrast.rio.crs.to_epsg():
            pass
        else:
            print("Geometry and raster CRS do not match, reprojecting geometry to match raster dataset...")
            G = G.to_crs(inrast.rio.crs)

        out = inrast.rio.clip_box(minx=G.total_bounds[0], miny=G.total_bounds[1], maxx=G.total_bounds[2],
                                  maxy=G.total_bounds[3], crs=inrast.rio.crs)

        return out

    else:
        raise ValueError("The output type specified is not recognized, choose 'rio_meta' or 'xarray'.")


def check_precip_string(strg: str) -> bool:
    b = ('pr' in strg) | ('prcp' in strg) | ('precip' in strg) | ('precipitation' in strg)
    return b

def check_min_temp_string(strg: str) -> bool:
    b = (('min' in strg) & ('t' in strg)) | (('min' in strg) & ('tmp' in strg)) | (('min' in strg) & ('temp' in strg)) | (('min' in strg) & ('temperature' in strg))
    return b

def check_max_temp_string(strg: str) -> bool:
    b = (('max' in strg) & ('t' in strg)) | (('max' in strg) & ('tmp' in strg)) | (('max' in strg) & ('temp' in strg)) | (('max' in strg) & ('temperature' in strg))
    return b