import os
from subprocess import check_call
from typing import Union
from pathlib import Path
import datetime

import geopandas as gpd
import pandas as pd
import rasterio as rio
import rioxarray
import fiona
import numpy as np
from shapely.geometry import shape
from gsflow.prms import PrmsData

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


# this is a copy from pygsflow code - develop branch, stand in until the develop branch is merged to master and the
# pandas deprication problem is fixed with lineterminator
def write_prms_datafile(data_obj: PrmsData, filename: Union[str, Path]):
    """
    Method to write PrmsData input to a PRMS Data file

    Parameters
    ----------
    data_obj: gsflow.prms.PrmsData
        A gsflow data object to write out to file.
    filename : str | Path
        Data file file name

    """

    with open(filename, "w") as fid:
        fid.write(data_obj.header)
        fid.write("\n")
        columns = data_obj.data_df.columns
        climate_data = []
        climate_count = {}
        climate_unique = []
        for col in columns:
            nm = col[:col.rfind('_')]
            if nm in PrmsData.data_names:
                climate_data.append(nm)
                if not (nm in climate_unique):
                    climate_unique.append(nm)
                if nm in climate_count.keys():
                    climate_count[nm] = climate_count[nm] + 1
                else:
                    climate_count[nm] = 1

        # write headers
        for clim_name in climate_unique:
            line = clim_name + " " + str(climate_count[clim_name]) + "\n"
            fid.write(line)
        fid.write(
            "#########################################################################\n"
        )
        pd_to_write = data_obj.data_df.copy()
        pd_to_write = pd_to_write.drop(["Date"], axis=1)

        try:
            pd_to_write.to_csv(
                fid, index=False, sep=" ", lineterminator="\n", header=False
            )
        except:
            # remove this once line_terminator is fully deprecated
            pd_to_write.to_csv(
                fid, index=False, sep=" ", line_terminator="\n", header=False
            )

def convert_pywatershed_dims(dimtup: tuple) -> tuple:
    """
    pywatershed metadata (used by the mihms pakcage) uses different dimension names than standard PRMS for ndays,
    nmonths, and one. This function converts any pywatershed values to PRMS/pygsflow equivalents.
    Args:
        dimtup: tuple
            tuple of parameter dimensions

    Returns: tuple
        A tuple with effected dimensions converted to those recognizable by PRMS

    """
    conv_dims = []
    for i in dimtup:
        if i == 'nmonth':
            i = 'nmonths'
        elif i == 'ndoy':
            i = 'ndays'
        elif i == 'scalar':
            i = 'one'
        else:
            pass

        conv_dims.append(i)

    return tuple(conv_dims)


def replace_array_values(a: np.ndarray, value_map: Union[dict, tuple]) -> np.ndarray:
    """
    Takes a numpy array and replaces values in the array with a new set of values based on a dictionary mapping or
    a tuple of numpy arrays of the format (key value array, target value array).
    Args:
        a: numpy.ndarray
            The array whose values need remapped (only test on 1D arrays).
        value_map: dict | tuple
            A dictionary with keys of the original values and the target values, or a tuple of numpy.ndarrays
            representing the same thing, all values present in input array "a" must be represented in the value_map.

    Returns: numpy.ndarray
        Returns the remapped array with new target values.

    """
    if isinstance(value_map, dict):
        keyval = np.array(list(value_map.keys()))
        toval = np.array(list(value_map.values()))
    elif isinstance(value_map, tuple):
        keyval, toval = value_map
    else:
        raise ValueError("The input value map is not a dictionary or tuple of keys and values.")

    sort_idx = np.argsort(keyval)
    idx = np.searchsorted(keyval, a, sorter=sort_idx)
    out = toval[sort_idx][idx]

    return out

def load_prms_statvar(statvr_fl: Union[str, Path]) -> pd.DataFrame:
    with open(statvr_fl, "r") as fid:
        nvals = int(fid.readline().strip())
        var_names = []
        for header in range(nvals):
            nm, elem = fid.readline().strip().split()
            nm = nm + "_" + elem
            var_names.append(nm)

        columns = ["ID"] + ["Year", "Month", "Day", "Hour", "Minute", "Second"] + var_names
        stat_df = pd.read_csv(fid, sep="\\s+", names=columns)
        dts = []
        for index, irow in stat_df.iterrows():
            dt = datetime.datetime(
                year=int(irow["Year"]),
                month=int(irow["Month"]),
                day=int(irow["Day"]),
                hour=int(irow["Hour"]),
                minute=int(irow["Minute"]),
                second=int(irow["Second"]),
            )
            dts.append(dt)

        stat_df["Date"] = dts
        stat_df = stat_df.set_index('Date')
        stat_df = stat_df.loc[:, [c for c in stat_df.columns if
                                             c not in ["Date", "Year", "Month", "Day", "Hour", "Minute", "Second"]]]

    return stat_df