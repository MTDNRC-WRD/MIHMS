from pathlib import Path
from typing import Union, Optional

import xarray as xr
import rasterio as rio
import pandas as pd
import geopandas as gpd
import numpy as np
from flopy.discretization import VertexGrid as FlopyVertexGrid
from gsflow.prms import PrmsData
from gsflow.builder.builder_utils import ea
import GRIDtools as gt

from mihms.prep.prms import PRMSParameters, PRMSVertexGrid
from mihms.prep.prms.utils import check_precip_string, check_max_temp_string, check_min_temp_string

def temp_1sta_from_stations(sta_locs: Union[str, Path, gpd.GeoDataFrame],
                            sta_data: Union[str, Path, pd.DataFrame],
                            param_obj: Optional[PRMSParameters] = None) -> tuple:
    """

    Args:
        sta_locs:
        sta_data:
        param_obj:

    Returns:

    """
    pass


def temp_1sta_from_gridded(gridded_data: Union[str, Path, xr.Dataset],
                           elev_data: Union[str, Path, rio.DatasetReader, xr.DataArray, xr.Dataset],
                           model_grid: Optional[Union[str, Path, PRMSVertexGrid, FlopyVertexGrid, gpd.GeoDataFrame]] = None,
                           param_obj: Optional[PRMSParameters] = None,
                           temp_units='celcius',
                           precip_units='mm') -> tuple:
    """

    Args:
        gridded_data: str | Path | xarray.Dataset
            A filename or path to a netcdf file, or a xarray Dataset that contains input variables precipitation,
            minimum temperature, and maximum temperature
        elev_data: str | Path | rasterio.DatasetReader | xarray.DataArray | xarray.Dataset
            Raster data of elevation to summarize over meteorology grid, anything accepted by GRIDtools.RasterClass
        model_grid: str | Path | PRMSVertexGrid | FlopyVertexGrid | gpd.GeoDataFrame
            An accepted grid type or GeoDataFrame of the model grid over which meteorology is to be distributed.
            This argument is ignored if 'param_obj' has a valid grid assigned.
        param_obj: PRMSParamters
            A valid parameter object. If no grid is assigned, must pass a grid to 'model_grid' argument, otherwise the
            grid assigned to 'param_obj' will be used.

    Returns: tuple
        A tuple (PRMSParameters, gsflow.PrmsData) with the edited PRMSParameters object and a pygsflow PrmsData object
        containing the formatted meteorology data.

    """
    if (param_obj is None) & (model_grid is None):
        raise ValueError("No model grid or parameter object was specified, must give a PRMSParameters object with grid"
                         "assigned or a valid model grid.")

    if (param_obj is None) & (model_grid is not None):
        param_obj = PRMSParameters(grid=model_grid)

    if (param_obj is not None) & (param_obj.grid is None):
        raise ValueError("PRMSParameters object was provided, but no grid is assigned. Assign grid to the parameters"
                         "object or pass a valid grid argument to the function.")

    if temp_units == 'celcius':
        tmpu = 1
        lapse_constraint = [-12, 12]
        allsnow_def = 0
    elif temp_units == 'fahrenheit':
        tmpu = 0
        lapse_constraint = [-20, 20]
        allsnow_def = 32
    else:
        raise ValueError("The temperature units are not recognized or are mispelled, use 'fahrenheit' or 'celcius'")

    if precip_units == 'mm':
        pru = 1
    elif precip_units == 'inches':
        pru = 0
    else:
        raise ValueError("The precip units are not recognized or are mispelled, use 'inches' or 'mm'")

    grid_vec = gt.utils.vectorize_grid(gridded_data)
    grid_vec = grid_vec.to_crs(param_obj.grid.crs)
    grid_cent = param_obj.grid.copy()
    grid_cent['geometry'] = param_obj.grid.centroid
    insct = grid_cent.sjoin(grid_vec, predicate='within')

    gvec_sel = grid_vec.loc[insct['index_right'].unique()]
    gvec_sel.sort_index(inplace=True)
    gvec_sel.reset_index(inplace=True)
    met_keys = pd.Series(gvec_sel.index, index=gvec_sel['index']).to_dict()

    insct['station_id'] = insct['index_right'].replace(met_keys) + 1

    grid_elev = gt.calc_zonal_stats(grid_vec, elev_data, stats='median', all_touched=True)
    stn_elev = grid_elev.loc[(insct['index_right'].values, 1, 'median'), :].values.ravel()

    vars = list(gridded_data.data_vars)
    var_names = {}
    for v in vars:
        if check_precip_string(v):
            var_names.update({v: 'precip'})
        elif check_min_temp_string(v):
            var_names.update({v: 'tmin'})
        elif check_max_temp_string(v):
            var_names.update({v: 'tmax'})
        else:
            raise ValueError(
                f"Variable name {v} not recognized as precipitation, minimum temperature, or maximum temperature.")

    gridded_data = gridded_data.rename(var_names)
    met_t, met_r, met_c = gridded_data.precip.values.shape
    met_shape = (met_r, met_c)
    station_idx = np.unravel_index(gvec_sel['index'].values, met_shape)
    modgrid_idx = np.unravel_index(insct['index_right'].values, met_shape)

    tminsel = gridded_data.tmin
    tminda = tminsel.rio.set_crs(gridded_data.rio.crs)
    tmaxsel = gridded_data.tmax
    tmaxda = tmaxsel.rio.set_crs(gridded_data.rio.crs)

    tmin_lr = lapse_rates_from_gridded(tminda, elev_data)
    tmin_lr = xr.where(tmin_lr > lapse_constraint[1], lapse_constraint[1], tmin_lr)
    tmin_lr = xr.where(tmin_lr < lapse_constraint[0], lapse_constraint[0], tmin_lr)
    tmin_lapse = tmin_lr.values[:, modgrid_idx[0], modgrid_idx[1]]
    tmax_lr = lapse_rates_from_gridded(tmaxda, elev_data)
    tmax_lr = xr.where(tmax_lr > lapse_constraint[1], lapse_constraint[1], tmax_lr)
    tmax_lr = xr.where(tmax_lr < lapse_constraint[0], lapse_constraint[0], tmax_lr)
    tmax_lapse = tmax_lr.values[:, modgrid_idx[0], modgrid_idx[1]]

    tmin_cols = [f"tmin_{x}" for x in range(len(station_idx[0]))]
    tmin_dataob = pd.DataFrame(gridded_data.tmin.values[:, station_idx[0], station_idx[1]], columns=tmin_cols)
    tmax_cols = [f"tmax_{x}" for x in range(len(station_idx[0]))]
    tmax_dataob = pd.DataFrame(gridded_data.tmax.values[:, station_idx[0], station_idx[1]], columns=tmax_cols)
    pr_cols = [f"precip_{x}" for x in range(len(station_idx[0]))]
    pr_dataob = pd.DataFrame(gridded_data.precip.values[:, station_idx[0], station_idx[1]], columns=pr_cols)

    dates = pd.DataFrame(pd.DatetimeIndex(gridded_data.time), columns=['Date'])
    dates['Year'] = dates['Date'].dt.year
    dates['Month'] = dates['Date'].dt.month
    dates['Day'] = dates['Date'].dt.day
    dates['Hour'] = dates['Date'].dt.hour
    dates['Minute'] = dates['Date'].dt.minute
    dates['Second'] = dates['Date'].dt.second
    datadf = pd.concat([dates, tmin_dataob, tmax_dataob, pr_dataob], axis=1)
    datadf = datadf[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'] + tmax_cols + tmin_cols + pr_cols + ['Date']]
    outdat = PrmsData(data_df=datadf)

    param_obj.add_parameter(
        'hru_psta',
        insct['station_id'].values,
        len(insct),
        'nhru'
    )

    param_obj.add_parameter(
        'hru_tsta',
        insct['station_id'].values,
        len(insct),
        'nhru'
    )

    param_obj.add_parameter(
        'max_missing',
        np.array([3]),
        1,
        'one'
    )

    param_obj.add_parameter(
        'rain_adj',
        np.zeros((12, len(param_obj.grid))),
        [12, len(param_obj.grid)],
        ['nmonths', 'nhru']
    )

    param_obj.add_parameter(
        'snow_adj',
        np.zeros((12, len(param_obj.grid))),
        [12, len(param_obj.grid)],
        ['nmonths', 'nhru']
    )

    param_obj.add_parameter(
        'tmax_adj',
        np.zeros((12, len(param_obj.grid))),
        [12, len(param_obj.grid)],
        ['nmonths', 'nhru']
    )

    param_obj.add_parameter(
        'tmin_adj',
        np.zeros((12, len(param_obj.grid))),
        [12, len(param_obj.grid)],
        ['nmonths', 'nhru']
    )

    param_obj.add_parameter(
        'tmax_allrain_offset',
        np.zeros((12, len(param_obj.grid))),
        [12, len(param_obj.grid)],
        ['nmonths', 'nhru']
    )

    param_obj.add_parameter(
        'tmax_allsnow',
        np.ones((12, len(param_obj.grid))) * allsnow_def,
        [12, len(param_obj.grid)],
        ['nmonths', 'nhru']
    )

    param_obj.add_parameter(
        'temp_units',
        np.array([tmpu]),
        1,
        'one'
    )

    param_obj.add_parameter(
        'precip_units',
        np.array([pru]),
        1,
        'one'
    )

    param_obj.add_parameter(
        'tsta_elev',
        stn_elev,
        len(stn_elev),
        'ntemp'
    )

    param_obj.add_parameter(
        'tmin_lapse',
        tmin_lapse,
        [tmin_lapse.shape[0], tmin_lapse.shape[1]],
        ['nmonths', 'nhru']
    )

    param_obj.add_parameter(
        'tmax_lapse',
        tmax_lapse,
        [tmax_lapse.shape[0], tmax_lapse.shape[1]],
        ['nmonths', 'nhru']
    )

    return (param_obj, outdat)



def xyz_dist_meteorology(sta_locs: Union[str, Path, gpd.GeoDataFrame],
                         sta_data: Union[str, Path, pd.DataFrame],
                         param_obj: Optional[PRMSParameters] = None) -> tuple:
    pass


def lapse_rates_from_gridded(temp_grid: Union[str, Path, xr.DataArray],
                             elev: Union[str, Path, rio.DatasetReader, xr.DataArray, xr.Dataset]) -> xr.DataArray:
    """
    Calculates the lapse rate from gridded temperature data and an elevation raster using first-order differential
    gradients of temperature and elevation.

    Args:
        temp_grid (str | Path | xarray.DataArray):
            A path to netcdf, or xarray DataArray object of temperature data with a valid "time" dimension.
        elev (str | Path | rasterio.DatasetReader | xarray.DataArray | xarray.Dataset):
            A raster of elevation data. The raster must cover the full extent of the temperature data grid.

    Returns (xarray.DataArray):
        A monthly dataset of temperature lapse rate covering the same grid as the input temperature data.
    """
    tmp_vec = gt.utils.vectorize_grid(temp_grid)
    elev_zon = gt.calc_zonal_stats(tmp_vec, elev, stats='median', all_touched=True)
    medelev = elev_zon.loc[(slice(None), slice(None), 'median'), :]
    medelev = medelev.iloc[:, 0].values.reshape((temp_grid.shape[1], temp_grid.shape[2]))
    medgradx, medgrady = np.gradient(medelev)
    mn_grad = np.mean(np.array([medgradx, medgrady]), axis=0)
    mon_tmp = temp_grid.groupby("time.month").mean("time")
    month_grads = []
    for m in mon_tmp.month.values - 1:
        tgradx, tgrady = np.gradient(mon_tmp.values[m])
        mn_tgrad = np.mean(np.array([tgradx, tgrady]), axis=0)
        month_grads.append(mn_tgrad)
    tgrad_arr = np.array(month_grads)
    lap_rate = (tgrad_arr / np.array([mn_grad])) * 1000
    lapse_rate = xr.DataArray(lap_rate, coords=mon_tmp.coords, dims=mon_tmp.dims, name='lapse_rate',
                              attrs={'units': 'temp/1000elev_units'})
    return lapse_rate

def mean_temp_warmest_month_by_hru(data_obj: PrmsData, hru_tsta: np.ndarray) -> tuple:
    """
    Extracts the mean min and max temperature for the warmest month of the year for every hru in model domain.

    Args:
        data_obj: gsflow.PrmsData
            A pygsflow data object that holds station data (proper labels for temp are tmin & tmax).
        hru_tsta: np.ndarray
            An array of temperature station values associated with each hru in model domain.

    Returns: tuple
        (minimum temperature, maximum temperature) each the same dimensions as hru_tsta input.

    """
    dataf = data_obj.data_df.copy()
    dataf.set_index('Date', inplace=True)
    gp = dataf.groupby(dataf.index.month).mean()
    mxmnmnth = gp.max()
    tminmxmn = mxmnmnth.filter(like='tmin').values
    tmaxmxmn = mxmnmnth.filter(like='tmax').values
    tsta_idx = hru_tsta - 1
    tminmxmn_f = tminmxmn[tsta_idx]
    tmaxmxmn_f = tmaxmxmn[tsta_idx]

    return (tminmxmn_f, tmaxmxmn_f)

def jh_temperature_coefficient(hru_elev: np.ndarray,
                               hru_mintemp: np.ndarray,
                               hru_maxtemp: np.ndarray,
                               elev_scalar: float = 1.0) -> np.ndarray:
    """
    Calculates the Jensen-Haise temperature adjustment coefficient using station min and max temperature.

    Args:
        hru_elev: numpy.ndarray
            The elevation of each hru in model domain. Must be in feet.
        hru_mintemp: numpy.ndarray
            The mean of the minimum temperature for the warmest month of the year for each hru in the
            domain. In celcius.
        hru_maxtemp: numpy.ndarray
            The mean of the maximum temperature for the warmest month of the year for each hru in the
            domain. In celcius.
        elev_scalar: float
            Conversion factor for elevation data default = 1.0, use 3.281 to convert meters to feet.

    Returns: numpy.ndarray
        The estimated value of jh_coef_hru.

    """
    jh_coef_hru = 27.5 - 0.25 * (ea(hru_maxtemp) - ea(hru_mintemp)) - ((hru_elev * elev_scalar) / 1000.0)

    return jh_coef_hru

def align_ddsolrad_with_gridded(gridded_data: Union[str, Path, xr.Dataset],
                                hru_lat: np.ndarray,
                                hru_slope: np.ndarray,
                                model_grid: Optional[Union[str, Path, PRMSVertexGrid, FlopyVertexGrid, gpd.GeoDataFrame]] = None,
                                param_obj: Optional[PRMSParameters] = None) -> PRMSParameters:
    pass


def align_et_gridded(gridded_data: Union[str, Path, xr.Dataset],
                    hru_elev: np.ndarray,
                    model_grid: Optional[Union[str, Path, PRMSVertexGrid, FlopyVertexGrid, gpd.GeoDataFrame]] = None,
                    param_obj: Optional[PRMSParameters] = None) -> PRMSParameters:
    pass


def sun_obliquity(dt: pd.DatetimeIndex) -> pd.Series:
    pass