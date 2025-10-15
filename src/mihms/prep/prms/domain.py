from pathlib import Path
import json
from typing import Union, Optional

import geopandas as gpd
import pandas as pd
import numpy as np
import xarray as xr
from flopy.utils.triangle import Triangle
from flopy.utils.voronoi import VoronoiGrid
from flopy.discretization import VertexGrid as FlopyVertexGrid
from gsflow import PrmsData

from mihms.config import project_root, prep
from mihms.utils.io import NpEncoder


class PRMSVertexGrid(FlopyVertexGrid):
    def __init__(self, grid: Union[str, Path, VoronoiGrid, FlopyVertexGrid]):
        if isinstance(grid, (str, Path)):
            with open(grid) as f:
                in_grid = json.load(f)
                self.gridprops = in_grid
        elif isinstance(grid, VoronoiGrid):
            self.gridprops = grid.get_gridprops_vertexgrid()
            self.gridprops['nlay'] = 1
        elif isinstance(grid, FlopyVertexGrid):
            self.gridprops = {}
            self.gridprops['nlay'] = grid.nlay
            self.gridprops['ncpl'] = grid.ncpl
            self.gridprops['nvert'] = grid.nvert
            self.gridprops['vertices'] = grid._vertices
            self.gridprops['cell2d'] = grid._cell2d
            self.gridprops['crs'] = grid.crs
        else:
            raise ValueError("Input grid not recognized. The input grid is neither a path to .grid file or a flopy grid object.")
        super().__init__(**self.gridprops)

    def save_grid(self, grid_name, save_location):
        if self.crs is not None:
            self.gridprops['crs'] = self.crs.to_wkt()
        pth = Path(save_location)
        flname = f"{grid_name}.grid"
        with open(pth / flname, 'w') as f:
            json.dump(self.gridprops, f, cls=NpEncoder)


def create_obs_dataset(obs_data: pd.DataFrame,
                       obs_geom: gpd.GeoDataFrame,
                       col_link: str,
                       model_grid: Union[str, Path, PRMSVertexGrid, FlopyVertexGrid, gpd.GeoDataFrame],
                       sub_basins: Optional[np.ndarray] = None,
                       stream_array: Optional[np.ndarray] = None):
    """

    Args:
        obs_data:
        obs_geom:
        col_link:
        model_grid:
        sub_basins:
        stream_array:

    Returns:

    """
    if isinstance(model_grid, (str, Path)):
        fpgrid = PRMSVertexGrid(model_grid)
        geom_grid = fpgrid.geo_dataframe
    elif isinstance(model_grid, (PRMSVertexGrid, FlopyVertexGrid)):
        geom_grid = model_grid.geo_dataframe
    elif isinstance(model_grid, gpd.GeoDataFrame):
        geom_grid = model_grid
    else:
        raise ValueError("The input grid object type is not supported.")

    geom_grid['hru'] = geom_grid.index + 1

    # this is to deal with shapefiles...they clip the column headers output by the PRMSCascades class
    #   the column names should be set as 'hru_subbasin' and 'stream_grid'
    sub_name = [c for c in geom_grid.columns if 'hru_sub' in c]
    if len(sub_name) != 0:
        geom_grid.rename(columns={sub_name[0]: 'hru_subbasin'}, inplace=True)
        sub_name = True
    else:
        sub_name = False
    strm_name = [c for c in geom_grid.columns if 'stream_g' in c]
    if len(strm_name) != 0:
        geom_grid.rename(columns={strm_name[0]: 'stream_grid'}, inplace=True)
        strm_name = True
    else:
        strm_name = False

    if (not sub_name) & (sub_basins is not None):
        geom_grid['hru_subbasin'] = sub_basins
        sub_name = True

    if (not strm_name) & (stream_array is not None):
        geom_grid['stream_grid'] = stream_array
        strm_name = True

    if 'Date' not in obs_data.columns:
        raise ValueError(
            "No date column is included in the input observation data. At least one column must be named 'Date' and have date-like formatted strings or values.")

    df = obs_data.set_index(pd.DatetimeIndex(obs_data['Date']))
    df.drop(columns='Date', inplace=True)

    if not df.columns.isin(obs_geom[col_link]).all():
        raise ValueError(
            "Not all columns in the observation data frame are represented in the geometry file, while the geometry inputs can have more sites than are in the" \
            "observation data frame, it must at a minimum contain all of the columns. Either check the obs_data input or the col_link argument.")

    obs_geom = obs_geom.loc[obs_geom[col_link].isin(df.columns), :]
    df = df.loc[:, obs_geom[col_link].to_list()]

    grd_insct = obs_geom.sjoin(geom_grid)

    if strm_name:
        strm_cells = geom_grid.loc[geom_grid['stream_grid'] != 0, :]
        strm_insct = obs_geom.sjoin_nearest(strm_cells)

    # get only rows associated with the obs_data columns
    geom_sel = obs_geom[col_link].isin(df.columns)
    # change crs to get lat/longs
    lat_lon_geom = obs_geom.to_crs(4326).geometry

    obs_dset = xr.Dataset(
        {
            'observation_data': (['time', 'location'], df.values)
        },
        coords={
            'lon': ('location', lat_lon_geom.x.values[geom_sel]),
            'lat': ('location', lat_lon_geom.y.values[geom_sel]),
            'hru': ('location', grd_insct['hru'].values[geom_sel]),
            'location': ('location', df.columns.values),
            'time': df.index.values
        },
        attrs={'description': "Observation data associated with PRMS model domain."}
    )

    if sub_name:
        obs_dset.coords['subbasin'] = ('location', grd_insct['hru_subbasin'].values)

    if strm_name:
        obs_dset.coords['segment'] = ('location', strm_insct['stream_grid'].values)

    return obs_dset

def append_obs_to_datafile(dfile: Union[str, Path, PrmsData], obsdset: xr.Dataset):
    raise NotImplementedError("This function is not available yet.")

def voronoi_grid_from_hydrography(
        basin_geom: gpd.GeoDataFrame,
        base_res: float,
        strm_geom: gpd.GeoDataFrame,
        strm_res: float,
        strm_buff: float,
        lake_geom: Optional[gpd.GeoDataFrame] = None,
        lake_res: Optional[float] = None,
        simp_basin: Optional[float] = None,
        simp_strms: Optional[float] = None,
        simp_sinks: Optional[float] = None,
        simplify_units: str = 'grid_resolution'
) -> tuple:
    """Function to create an unstructured Voronoi grid for PRMS modeling.
        geometries - make sure there are no multipolygons these can make things weird
        simplify_units - determines the units in which input geometries are simplified by
            options are ['grid_resolution', 'map']
                - 'grid_resolution' (default) means that the simplification and stream buffer is specified as a fraction of the associated geometry resolution (e.g., if base_res=1000 meters then simp_basin=2 would use 1000/2=500 meters as the
                  simplification factor). If strm_buff = 3.0, the buffer would be 3*strm_res.
                - 'map' means that the simplification factor and strm_buff is specified in the map units (e.g., meters, feet, etc.), so 500 would be 500 m.
    """
    # assign simplification units
    if simplify_units == 'grid_resolution':
        strm_buff = strm_buff * strm_res

        if simp_basin is None:
            pass
        else:
            simp_basin = base_res / simp_basin

        if simp_strms is None:
            pass
        else:
            simp_strms = strm_res / simp_strms

        if (lake_geom is not None) & (simp_sinks is None):
            pass
        elif (lake_geom is not None) & (simp_sinks is not None):
            simp_sinks = lake_res / simp_sinks

    # do an initial simplificatin of input stream layer (has a lot of vertices if using nhd)
    if len(strm_geom.index) > 1:
        print("Input streams have more than one geometry, dissolving geometries...")
        strm_geom = strm_geom.dissolve()

    smpstrms = gpd.GeoDataFrame(
        geometry=strm_geom.geometry.simplify(strm_res)
    )
    # buffer stream layer to create a polygon region around streams
    buff_strms = smpstrms.copy()
    buff_strms['geometry'] = buff_strms.geometry.buffer(strm_buff, cap_style='round', resolution=8, join_style='round')
    # simplify geometry layers if specified by user, this is recommended as it removes input nodes to Triangle and eliminates
    #   very small grid cells being created
    if simp_strms is not None:
        strm_reg = buff_strms.geometry.simplify(simp_strms)
    else:
        strm_reg = buff_strms.geometry

    if simp_basin is not None:
        basin_reg = basin_geom.geometry.simplify(simp_basin)
    else:
        basin_reg = basin_geom.geometry

    if (lake_geom is not None) & (simp_sinks is not None):
        lake_reg = lake_geom.exterior.simplify(simp_sinks)
        lake_reg = lake_reg.polygonize().force_2d()
    elif (lake_geom is not None) & (simp_sinks is None):
        lake_reg = lake_geom.exterior
    else:
        lake_reg = None

    # If any geometries go outside the basin boundary, clip and adjust so the basin is the outermost polygon
    if not strm_reg.within(basin_reg).all():
        strm_reg = strm_reg.clip(basin_reg.buffer(-strm_res, join_style='mitre'))

    if lake_reg is not None:
        if not lake_reg.within(basin_reg).all():
            lake_reg = strm_reg.clip(basin_reg.buffer(-strm_res, join_style='mitre'))

    # Determine all overlapping regions created from geometry inputs - sometimes the stream lines buffer overlaps, creating holes that get ingonred by Triangle if not accounted for
    bsn_dif = basin_reg.difference(strm_reg.geometry[0]).explode().reset_index().drop(columns='index')
    if lake_reg is not None:
        bsn_dif = bsn_dif.difference(lake_reg.geometry[0]).explode().reset_index().drop(columns='index')
        strm_dif = strm_reg.difference(lake_reg.geometry[0]).explode().reset_index().drop(columns='index')
        lake_dif = lake_reg.difference(strm_reg.geometry[0]).explode().reset_index()
        bsn_dif = gpd.GeoDataFrame({'region': ['basin'] * len(bsn_dif)}, geometry=bsn_dif.geometry,
                                   index=range(0, len(bsn_dif)))
        strm_dif = gpd.GeoDataFrame({'region': ['stream'] * len(strm_dif)}, geometry=strm_dif.geometry)
        lake_dif = gpd.GeoDataFrame({'region': ['lake'] * len(lake_dif)}, geometry=lake_dif.geometry)
        difs = [bsn_dif, strm_dif, lake_dif]
    else:
        strm_dif = strm_reg
        bsn_dif = gpd.GeoDataFrame({'region': ['basin'] * len(bsn_dif)}, geometry=bsn_dif.geometry,
                                   index=range(0, len(bsn_dif)))
        strm_dif = gpd.GeoDataFrame({'region': ['stream'] * len(strm_dif)}, geometry=strm_dif.geometry)
        difs = [bsn_dif, strm_dif]

    diff_polys = pd.concat(difs, ignore_index=True)
    # sample a point within each polygon that is used by Triangle to define a refinement region
    diff_pnts = diff_polys.copy()
    diff_pnts['geometry'] = diff_polys.sample_points(1)
    # initialize Triangle and it's scratch working directory (right now housed within the MIHMS project root)
    tri_pth = project_root / prep.tri_working
    tri_pth.mkdir(exist_ok=True)
    tri_grid = Triangle(exe_name=project_root / prep.tri_exe, angle=20, model_ws=tri_pth)
    # add polygons to triangle instance
    tri_grid.add_polygon(basin_reg[0], ignore_holes=True)
    if lake_reg is not None:
        tri_grid.add_polygon(lake_reg[0], ignore_holes=True)
    tri_grid.add_polygon(strm_reg[0], ignore_holes=True)
    # create GeoDataFrame of refinement regions for output
    basin_rgdf = gpd.GeoDataFrame(basin_reg)
    basin_rgdf['regID'] = basin_rgdf.index.astype(int)
    basin_rgdf['region'] = 'basin'
    stream_rgdf = gpd.GeoDataFrame(strm_reg)
    stream_rgdf['regID'] = stream_rgdf.index.astype(int)
    stream_rgdf['region'] = 'stream'
    if lake_reg is not None:
        lakes_rgdf = gpd.GeoDataFrame(lake_reg)
        lakes_rgdf['regID'] = lakes_rgdf.index.astype(int)
        lakes_rgdf['region'] = 'lake'
        reg_gdf = pd.concat([basin_rgdf, stream_rgdf, lakes_rgdf], ignore_index=True)
    else:
        reg_gdf = pd.concat([basin_rgdf, stream_rgdf], ignore_index=True)
    reg_gdf.rename(columns={0: 'geometry'}, inplace=True)
    reg_gdf = reg_gdf.set_geometry('geometry')
    # add all refinement regions defined earlier
    reg_resltns = {
        'basin': base_res,
        'stream': strm_res,
        'lake': lake_res
    }

    for i in range(len(diff_pnts)):
        desg = diff_pnts.loc[i, 'region']
        tri_grid.add_region(diff_pnts.geometry[i], maximum_area=reg_resltns[desg] ** 2)
    # build TIN
    tri_grid.build()
    # build Voronoi polygons
    vor = VoronoiGrid(tri_grid)
    # Initilize PRMSVertexGrid straight from flopy VoronoiGrid
    vgrid = PRMSVertexGrid(vor)
    vgrid.crs = basin_geom.crs

    return (vgrid, tri_grid, reg_gdf)


def fill_voronoi_nan(array: np.ndarray, grid: gpd.GeoDataFrame) -> np.ndarray:
    in_array = array.copy()

    nanidx = np.where(np.isnan(in_array))[0]
    n_nans = len(nanidx)

    if n_nans != 0:
        print(f"Filling {n_nans} missing grid cell values...")
        iter = 0
        while iter < 10:
            nanidx = np.where(np.isnan(in_array))[0]
            n_nans = len(nanidx)

            if n_nans == 0:
                break

            for idx in nanidx:
                cell_adj = grid.loc[grid.intersects(grid.loc[idx, 'geometry']), :].index
                in_array[idx] = np.nanmean(in_array[cell_adj])

            iter += 1

    else:
        print("Input array has no missing values.")

    return in_array


def find_undeclared_sinks(fdobject: np.ndarray) -> np.ndarray:
    sinks = np.zeros(fdobject.shape, dtype=int)
    for i in range(len(fdobject)):
        if fdobject[fdobject[i]] == i:
            sinks[i] = 1
        else:
            continue

    return sinks
