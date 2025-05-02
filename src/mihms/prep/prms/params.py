import os
from pathlib import Path
import copy

import geopandas as gpd
import pandas as pd
import rasterio as rio
import xarray as xr
import numpy as np
from flopy.discretization import VertexGrid as FlopyVertexGrid
import GRIDtools as gt
from typing import Union, Optional
from gsflow.prms import ParameterRecord
from gsflow import PrmsParameters as pygsflowparams
import warnings
import inspect
import dany
from shapely.geometry import LineString

from mihms.prep.prms import meta
from mihms.utils.io import user_warning
from mihms.prep.prms import PRMSVertexGrid, find_undeclared_sinks

prms_dtypes = {
    'integer': {'PRMS': 1, 'numpy': np.integer, 'netcdf': 'i4', 'python': int},
    'real': {'PRMS': 2, 'numpy': np.float32, 'netcdf': 'f4', 'python': float},
    'double': {'PRMS': 3, 'numpy': np.float64, 'netcdf': 'f8', 'python': float},
    'string': {'PRMS': 4, 'numpy': np.dtypes.StringDType(), 'netcdf': 'S1', 'python': str}
}


class PRMSParameters:

    def __init__(self, param_ds: Optional[xr.Dataset] = None,
                 grid: Optional[Union[str, Path, PRMSVertexGrid, FlopyVertexGrid, gpd.GeoDataFrame]] = None):

        if param_ds is None:
            self.parameters = xr.Dataset()
        elif (param_ds is not None) & (validate_param_dset(param_ds)):
            self.parameters = param_ds
        else:
            raise ValueError("The input xarray dataset contains invalid parameters or dimensions.")

        self._grid = None
        self._flopygrid = None

        if grid is not None:
            self.grid = grid

        self._pygsflow_dimrecs = None

    @property
    def grid(self):
        return self._grid

    def add_parameter_from_raster(self, param_name: str,
                                  dim_names: Union[str, list],
                                  in_raster: Union[str, Path, rio.DatasetReader, xr.DataArray, xr.Dataset],
                                  stat: str,
                                  **kwargs):
        if self.grid is None:
            raise RuntimeError(
                "No PRMSParameters.grid was specified. Spatial parameter mapping functions are not accessible without a grid object.")

        zonal = gt.calc_zonal_stats(self.grid, in_raster, stats=stat, **kwargs)

        if len(zonal.values) != len(self.grid):
            warnings.warn(
                " ".join(
                    [
                        f"The returned zonal statistics do not match the number of grid cells in the grid.",
                        "This may impact the mapping of the parameter."
                    ]
                ),
                UserWarning,
                stacklevel=2
            )

        self.add_parameter(param_name, zonal.values.ravel(), list(zonal.values.shape), dim_names)

    def add_parameter(self, param_name: str,
                      values: np.array,
                      dims: Union[int, list],
                      dim_names: Union[str, list],
                      add_attrs: Optional[dict] = None,
                      arr_format: str = 'C'):

        if not meta.is_available(param_name):
            raise ValueError(f"The input parameter {param_name} is not recognized as a valid parameter.")

        if isinstance(dims, list) & isinstance(dim_names, list):
            if len(dims) != len(dim_names):
                raise ValueError("The number of dimension values and dimension names do not match.")

        if isinstance(dim_names, str):
            if not meta.is_available(dim_names):
                raise ValueError(f"The parameter dimension {dim_names} is not recognized.")
            dim_names = [dim_names]
            dims = [dims]
        elif isinstance(dim_names, list):
            if any(x not in list(meta.dimensions.keys()) for x in dim_names):
                raise ValueError(
                    f"A specified parameter dimension in {dim_names} for parameter {param_name} is not recognized.")
        else:
            raise ValueError("The dim_names argument is invalid, a string or tuple of strings is accepted.")

        if param_name in list(self.parameters.data_vars):
            warnings.warn(
                " ".join(
                    [
                        f"The parameter already exists, overwriting the existing '{param_name}' parameter."
                    ]
                ),
                UserWarning,
                stacklevel=2
            )
            attrs_m = meta.find_variables(param_name)[param_name]
            attrs = copy.deepcopy(attrs_m)
            if 'requires' in attrs:
                del attrs['requires']
            if add_attrs is not None:
                attrs.update(add_attrs)
            if (len(dim_names) > 1) & (arr_format == 'F'):
                self.parameters[param_name] = (
                (dim_names[-1], dim_names[0]), np.reshape(values, (dims[-1], dims[0])), attrs)
            else:
                self.parameters[param_name] = (tuple(dim_names), values, attrs)
        else:
            attrs_m = meta.find_variables(param_name)[param_name]
            attrs = copy.deepcopy(attrs_m)
            if 'requires' in attrs:
                del attrs['requires']
            if add_attrs is not None:
                attrs.update(add_attrs)
            if (len(dim_names) > 1) & (arr_format == 'F'):
                self.parameters[param_name] = (
                (dim_names[-1], dim_names[0]), np.reshape(values, (dims[-1], dims[0])), attrs)
            else:
                self.parameters[param_name] = (tuple(dim_names), values, attrs)

    def add_pygsflow_record(self, record: ParameterRecord):
        if not isinstance(record, ParameterRecord):
            raise ValueError("The input record is not a pygsflow ParameterRecord object.")
        if record.file_name is not None:
            if record.section == 'Dimensions':
                if self._pygsflow_dimrecs is None:
                    self._pygsflow_dimrecs = [record]
                else:
                    self._pygsflow_dimrecs.append(record)
            else:
                self.add_parameter(record.name,
                                   record.values,
                                   record.dims,
                                   record.dimensions_names,
                                   add_attrs={'file_name': record.file_name},
                                   arr_format='F')
        else:
            if record.section == 'Dimensions':
                if self._pygsflow_dimrecs is None:
                    self._pygsflow_dimrecs = [record]
                else:
                    self._pygsflow_dimrecs.append(record)
            else:
                self.add_parameter(record.name,
                                   record.values,
                                   record.dims,
                                   record.dimensions_names,
                                   arr_format='F')

    @grid.setter
    def grid(self, ingrid: Union[str, Path, PRMSVertexGrid, FlopyVertexGrid, gpd.GeoDataFrame]):
        if isinstance(ingrid, (str, Path)):
            fpgrid = PRMSVertexGrid(ingrid)
            geom_grid = fpgrid.geo_dataframe
            if (self.parameters is not None) & ('nhru' in list(self.parameters.sizes.keys())):
                if len(geom_grid) == self.parameters.sizes['nhru']:
                    self._grid = geom_grid
                    self._flopygrid = fpgrid
                else:
                    raise ValueError("The input grid does not match the number of HRUs.")
            else:
                warnings.warn(
                    " ".join(
                        [
                            f"There are no parameters specified yet and or no nhru dimension to check grid.",
                            "Assigning grid anyway, make sure to check any inserted parameters match the grid size."
                        ]
                    ),
                    UserWarning,
                    stacklevel=2
                )
                self._grid = geom_grid
                self._flopygrid = fpgrid
        elif isinstance(ingrid, (PRMSVertexGrid, FlopyVertexGrid)):
            geom_grid = ingrid.geo_dataframe
            if (self.parameters is not None) & ('nhru' in list(self.parameters.sizes.keys())):
                if len(geom_grid) == self.parameters.sizes['nhru']:
                    self._grid = geom_grid
                    self._flopygrid = ingrid
                else:
                    raise ValueError("The input grid does not match the number of HRUs.")
            else:
                warnings.warn(
                    " ".join(
                        [
                            f"There are no parameters specified yet and or no nhru dimension to check grid.",
                            "Assigning grid anyway, make sure to check any inserted parameters match the grid size."
                        ]
                    ),
                    UserWarning,
                    stacklevel=2
                )
                self._grid = geom_grid
                self._flopygrid = ingrid
        elif isinstance(ingrid, gpd.GeoDataFrame):
            if (self.parameters is not None) & ('nhru' in list(self.parameters.sizes.keys())):
                if len(ingrid) == self.parameters.sizes['nhru']:
                    self._grid = ingrid
                else:
                    raise ValueError("The input grid does not match the number of HRUs.")
            else:
                warnings.warn(
                    " ".join(
                        [
                            f"There are no parameters specified yet and or no nhru dimension to check grid.",
                            "Assigning grid anyway, make sure to check any inserted parameters match the grid size."
                        ]
                    ),
                    UserWarning,
                    stacklevel=2
                )
                self._grid = ingrid

        else:
            raise ValueError("The input grid object type is not supported.")

    # pygsflow param object has to include dimensions section as well
    @property
    def pygsflow_param_obj(self):
        param_list = []
        if len(list(self.parameters.data_vars)) == 0:
            raise ValueError("There are no parameters specified.")
        if self._pygsflow_dimrecs is None:
            for key, value in self.parameters.sizes.items():
                dim_record = ParameterRecord(name=key, values=[value], datatype=1)
                param_list.append(dim_record)
        else:
            for key, value in self.parameters.sizes.items():
                if key in [x.name for x in self._pygsflow_dimrecs]:
                    continue
                else:
                    dim_record = ParameterRecord(name=key, values=[value], datatype=1)
                    param_list.append(dim_record)
        for v in list(self.parameters.data_vars):
            darray = self.parameters[v]
            vals = darray.values
            vdtype = get_prms_dtype(vals.dtype, 'numpy')
            dims = darray.dims
            recorddims = []
            for d in range(len(dims)):
                rdim = [dims[d], vals.shape[d]]
                recorddims.append(rdim)

            # assume everything coming from PRMSparameters class is C ordered, translate to Fortran ordered for
            #   pygsflow object
            recorddims.reverse()
            record = ParameterRecord(
                v,
                vals.ravel(),
                dimensions=recorddims,
                datatype=vdtype
            )
            param_list.append(record)

        return pygsflowparams(param_list)

    @staticmethod
    def load_pygsflow_obj(param_obj: pygsflowparams):
        newparams = PRMSParameters()
        for rec in param_obj.parameters_list:
            newparams.add_pygsflow_record(rec)

        return newparams

    @staticmethod
    def load_paramfile(paramfiles: Union[str, Path, list]):
        pgsparams = pygsflowparams.load_from_file(paramfiles)
        inparams = PRMSParameters.load_pygsflow_obj(pgsparams)

        return inparams

    @staticmethod
    def load_netcdf(ncfiles: Union[str, Path, list]):
        if isinstance(ncfiles, (str, Path)):
            params = xr.open_dataset(ncfiles)
        elif isinstance(ncfiles, list):
            param_lst = []
            for f in ncfiles:
                ds = xr.open_dataset(f)
                param_lst.append(ds)
            params = xr.merge(param_lst)
        else:
            raise ValueError("The input file path(s) were not pathlike or list of paths.")

        return PRMSParameters(params)

class PRMSCascades(PRMSParameters):

    def __init__(self,
                 modelgrid: Union[str, Path, PRMSVertexGrid, FlopyVertexGrid],
                 elev_unit: Optional[str] = 'meters',
                 hru_types: Optional[np.ndarray] = None):
        super().__init__(grid=modelgrid)
        self.fdir = None
        self._elevation = None
        self._stream_grid = None
        self.segments = None
        self.elev_units = elev_unit

        if isinstance(modelgrid, (PRMSVertexGrid, FlopyVertexGrid)):
            self._flopygrid = modelgrid
        elif isinstance(modelgrid, (str, Path)):
            self._flopygrid = PRMSVertexGrid(modelgrid)
        else:
            raise ValueError("The input grid type is not recognized")

        if hru_types is None:
            self.hru_types = np.ones((len(self.grid),), dtype=int)
        else:
            self.hru_types = hru_types

        self.add_parameter(
            "hru_subbasin",
            np.ones((len(self.grid),), dtype=int),
            len(self.grid),
            'nhru'
        )

        self._grid_lat_lon = self.grid.centroid.to_crs(4326)
        self.add_parameter(
            "hru_lat",
            self._grid_lat_lon.geometry.y.values,
            len(self.grid),
            'nhru'
        )
        self.add_parameter(
            "hru_lon",
            self._grid_lat_lon.geometry.x.values,
            len(self.grid),
            'nhru'
        )

    @property
    def hru_types(self):
        return self._hru_types

    @hru_types.setter
    def hru_types(self, type_array: np.ndarray):
        if len(type_array) == len(self.grid):
            self._hru_types = type_array
            if 'hru_type' not in list(self.parameters.data_vars):
                self.add_parameter(
                    "hru_type",
                    self._hru_types,
                    len(self.grid),
                    'nhru'
                )
            else:
                self.parameters['hru_type'].values = self._hru_types
        else:
            raise ValueError("The input hru_type array does not match the number of grid cells in the grid")

    def map_hru_type(self, grid_features: gpd.GeoDataFrame):
        type_dict = {
            'inactive': 0,
            'basin': 1,
            'stream': 1,
            'lake': 2,
            'swale': 3,
            'glacier': 4
        }
        gridcell_types = np.ones(len(self.grid), dtype=int)
        lake_hrus = np.zeros(len(self.grid), dtype=int)
        for i, r in grid_features.iterrows():
            if (r['region'] == 'basin') | (r['region'] == 'stream'):
                continue
            msk = self.grid.intersects(r.geometry, align=True).values
            gridcell_types[msk] = type_dict[r['region']]
            if r['region'] == 'lake':
                lake_hrus[msk] = r['regID'] + 1

        self.add_parameter(
            "lake_hru_id",
            lake_hrus,
            len(self.grid),
            'nhru'
        )

        self.hru_types = gridcell_types

    @property
    def stream_grid(self):
        return self._stream_grid

    @stream_grid.setter
    def stream_grid(self, strm_array: np.ndarray):
        if len(strm_array) == len(self.grid):
            self._stream_grid = strm_array
            self._strm_ids = np.where(self._stream_grid != 0)[0]

    def streams_from_hydrography(self,
                                 streams: gpd.GeoDataFrame,
                                 segment_ids: bool = False,
                                 overlap: str = 'last'):
        strm_mask = np.zeros(len(self.grid), dtype=int)
        if not segment_ids:
            s_intx = []
            for geom in streams.geometry.values:
                strmsk = self.grid.intersects(geom, align=True).values
                s_intx.extend(list(self.grid.index[strmsk].values))

            strm_mask[s_intx] = 1
        else:
            if overlap == 'last':
                for i in range(len(streams)):
                    strmsk = self.grid.intersects(streams.iloc[i, :].geometry, align=True).values
                    strm_mask[strmsk] = streams.index[i] + 1
            elif overlap == 'first':
                for i in range(len(streams)):
                    strmsk = self.grid.intersects(streams.iloc[i, :].geometry, align=True).values
                    segm = strm_mask[self.grid.index[strmsk].values]
                    if (segm != 0).any():
                        pseg = np.where(segm == 0)
                        strm_mask[pseg] = streams.index[i] + 1
                    else:
                        strm_mask[segm] = streams.index[i] + 1
            else:
                raise ValueError("The overlap argument is not recognized, choose either 'first' or 'last'")

        self.stream_grid = strm_mask
        segids = np.unique(self._stream_grid[self._strm_ids])
        seggeoms = streams.loc[segids - 1, ['geometry']]
        seggeoms['segmentids'] = segids
        self.segments = seggeoms

    @property
    def hru_elevations(self):
        return self._elevation

    @hru_elevations.setter
    def hru_elevations(self, elevs: np.ndarray):
        if len(elevs) == len(self.grid):
            self._elevation = elevs
            if 'hru_elev' not in list(self.parameters.data_vars):
                self.add_parameter(
                    "hru_elev",
                    self._elevation,
                    len(self.grid),
                    'nhru'
                )
            else:
                self.parameters['hru_elev'].values = self._elevation
        else:
            raise ValueError("The input hru_type array does not match the number of grid cells in the grid")

    @property
    def elev_units(self):
        return self._elev_units

    @elev_units.setter
    def elev_units(self, unit: str):
        e_units_dict = {
            'meters': 1,
            'feet': 0
        }
        if unit not in ['meters', 'feet']:
            raise ValueError("The input units are not recognized or are mis-spelled")
        else:
            self._elev_units = (unit, e_units_dict[unit])
            if 'elev_units' not in list(self.parameters.data_vars):
                self.add_parameter(
                    'elev_units',
                    np.array([self._elev_units[1]]),
                    1,
                    'one'
                )
            else:
                self.parameters['elev_units'].values = np.array([self._elev_units[1]])

    def map_flow_directions(self, elev: Optional[np.ndarray] = None):
        if self._flopygrid is None:
            raise AttributeError(
                "No flopy grid is assigned, to map flow directions using d-any or pygsflow, a flopy grid must be assigned.")

        if (elev is None) & (self._elevation is None):
            raise ValueError(
                "Elevation data has not been added and no elevations were passed to the elev argument. Flow Directions cannot be calculated.")
        elif (elev is None) & (self._elevation is not None):
            pass
        elif (elev is not None) & (self._elevation is None):
            self.hru_elevations = elev
        elif (elev is not None) & (self._elevation is not None):
            msg = "Elevation data was input as an argument but elevation data is already mapped to the grid, existing elevations will be overwritten"
            user_warning(
                msg,
                inspect.getframeinfo(
                    inspect.currentframe()
                ),
            )
            self.hru_elevations = elev

        self.fdir = dany.FlowDirections(self._flopygrid, self._elevation)
        self.fdir.flow_directions()
        self.fdir.flow_accumulation()
        # assume meters for projected crs...probably force everything to 5071 in the future
        self.add_parameter(
            "hru_area",
            self.fdir.area / 4046.86,
            len(self.grid),
            'nhru'
        )
        self.add_parameter(
            "hru_aspect",
            self.fdir.aspect,
            len(self.grid),
            'nhru'
        )
        self.add_parameter(
            "hru_slope",
            self.fdir.slope,
            len(self.grid),
            'nhru'
        )

    def cascades_from_hydrography(self,
                                  streams: Optional[gpd.GeoDataFrame] = None,
                                  lakes: Optional[gpd.GeoDataFrame] = None,
                                  swales: Optional[gpd.GeoDataFrame] = None):
        if self.fdir is None:
            if self._elevation is None:
                raise AttributeError(
                    "Flow directions have not been mapped to the grid and there is no elevation data assigned. Run map_flow_directions() or assign hru_elevation attribute before mapping cascades.")
            else:
                self.map_flow_directions()

        if self._stream_grid is None:
            if streams is None:
                raise AttributeError(
                    "Streams have not been delineated for the grid and no stream geometries were provided as an argument")
            else:
                self.streams_from_hydrography(streams, segment_ids=True)
        # use dany stream object but edit to be more like CRT results (stream cells are not included, a cascade link ends at hydrographic feature)
        stmobj = dany.PrmsStreams(self._flopygrid, self.fdir)
        casc = stmobj.get_cascades(self.stream_grid)
        hru_up, hru_dwn, hru_pct, hru_seg_id = casc
        stmovlp = np.where(np.isin(hru_up - 1, self._strm_ids))[0]
        hru_up = np.delete(hru_up, stmovlp)
        hru_dwn = np.delete(hru_dwn, stmovlp)
        hru_pct = np.delete(hru_pct, stmovlp)

        segids = np.unique(self._stream_grid[self._strm_ids])
        segtype = np.zeros(segids.shape)
        # same process but removing cascades within lakes or swales
        if lakes is not None:
            lakes['region'] = 'lake'
            lakes['regID'] = lakes.index.astype(int)

        if swales is not None:
            swales['region'] = 'swale'
            swales['regID'] = swales.index.astype(int)

        if (lakes is not None) & (swales is not None):
            snkfeat = pd.concat([lakes, swales], ignore_index=True)
            self.map_hru_type(snkfeat)
        elif (lakes is not None) & (swales is None):
            snkfeat = lakes
            self.map_hru_type(snkfeat)
        elif (lakes is None) & (swales is not None):
            snkfeat = swales
            self.map_hru_type(snkfeat)
        else:
            pass

        if np.isin(np.array([2, 3]), np.unique(self._hru_types)).any():
            snk_ids = np.where(np.isin(self._hru_types, np.array([2, 3])))[0]
            snkovlp = np.where(np.isin(hru_up - 1, snk_ids))[0]
            hru_up = np.delete(hru_up, snkovlp)
            hru_dwn = np.delete(hru_dwn, snkovlp)
            hru_pct = np.delete(hru_pct, snkovlp)
        # set stream segment parameters - lake_segment_id and segment_type with lakes
        if np.isin(2, np.unique(self._hru_types)):
            lake_segid = segtype.copy()
            lk_ids = self.parameters["lake_hru_id"].values
            for i in np.unique(lk_ids):
                lk = np.where(lk_ids == i)[0]
                strmlakeids = np.unique(self._stream_grid[lk])
                lake_segid = np.where(np.isin(segids, strmlakeids), i, lake_segid)

            lksegtypeids = np.unique(self._stream_grid[np.where(self._hru_types == 2)[0]])
            segtype = np.where(np.isin(segids, lksegtypeids), 2, segtype)

            self.add_parameter(
                "lake_segment_id",
                lake_segid,
                len(segids),
                'nsegment'
            )

            self.add_parameter(
                "segment_type",
                segtype,
                len(segids),
                'nsegment'
            )

        else:
            # set stream segment parameters segment_type with no lakes
            self.add_parameter(
                "segment_type",
                segtype,
                len(segids),
                'nsegment'
            )

        hru_seg_id = self._stream_grid.copy()[hru_dwn - 1]

        # set stream segment parameters - tosegment; *FUTURE* segment_outflow_id
        # currently this looks for unsuccessful stream segment routing...but doesn't do anything with it, could be used
        # to correct any sinks in the stream network...did not run into that problem with the development test case
        nabs = self._flopygrid.neighbors(method='queen')
        mine = []
        for seg in segids:
            idx = np.where(self._stream_grid == seg)[0]
            me = self._elevation[idx].min()
            mine.append(me)
        seg_elev = np.array(mine)

        sneighb = []
        sorted = []
        dwn_seg = []
        for seg in segids:
            idx = np.where(self._stream_grid == seg)[0]
            allnabs = []
            for i in idx:
                nl = nabs[i]
                allnabs.extend(nl)
            alnb_array = np.array(allnabs)
            stmnbs = self._stream_grid[alnb_array]
            snb = stmnbs[(stmnbs != 0)]
            snb = np.unique(snb[snb != seg])
            if snb.size == 1:
                if seg_elev[np.where(segids == seg)] < seg_elev[np.where(segids == snb[0])]:
                    sorted.append(1)
                    dwn_seg.append(0)
                elif seg_elev[np.where(segids == seg)] > seg_elev[np.where(segids == snb[0])]:
                    sorted.append(1)
                    dwn_seg.append(snb[0])
                else:
                    sorted.append(0)
                    dwn_seg.append(-1)
            else:
                snb_srt = np.argsort(seg_elev[np.where(np.isin(segids, snb))[0]])
                snb = snb[snb_srt]
                if np.unique(snb).size == snb.size:
                    sorted.append(1)
                    dwn_seg.append(snb[0])
                else:
                    sorted.append(0)
                    dwn_seg.append(-1)
            sneighb.append(snb.tolist())

        strm_con = dict(zip(segids.tolist(), sneighb))
        dwn_sg_id = np.array(dwn_seg)
        sg_srt = np.array(sorted)

        self._stream_graph = strm_con
        self._strm_sort = sg_srt

        self.add_parameter(
            "tosegment",
            dwn_sg_id,
            len(segids),
            'nsegment'
        )

        self.add_parameter(
            'hru_up_id',
            hru_up,
            len(hru_up),
            'ncascade'
        )

        self.add_parameter(
            'hru_down_id',
            hru_dwn,
            len(hru_dwn),
            'ncascade'
        )

        self.add_parameter(
            'hru_pct_up',
            hru_pct,
            len(hru_pct),
            'ncascade'
        )

        self.add_parameter(
            'hru_strmseg_down_id',
            hru_seg_id,
            len(hru_seg_id),
            'ncascade'
        )

    def groundwater_cascades(self, gw_dem: Optional[np.ndarray] = None):
        if gw_dem is None:
            cascade_params = np.array(['hru_up_id', 'hru_down_id', 'hru_pct_up', 'hru_strmseg_down_id'])
            if not np.isin(cascade_params, np.array(list(self.parameters.data_vars))).all():
                raise AttributeError(
                    "Cascade parameters were not found in the parameter Dataset and no GW map for routing was provided, compute cascades first for default groundwater cascades")

            self.add_parameter(
                'gw_up_id',
                self.parameters['hru_up_id'].values,
                len(self.parameters['hru_up_id'].values),
                'ncascdgw'
            )

            self.add_parameter(
                'gw_down_id',
                self.parameters['hru_down_id'].values,
                len(self.parameters['hru_down_id'].values),
                'ncascdgw'
            )

            self.add_parameter(
                'gw_pct_up',
                self.parameters['hru_pct_up'].values,
                len(self.parameters['hru_pct_up'].values),
                'ncascdgw'
            )

            self.add_parameter(
                'gw_strmseg_down_id',
                self.parameters['hru_strmseg_down_id'].values,
                len(self.parameters['hru_strmseg_down_id'].values),
                'ncascdgw'
            )
        else:
            raise NotImplementedError("GW separate routing has not been implemented yet.")

    def map_subbasins(self, pour_pnts: gpd.GeoDataFrame):
        if self.fdir is None:
            raise AttributeError(
                "Flow directions have not been mapped to the grid. Run map_flow_directions() or compute cascades before delineating subbasins.")
        else:
            sub_array = self.fdir.get_subbasins(pour_pnts.geometry.values)
            self.parameters['hru_subbasin'].values = sub_array
            subbasins = np.unique(sub_array)
            subs_dwn = []
            for p in pour_pnts.geometry.values:
                pnt_id = self.grid.intersects(p).values
                gpnt = self.grid[pnt_id].index.values
                pfd = self.fdir.flow_direction_array[gpnt]
                if gpnt == pfd:
                    subs_dwn.append(0)
                else:
                    dwn_sub = int(sub_array[pfd][0])
                    subs_dwn.append(dwn_sub)
            subs_dwn = np.array(subs_dwn)

            self.add_parameter(
                'subbasin_down',
                subs_dwn,
                len(subbasins),
                'nsub'
            )

    def export_cascades_grid(self, filename: Union[str, Path], add_params: Optional[list] = None,
                             flow_directions: bool = False, flow_accum: bool = False, sinks: bool = False):
        out = self.grid.copy()
        if add_params is not None:
            for i in add_params:
                out[i] = self.parameters[i].values

        if flow_directions:
            out['flow_directions'] = self.fdir.flow_direction_array

        if flow_accum:
            out['flow_accum'] = self.fdir.flow_accumulation_array

        if sinks:
            out['sinks'] = find_undeclared_sinks(self.fdir.flow_direction_array)

        out.to_file(filename)

    def export_hru_cascade_links(self, filename: Union[str, Path]):
        cascade_params = np.array(['hru_up_id', 'hru_down_id', 'hru_pct_up', 'hru_strmseg_down_id'])
        if not np.isin(cascade_params, np.array(list(self.parameters.data_vars))).all():
            raise AttributeError("Cascade parameters were not found in the parameter Dataset")
        up = self.grid.loc[self.parameters['hru_up_id'].values - 1, :].centroid
        dwn = self.grid.loc[self.parameters['hru_down_id'].values - 1, :].centroid
        casc_lines = [LineString([p1, p2]) for p1, p2 in zip(up, dwn)]
        casgdf = gpd.GeoDataFrame(
            {'hru_up': self.parameters['hru_up_id'].values,
             'hru_dwn': self.parameters['hru_down_id'].values,
             'hru_pct_up': self.parameters['hru_pct_up'].values,
             'hru_seg_dwn': self.parameters['hru_strmseg_down_id'].values},
            geometry=casc_lines,
            crs=5071)
        casgdf.to_file(filename)

    def export_stream_segments(self, filename: Union[str, Path]):
        stm_out = self.segments.copy()
        if 'tosegment' in list(self.parameters.data_vars):
            stm_out['tosegment'] = self.parameters['tosegment'].values
        if 'segment_type' in list(self.parameters.data_vars):
            stm_out['segment_type'] = self.parameters['segment_type'].values
        if 'lake_segment_id' in list(self.parameters.data_vars):
            stm_out['lake_segment_id'] = self.parameters['lake_segment_id'].values

        stm_out.to_file(filename)


def join_pygsflow_params(param_objs: list, separate_files: bool = False, file_names: Optional[list] = None):
    joined_lst = []
    if not separate_files:

        for p in param_objs:
            for r in p.parameters_list:
                if r in joined_lst:
                    continue
                else:
                    joined_lst.append(r)
    else:
        if file_names is None:
            msg = (
                "No filenames were provided, using defaults. "
                f"Defaults saved to '{os.getcwd()}'"
            )
            user_warning(
                msg,
                inspect.getframeinfo(
                    inspect.currentframe()
                ),
            )
            file_names = [f'paramfile_0{n}.params' for n in range(len(param_objs))]
        else:
            if len(file_names) != len(param_objs):
                raise ValueError(
                    "The length of the input filenames list does not match the number of input parameter objects.")
            pass

        for i, p in enumerate(param_objs):
            for r in p.parameters_list:
                if r in joined_lst:
                    continue
                else:
                    r.file_name = file_names[i]
                    joined_lst.append(r)

    return pygsflowparams(joined_lst)


def validate_param_dset(param_dset: xr.Dataset) -> bool:
    dims = list(param_dset.sizes.keys())
    vars = list(param_dset.data_vars)
    if any(x not in list(meta.dimensions.keys()) for x in dims):
        vald = False
    else:
        vald = True
    if any(x not in list(meta.parameters.keys()) for x in vars):
        vald = False
    else:
        vald = True

    return vald


def get_prms_dtype(dtype, platform):
    for key, item in prms_dtypes.items():
        if platform == 'numpy':
            if (key == 'integer') or (key == 'string'):
                if np.issubdtype(dtype, item[platform]):
                    outtype = item['PRMS']
                    return outtype
            else:
                if np.isdtype(dtype, item[platform]):
                    outtype = item['PRMS']
                    return outtype
        elif platform != 'numpy':
                if dtype == item[platform]:
                    outtype = item['PRMS']
                    return outtype
        else:
            raise ValueError("The input dtype could not be resolved to a prms dtype code.")