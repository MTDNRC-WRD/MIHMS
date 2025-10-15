import os
from pathlib import Path
import copy
import warnings
import inspect
from typing import Union, Optional


import geopandas as gpd
import pandas as pd
import rasterio as rio
import xarray as xr
import numpy as np
from flopy.discretization import VertexGrid as FlopyVertexGrid
from GRIDtools.zonal import calc_zonal_stats
from gsflow.prms import ParameterRecord
from gsflow import PrmsParameters as pygsflowparams
from gsflow.builder import builder_utils as bu
import dany
from shapely.geometry import LineString

from mihms.config import prep
from mihms.prep.prms.domain import PRMSVertexGrid, find_undeclared_sinks
from mihms.prep.prms import meta
from mihms.utils.io import user_warning
from mihms.prep.utils import convert_pywatershed_dims, replace_array_values

prms_dtypes = {
    'integer': {'PRMS': 1, 'numpy': np.integer, 'netcdf': 'i4', 'python': int},
    'real': {'PRMS': 2, 'numpy': np.float32, 'netcdf': 'f4', 'python': float},
    'double': {'PRMS': 2, 'numpy': np.float64, 'netcdf': 'f8', 'python': float},
    'string': {'PRMS': 4, 'numpy': str, 'netcdf': 'S1', 'python': str}
}

# This is to provided dimension checks that incorporate PRMS flexible parameter specifications
# (nhru, nmonths) is the max-dimensional space but can be specified in the paramter file as a smaller dimensional space
# (ie, if dimension 'one' is used, the single value is expanded to the max-dimensional space within PRMS)
dims_remap = {
    ('one',): [('one',)],
    ('four',): [('four',)],
    ('ndays',): [('ndays',)],
    ('nglres',): [('nglres',)],
    ('nlapse',): [('nlapse',)],
    ('nmonths',): [('nmonths',)],
    ('seven',): [('seven',)],
    ('ncbh',): [('ncbh',)],
    ('nconsumed',): [('nconsumed',)],
    ('nevap',): [('nevap',)],
    ('nexternal',): [('nexternal',)],
    ('nhumid',): [('nhumid',)],
    ('nlakeelev',): [('nlakeelev',)],
    ('nmap',): [('nmap',)],
    ('nmap2hru',): [('nmap2hru',)],
    ('nobs',): [('nobs',)],
    ('npoigages',): [('npoigages',)],
    ('nrain',): [('nrain',)],
    ('nratetbl',): [('nratetbl',)],
    ('nsnow',): [('nsnow',)],
    ('nsol',): [('nsol',)],
    ('nstreamtemp',): [('nstreamtemp',)],
    ('ntemp',): [('ntemp',)],
    ('nwateruse',): [('nwateruse',)],
    ('nwind',): [('nwind',)],
    ('ncascade',): [('ncascade',)],
    ('ncascdgw',): [('ncascdgw',)],
    ('ndepl',): [('ndepl',)],
    ('ndeplval',): [('ndeplval',)],
    ('mxnsos',): [('mxnsos',)],
    ('ngate',): [('ngate',)],
    ('ngate2',): [('ngate2',)],
    ('ngate3',): [('ngate3',)],
    ('ngate4',): [('ngate4',)],
    ('nstage',): [('nstage',)],
    ('nstage2',): [('nstage2',)],
    ('nstage3',): [('nstage3',)],
    ('nstage4',): [('nstage4',)],
    ('ngwcell',): [('ngwcell',)],
    ('nhrucell',): [('nhrucell',)],
    ('nlake',): [('nlake',)],
    ('nlake_hrus',): [('nlake_hrus',)],
    ('nsegment',): [('nsegment',)],
    # these combos are confirmed by PRMS documentation, there may be more that work for other paramters?
    # This one is special...pywatershed treats a parameter as ('nmonths', 'nlake') which is not the same as the
    #   PRMS documentation which has this at 'nhru'
    ('nmonths', 'nlake'): [('one',), ('nlake',), ('nmonths',), ('nmonths', 'nlake')],
    ('ngw',): [('ngw',), ('one',), ('nsub',)],
    ('nmonths', 'ngw'): [('one',), ('ngw',), ('nmonths',), ('nsub',), ('nmonths', 'nsub'), ('nmonths', 'ngw')],
    ('nhru',): [('nhru',), ('one',), ('nsub',)],
    ('nmonths', 'nhru'): [('one',), ('nhru',), ('nmonths',), ('nsub',), ('nmonths', 'nsub'), ('nmonths', 'nhru')],
    ('nssr',): [('nssr',), ('one',), ('nsub',)],
    ('nmonths', 'nssr'): [('one',), ('nssr',), ('nmonths',), ('nsub',), ('nmonths', 'nsub'), ('nmonths', 'nssr')],
    ('nsub',): [('one',), ('nsub',)]
}


class PRMSParameters:

    def __init__(self, param_ds: Optional[xr.Dataset] = None,
                 grid: Optional[Union[str, Path, PRMSVertexGrid, FlopyVertexGrid, gpd.GeoDataFrame]] = None):

        self.parameters = param_ds
        self._grid = None
        self._flopygrid = None

        if grid is not None:
            self.grid = grid

        self._pygsflow_dimrecs = None

        self._from_files = []


    @property
    def grid(self):
        return self._grid

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
                msg = " ".join(
                        [
                            f"There are no parameters specified yet and or no nhru dimension to check grid.",
                            "Assigning grid anyway, make sure to check any inserted parameters match the grid size."
                        ]
                    )
                user_warning(
                    msg,
                    inspect.getframeinfo(
                        inspect.currentframe()
                    ),
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
                msg = " ".join(
                        [
                            f"There are no parameters specified yet and or no nhru dimension to check grid.",
                            "Assigning grid anyway, make sure to check any inserted parameters match the grid size."
                        ]
                    )
                user_warning(
                    msg,
                    inspect.getframeinfo(
                        inspect.currentframe()
                    ),
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
                msg = " ".join(
                        [
                            f"There are no parameters specified yet and or no nhru dimension to check grid.",
                            "Assigning grid anyway, make sure to check any inserted parameters match the grid size."
                        ])
                user_warning(
                    msg,
                    inspect.getframeinfo(
                        inspect.currentframe()
                    ),
                )
                self._grid = ingrid

        else:
            raise ValueError("The input grid object type is not supported.")

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, in_ds: Union[xr.Dataset, None]):
        if in_ds is None:
            self._parameters = xr.Dataset()
        elif (in_ds is not None) & (validate_param_dset(in_ds)):
            # check all paramter types, apparently xarray can save/load weird types to netcdf
            for var in list(in_ds.data_vars):
                pchk, ptyp = self._check_param_dtype(var, in_ds[var].values)
                if pchk:
                    pass
                else:
                    msg = (
                        f"The datatype of variable {var} does not match the required datatype of the specified paramater, "
                        f"changing the datatype to {ptyp}.")
                    user_warning(
                        msg,
                        inspect.getframeinfo(
                            inspect.currentframe()
                        ),
                    )
                    in_ds[var].values = in_ds[var].values.astype(ptyp)
            self._parameters = in_ds
        else:
            raise ValueError("The input xarray dataset contains invalid parameters or dimensions.")

    # pygsflow param object has to include dimensions section as well
    @property
    def pygsflow_param_obj(self):
        param_list = []
        if len(list(self.parameters.data_vars)) == 0:
            raise ValueError("There are no parameters specified.")
        if self._pygsflow_dimrecs is None:
            for key, value in self.parameters.sizes.items():
                if len(self._from_files) > 0:
                    dim_record = ParameterRecord(name=key, values=[value], datatype=1, file_name=self._from_files[0])
                else:
                    dim_record = ParameterRecord(name=key, values=[value], datatype=1)
                param_list.append(dim_record)
        else:
            param_list = param_list + self._pygsflow_dimrecs
            for key, value in self.parameters.sizes.items():
                if key in [x.name for x in self._pygsflow_dimrecs]:
                    continue
                else:
                    if len(self._from_files) > 0:
                        dim_record = ParameterRecord(name=key, values=[value], datatype=1,
                                                     file_name=self._from_files[0])
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
            if 'file_name' in self.parameters[v].attrs.keys():
                record = ParameterRecord(
                    v,
                    vals.ravel(),
                    dimensions=recorddims,
                    datatype=vdtype,
                    file_name=self.parameters[v].attrs['file_name']
                )
            elif ('file_name' not in self.parameters[v].attrs.keys()) & (len(self._from_files) > 0):
                record = ParameterRecord(
                    v,
                    vals.ravel(),
                    dimensions=recorddims,
                    datatype=vdtype,
                    file_name=self._from_files[0]
                )
            else:
                record = ParameterRecord(
                    v,
                    vals.ravel(),
                    dimensions=recorddims,
                    datatype=vdtype
                )
            param_list.append(record)

        return pygsflowparams(param_list)

    def add_parameter_from_raster(self, param_name: str,
                                  dim_names: Union[str, list],
                                  in_raster: Union[str, Path, rio.DatasetReader, xr.DataArray, xr.Dataset],
                                  stat: str,
                                  **kwargs):
        if self.grid is None:
            raise RuntimeError(
                "No PRMSParameters.grid was specified. Spatial parameter mapping functions are not accessible without a grid object.")

        zonal = calc_zonal_stats(self.grid, in_raster, stats=stat, **kwargs)

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
                      values: np.ndarray,
                      dims: Union[int, list],
                      dim_names: Union[str, list],
                      add_attrs: Optional[dict] = None,
                      arr_format: str = 'C'):

        # Check available parameters and dimensions to see if they are valid.
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

        # Use metadata for parameters to check datatype
        # Current behavior is to adjust if it does not conform, and give user warning
        attrs_m = meta.find_variables(param_name)[param_name]
        metadims = convert_pywatershed_dims(attrs_m['dims'])
        pchk, ptyp = self._check_param_dtype(param_name, values)
        if pchk:
            pass
        else:
            msg = (f"The datatype of the input values does not match the required datatype of the specified paramater, "
                   f"changing the datatype to {ptyp} for the {param_name} paramter")
            user_warning(
                msg,
                inspect.getframeinfo(
                    inspect.currentframe()
                ),
            )
            values = values.astype(ptyp)

        if param_name in list(self.parameters.data_vars):
            msg = " ".join(
                    [
                        f"The parameter already exists, overwriting the existing '{param_name}' parameter."
                    ])
            user_warning(
                msg,
                inspect.getframeinfo(
                    inspect.currentframe()
                ),
            )
            attrs = self.parameters[param_name].attrs
            if 'requires' in attrs:
                del attrs['requires']
            if add_attrs is not None:
                attrs.update(add_attrs)
        else:
            attrs = copy.deepcopy(attrs_m)
            if 'requires' in attrs:
                del attrs['requires']
            if add_attrs is not None:
                attrs.update(add_attrs)

        if (len(dim_names) > 1) & (arr_format == 'F'):
            if (dim_names[-1], dim_names[0]) not in dims_remap[metadims]:
                raise ValueError("The specified dimensions are not available for the specified parameter, check"
                                 "dimensions names.")
            if values.shape != (dims[-1], dims[0]):
                values = np.reshape(values, (dims[-1], dims[0]))
            self.parameters[param_name] = (
            (dim_names[-1], dim_names[0]), values, attrs)
        else:
            if tuple(dim_names) not in dims_remap[metadims]:
                raise ValueError("The specified dimensions are not available for the specified parameter, check"
                                 "dimensions names or order.")
            if values.shape != tuple(dims):
                values = np.reshape(values, tuple(dims))
            self.parameters[param_name] = (tuple(dim_names), values, attrs)

    def add_pygsflow_record(self, record: ParameterRecord):
        if not isinstance(record, ParameterRecord):
            raise ValueError("The input record is not a pygsflow ParameterRecord object.")
        if record.file_name is not None:
            if record.file_name not in self._from_files:
                self._from_files.append(record.file_name)
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

    def set_values(self, param_name: str, newvals: np.ndarray):
        if self._parameters[param_name].values.shape != newvals.shape:
            raise ValueError(f"The dimensions of the input values do not match the dimensions of {param_name} dimensions. "
                             f"{param_name} dimensions are {self._parameters[param_name].values.shape}, input values "
                             f"are {newvals.shape}.")

        self._parameters[param_name].values = newvals

    def select_parameters(self, params: Union[str, list, np.ndarray]) -> Union[xr.DataArray, xr.Dataset]:
        if isinstance(params, str):
            return self._parameters[params]
        else:
            if isinstance(params, np.ndarray):
                params = params.tolist()

            keepcoords = list(self._parameters.coords)
            sel = keepcoords + params

            return self._parameters[sel]

    def merge_parameters(self, paramob):
        if isinstance(paramob, PRMSParameters):
            merged = xr.merge([self._parameters, paramob.parameters])
        elif isinstance(paramob, xr.Dataset):
            merged = xr.merge([self._parameters, paramob])
        elif isinstance(paramob, list):
            ob_list = []
            for po in paramob:
                if isinstance(po, PRMSParameters):
                    ob_list.append(po.parameters)
                elif isinstance(po, xr.Dataset):
                    ob_list.append(po)
                else:
                    raise ValueError("Item in input list is neither a PRMSParameter object nor xarray Dataset.")
            merged = xr.merge([self._parameters]+ob_list)
        else:
            raise ValueError("Input is not recognized as either PRMSParameter object or xarray Dataset.")
        self.parameters = merged

    def remove_parameters(self, params: Union[str, list, np.ndarray]):
        if isinstance(params, str):
            params = [params]

        keys = [p for p in params if p in list(self._parameters.data_vars)]
        self.parameters = self._parameters.drop_vars(keys)

    # Haven't implemented time_idx selection for multi-dimensional parameters...but most calibration params
    #   are only mapped to HRU with no time dimension.
    # TODO: add time indexing functionality
    def change_parameter_by_subbasin(self,
                                     name: str,
                                     values: Union[int, float, np.ndarray],
                                     subbasin_idx: Union[int, np.ndarray],
                                     operation: str = 'replace',
                                     time_idx: Optional[Union[int, np.ndarray]] = None,
                                     subbasins: Optional[np.ndarray] = None):
        """

        Changes a parameter value(s) to the specified value(s). The values are mapped to the parameter based on,
        sub-basins specified in the PRMS domain. This allows mapping new values or altering values basin on sub-basin
        ID's, such that the parameter is altered for all HRU's with that sub-basin ID.

        Args:
            name:
                The name of the parameter to change. For this function only paramters with 'nhru', 'ngw', or 'nssr'
                are supported. Segments and Lakes cannot be changed by sub-basin.
            values: int | float | np.ndarray
                The value(s) to set as the new parameter values. Values argument must be the same dimensions as the
                sub_id argument .
            subbasin_idx:
                PRMS sub-basin ID to map values to.
            operation:
                Determines the method that the input values are applied to the parameter.
                    - 'replace' applies the input value(s) directly
                    - 'multiply' multiplies the existing parameter values by the input value(s)
                    - 'add' adds the input value(s) to the existing parameter values
            time_idx:
                The ID of the time dimension if the parameter has one. Usually 'nmonths' - integer 1-12.
            subbasins: Optional
                An array of the PRMS parameter 'hru_subbasin' which is used for selecting the HRU's for parameter
                assignment. If this parameter exists in the parameters property of the class then it is not needed, if
                it does not, it will need to be passed to the function using this argument.

        Returns: None
            Adjusts the parameter values.
        """
        if name not in list(self._parameters.data_vars):
            raise KeyError(f"{name} parameter was not found in the parameter set, or is invalid.")

        pdim = self._parameters[name].dims
        cdim = None
        for d in pdim:
            if d in ['nhru', 'ngw', 'nssr', 'one']:
                cdim = d
        if cdim is None:
            raise NotImplementedError(f"The {name} parameter's dimension is not compatible with assignment by subbasin.")

        if 'hru_subbasin' in list(self._parameters.data_vars):
            subbasins = self.parameters['hru_subbasin'].values
        else:
            if subbasins is None:
                raise ValueError("The 'hru_subbasin' parameter does not exist in the parameter object and none was"
                                 " provided in the subbasins argument.")

        if isinstance(values, np.ndarray):
            if not isinstance(subbasin_idx, np.ndarray):
                raise ValueError("When multiple values are passed, subbasin_idx must be an equal length numpy array.")
            if len(values) != len(subbasin_idx):
                raise ValueError("The values and subbasin_idx arguments are not the same length.")
            for i, v in enumerate(values):
                idvals = np.where(subbasins == subbasin_idx[i])[0]
                self.change_parameter(name, v, cdim, operation=operation, idx={cdim: idvals})
        else:
            if isinstance(subbasin_idx, np.ndarray):
                if len(subbasin_idx) > 1:
                    idvals = np.argwhere(np.isin(subbasins, subbasin_idx)).ravel()
                    self.change_parameter(name, values, cdim, operation=operation, idx={cdim: idvals})
            else:
                idvals = np.where(subbasins == subbasin_idx)[0]
                self.change_parameter(name, values, cdim, operation=operation, idx={cdim: idvals})
            

    def change_parameter(self,
                         name: str,
                         values: Union[int, float, np.ndarray],
                         dim: Union[str, tuple] = 'one',
                         operation: str = 'replace',
                         idx: Optional[dict] = None):
        """

        Changes a parameter value(s) to the specified value(s). The values are mapped to the parameter based on user
        specified dimensions, array indices, or both. This allows mapping of the full parameter dimensional space,
        or just one of many dimensions.

        Args:
            name: str
                The name of the parameter to change.
            values: int | float | np.ndarray
                The value(s) to set as the new parameter values. These must be specified in coordination with the
                dim argument, meaning it must conform to a compatible shape provided the given dimensions.
            dim: str | tuple
                A string or tuple of strings that determine what shape the input values should be mapped as. If only
                one value is provided, it will be expanded to these dimensions.
            operation: str
                Determines the method that the input values are applied to the parameter.
                    - 'replace' applies the input value(s) directly
                    - 'multiply' multiplies the existing parameter values by the input value(s)
                    - 'add' adds the input value(s) to the existing parameter values
            idx: Optional dict
                A dictionary of array indices to map the specified value(s) to. Each keyword in the dictionary must
                match a dimensions provided in the dim argument.

        Returns: None
            Adjusts the parameter values.
        """
        if name not in list(self._parameters.data_vars):
            raise KeyError(f"{name} parameter was not found in the parameter set, or is invalid.")

        par_dims = self._parameters[name].dims
        par_shp = self._parameters[name].shape

        if isinstance(dim, str):
            dim = (dim,)

        if (dim == ('one',)) & (par_dims == ('one',)):
            if operation == 'replace':
                self._parameters[name].values = np.ones(par_shp) * values
            elif operation == 'multiply':
                self._parameters[name].values = self._parameters[name].values * values
            elif operation == 'add':
                self._parameters[name].values = self._parameters[name].values + values
            else:
                raise ValueError("The operation argument is not recognized as 'replace', 'multiply', or 'add'")

        if dim == ('one',):
            dim = par_dims

        for d in dim:
            if d not in par_dims:
                raise ValueError(f"The dimension {dim}, does not exist for the {name} parameter.")

        pdim_list = []
        vsize = []
        for i in par_dims:
            pds = slice(None)
            pdim_list.append(pds)
            vsize.append(1)

        if idx is not None:
            for k, v in idx.items():
                if k not in dim:
                    raise ValueError(f"Indices were provided for dimension {dim}, which does not match any dimension in"
                                     f" the dim argument.")
                dimid = par_dims.index(k)
                pdim_list[dimid] = v

        else:
            for d in dim:
                did = par_dims.index(d)
                vsize[did] = par_shp[did]
            vsize = tuple(vsize)
            if isinstance(values, (int, float)):
                if vsize != par_shp:
                    raise ValueError(f"Only one value was provided for the dimension(s) {dim}, when only one dimension"
                                 f" is specified, the values must match the shape of that dimension.")
                pass
            else:
                values = np.reshape(values, vsize)

        p_sel = tuple(pdim_list)

        if operation == 'replace':
            self._parameters[name].values[p_sel] = values
        elif operation == 'multiply':
            self._parameters[name].values[p_sel] = self._parameters[name].values[p_sel] * values
        elif operation == 'add':
            self._parameters[name].values[p_sel] = self._parameters[name].values[p_sel] + values
        else:
            raise ValueError("The operation argument is not recognized as 'replace', 'multiply', or 'add'")

    # This appears to work for a single file, but if multiple files were loaded using pygsflow methods, I'm not sure
    #   that this writes/orients correctly
    def write_paramfile(self, flname: Optional[Union[str, Path, list[Union[str, Path]]]] = None):
        if len(self._from_files) != 0:
            if flname is not None:
                if (isinstance(flname, (str, Path))) & (len(self._from_files) > 1):
                    msg = (
                        f"Only one filename was provided to rename multiple parameter files, "
                        f"defaulting to the original filenames.")
                    user_warning(
                        msg,
                        inspect.getframeinfo(
                            inspect.currentframe()
                        ),
                    )
                    flname = self._from_files
                elif (isinstance(flname, list)) & (len(self._from_files) != len(flname)):
                    msg = (
                        f"{len(flname)} filenames provided for {len(self._from_files)} parameter files to rename, "
                        f"defaulting to the original filenames.")
                    user_warning(
                        msg,
                        inspect.getframeinfo(
                            inspect.currentframe()
                        ),
                    )
                    flname = self._from_files
                else:
                    if isinstance(flname, (str, Path)):
                        flname = [flname]
            else:
                flname = self._from_files

            arch_param = self.parameters.copy()
            arch_dimrecs = self._pygsflow_dimrecs.copy()
            for i, f in enumerate(self._from_files):
                if isinstance(f, Path):
                    ff = f.as_posix()
                else:
                    ff = f
                alt_dimrecs = []
                if arch_dimrecs is not None:
                    for r in arch_dimrecs:
                        if isinstance(r.file_name, Path):
                            rf = r.file_name.as_posix()
                        else:
                            rf = r.file_name
                        if rf == ff:
                            alt_dimrecs.append(r)
                        else:
                            continue

                varlist = []
                for v in list(arch_param.data_vars):
                    var = arch_param[v]
                    if isinstance(var.attrs['file_name'], Path):
                        vf = var.attrs['file_name'].as_posix()
                    else:
                        vf = var.attrs['file_name']
                    if vf == ff:
                        varlist.append(v)
                    else:
                        continue

                if len(alt_dimrecs) == 0:
                    self._pygsflow_dimrecs = None
                else:
                    self._pygsflow_dimrecs = alt_dimrecs
                self.parameters = arch_param[varlist]
                self.pygsflow_param_obj.write(flname[i])

            self._pygsflow_dimrecs = arch_dimrecs
            self.parameters = arch_param
        else:
            if flname is None:
                raise ValueError("A file name must be given when parameters are not loaded from file.")
            if isinstance(flname, list):
                flname = flname[0]
            self.pygsflow_param_obj.write(flname)

    def _check_param_dtype(self, pname: str, pvals: np.ndarray) -> tuple:
        attrs_m = meta.find_variables(pname)[pname]

        if attrs_m['type'] == 'I':
            ptyp = int
            if np.issubdtype(pvals.dtype, np.integer):
                out = True
            else:
                out = False
        elif attrs_m['type'] == 'F':
            ptyp = float
            if np.issubdtype(pvals.dtype, np.floating):
                out = True
            else:
                out = False
        else:
            # just default to float? Not sure if there are type errors in the pywatershed metadata
            ptyp = float
            out = True

        return out, ptyp

    @staticmethod
    def load_pygsflow_obj(param_obj: pygsflowparams):
        newparams = PRMSParameters()
        for rec in param_obj.parameters_list:
            newparams.add_pygsflow_record(rec)

        return newparams

    @staticmethod
    def load_paramfile(paramfiles: Union[str, Path, list]):
        if isinstance(paramfiles, (str, Path)):
            paramfiles = [paramfiles]
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
        gridcell_types = np.zeros(len(self.grid), dtype=int)
        lake_hrus = np.zeros(len(self.grid), dtype=int)
        for i, r in grid_features.iterrows():
            if (r['region'] == 'basin') | (r['region'] == 'stream'):
                msk = self.grid.intersects(r.geometry, align=True).values
                gridcell_types[msk] = type_dict[r['region']]
            elif r['region'] == 'lake':
                msk = self.grid.intersects(r.geometry, align=True).values
                gridcell_types[msk] = type_dict[r['region']]
                lake_hrus[msk] = r['regID'] + 1

                self.add_parameter(
                    "lake_hru_id",
                    lake_hrus,
                    len(self.grid),
                    'nhru'
                )
            elif (r['region'] == 'swale') | (r['region'] == 'glacier'):
                raise NotImplementedError("Mapping swale and glacier hru_type and related parameters not supported yet.")
            else:
                raise ValueError(f"The region type {r['region']} specified in row {i} of the GeoDataFrame is not valid, "
                                 f"should be inactive, basin, stream, lake, swale, or glacier.")

            if np.isin(2, np.unique(gridcell_types)):
                nlakehrus = np.where(gridcell_types == 2)[0]
                self.parameters.coords['nlake_hrus'] = (('nlake_hrus',), nlakehrus)

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
            # overlapping stream segments on the model grid creates an index problem here as small geometries are
            # overwritten
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

        # reindex the stream segments so they are compatible with PRMS, a range between 1 and nsegment
        allstrmvals = np.unique(strm_mask)
        newstrmvals = np.arange(allstrmvals.size)
        newstrmarray = replace_array_values(strm_mask, (allstrmvals, newstrmvals))
        self.stream_grid = newstrmarray
        # keep original geometry indexes for reference
        segids = np.unique(self._stream_grid[self._strm_ids])
        seggeoms = streams.loc[allstrmvals[allstrmvals != 0] - 1, ['geometry']]
        seggeoms['geom_idx'] = allstrmvals[allstrmvals != 0] - 1
        seggeoms['seg_ids'] = segids
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
            np.abs(self.fdir.slope),
            len(self.grid),
            'nhru'
        )

    def cascades_from_hydrography(self,
                                  streams: Optional[gpd.GeoDataFrame] = None,
                                  lakes: Optional[gpd.GeoDataFrame] = None,
                                  swales: Optional[gpd.GeoDataFrame] = None,
                                  cascade_streams: bool = True):
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

        if not cascade_streams:
            newtypes = self._hru_types.copy()
            newtypes[self._strm_ids] = 3
            self.hru_types = np.where((self._hru_types != 2), newtypes, self._hru_types)
            #stmovlp = np.where(np.isin(hru_up - 1, self._strm_ids))[0]
            #hru_up = np.delete(hru_up, stmovlp)
            #hru_dwn = np.delete(hru_dwn, stmovlp)
            #hru_pct = np.delete(hru_pct, stmovlp)

        if np.isin(np.array([2, 3]), np.unique(self._hru_types)).any():
            snk_ids = np.where(np.isin(self._hru_types, np.array([2, 3])))[0]
            snkovlp = np.where(np.isin(hru_up - 1, snk_ids))[0]
            hru_up = np.delete(hru_up, snkovlp)
            hru_dwn = np.delete(hru_dwn, snkovlp)
            hru_pct = np.delete(hru_pct, snkovlp)
        # set stream segment parameters - lake_segment_id and segment_type with lakes
        if np.isin(2, np.unique(self._hru_types)):
            lake_segid = segtype.copy()
            if "lake_hru_id" not in list(self.parameters.data_vars):
                raise AttributeError("Lakes have been mapped in the 'hru_type' parameter but the 'lake_hru_id' parameter "
                                     "is not specified, this could be caused by providing a custom 'hru_type' array "
                                     "when initializing the PRMSCascades class instead of using the map_hru_type() func. "
                                     "Specify the 'lake_hru_id' param to continue.")
            lk_ids = self.parameters["lake_hru_id"].values
            for i in np.unique(lk_ids):
                lk = np.where(lk_ids == i)[0]
                strmlakeids = np.unique(self._stream_grid[lk])
                lake_segid = np.where(np.isin(segids, strmlakeids), i, lake_segid)

            segtype[np.where(lake_segid != 0)[0]] = 2

            self.add_parameter(
                "lake_segment_id",
                lake_segid.astype(int),
                len(segids),
                'nsegment'
            )

            self.add_parameter(
                "segment_type",
                segtype.astype(int),
                len(segids),
                'nsegment'
            )

        else:
            # set stream segment parameters segment_type with no lakes
            self.add_parameter(
                "segment_type",
                segtype.astype(int),
                len(segids),
                'nsegment'
            )

        hru_seg_id = self._stream_grid.copy()[hru_dwn - 1]

        # set stream segment parameters - tosegment; *FUTURE* segment_outflow_id
        # stream network elevations assigned by segment and treated like a DEM, so pits are filled prior to
        # assessing stream connection - so far successfull at removing sinks
        nabs = self._flopygrid.neighbors(method='queen')
        mine = []
        for seg in segids:
            idx = np.where(self._stream_grid == seg)[0]
            me = self._elevation[idx].min()
            mine.append(me)
        seg_elev = np.array(mine)

        sneighb = []
        edges = []
        for i, seg in enumerate(segids):
            idx = np.where(self._stream_grid == seg)[0]
            allnabs = []
            for n in idx:
                nl = nabs[n]
                allnabs.extend(nl)
            alnb_array = np.array(allnabs)
            stmnbs = self._stream_grid[alnb_array]
            snb = stmnbs[np.isin(stmnbs, np.array([0, seg]), invert=True)]
            snb = np.unique(snb)
            if snb.size == 1:
                edges.append(i)
            sneighb.append((snb - 1).tolist())

        strm_con = dict(zip((segids - 1).tolist(), sneighb))

        wf = np.ones(seg_elev.size) * 1e+10
        wf[edges] = seg_elev[edges]

        modified = True
        niter = 0
        while modified:
            niter += 1
            modified, wf = dany.dem_conditioning._inner_flood_and_drain_fill(seg_elev, wf, 2e-06, strm_con)


        sorted = []
        dwn_seg = []
        for i, seg in enumerate(segids):
            snb = np.array(strm_con[i])
            if snb.size == 1:
                if wf[np.where(segids == seg)] < wf[np.where(segids == snb[0]+1)]:
                    sorted.append(1)
                    dwn_seg.append(0)
                elif wf[np.where(segids == seg)] > wf[np.where(segids == snb[0]+1)]:
                    sorted.append(1)
                    dwn_seg.append(snb[0]+1)
                else:
                    sorted.append(0)
                    dwn_seg.append(-1)
            else:
                snb_srt = np.argsort(wf[np.where(np.isin(segids, snb+1))[0]])
                snb = snb[snb_srt]
                snb_l = snb[0]+1
                if np.unique(snb).size == snb.size:
                    sorted.append(1)
                    dwn_seg.append(snb_l)
                else:
                    sorted.append(0)
                    dwn_seg.append(-1)

        dwn_sg_id = np.array(dwn_seg)
        sg_srt = np.array(sorted)

        circ = []
        dwn_sg_id = np.array(dwn_seg)
        for sd in range(len(dwn_sg_id)):
            if dwn_seg[dwn_seg[sd] - 1] == sd + 1:
                circ.append(sd)

        if len(circ) != 0:
            msg = f"Circular routing was detected in the stream network, the stream segments {circ} are undeclared pits."
            user_warning(
                msg,
                inspect.getframeinfo(
                    inspect.currentframe()
                ),
            )

        self._stream_graph = dict([(k+1, (np.array(v)+1).tolist()) for k,v in strm_con.items()])
        self._strm_sort = sg_srt
        self._segment_elev = wf

        self.add_parameter(
            "tosegment",
            dwn_sg_id.astype(int),
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
            hru_seg_id.astype(int),
            len(hru_seg_id),
            'ncascade'
        )

        self.add_parameter(
            'cascade_flg',
            np.array([0], dtype=int),
            1,
            'one'
        )

        self.add_parameter(
            'cascade_tol',
            np.array([0], dtype=float),
            1,
            'one'
        )

        self.add_parameter(
            'circle_switch',
            np.array([1], dtype=int),
            1,
            'one'
        )

    def groundwater_cascades(self, gw_dem: Optional[np.ndarray] = None):
        if gw_dem is None:
            cascade_params = np.array(['hru_up_id', 'hru_down_id', 'hru_pct_up', 'hru_strmseg_down_id'])
            if not np.isin(cascade_params, np.array(list(self.parameters.data_vars))).all():
                raise AttributeError(
                    "Cascade parameters were not found in the parameter Dataset and no GW map for routing was provided, compute cascades first for default groundwater cascades")

            self.add_parameter('gw_up_id', self.parameters['hru_up_id'].values, len(self.parameters['hru_up_id'].values), 'ncascdgw')
            self.add_parameter('gw_down_id', self.parameters['hru_down_id'].values, len(self.parameters['hru_down_id'].values), 'ncascdgw')
            self.add_parameter('gw_pct_up', self.parameters['hru_pct_up'].values, len(self.parameters['hru_pct_up'].values), 'ncascdgw')
            self.add_parameter('gw_strmseg_down_id', self.parameters['hru_strmseg_down_id'].values, len(self.parameters['hru_strmseg_down_id'].values), 'ncascdgw')
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

    def export_cascades_grid(self,
                             filename: Union[str, Path],
                             add_params: Optional[list] = None,
                             flow_directions: bool = False,
                             flow_accum: bool = False,
                             stream_grid: bool = False,
                             sinks: bool = False):
        out = self.grid.copy()
        if add_params is not None:
            for i in add_params:
                out[i] = self.parameters[i].values

        if flow_directions:
            out['flow_directions'] = self.fdir.flow_direction_array

        if flow_accum:
            out['flow_accum'] = self.fdir.flow_accumulation_array

        if stream_grid:
            out['stream_grid'] = self.stream_grid

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
        stm_out['segelev'] = self._segment_elev
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
                if np.issubdtype(dtype, item[platform]):
                    outtype = item['PRMS']
                    return outtype
        elif platform != 'numpy':
                if dtype == item[platform]:
                    outtype = item['PRMS']
                    return outtype
        else:
            raise ValueError("The input dtype could not be resolved to a prms dtype code.")

def ddsolrad_defaults(paramob: PRMSParameters,
                      spatial_dim: Optional[str] = 'nhru',
                      time_dim: bool = True,
                      nsub: Optional[int] = None):
    """
    Assigns the PRMS defaults for the ddsolrad module to a parameter object. If the parameter object has no grid attribute
    assigned, the defaults are mapped to just the time dimension, in this case PRMS dimension 'nmonths.'
    Args:
        paramob: mihms.prep.prms.params.PRMSParameters
            The parameter object to which the default ddsolrad parameters will be appended, includes defaults for
            parameters:
            - dday_intcp
            - dday_slope
            - radadj_intcp
            - radadj_slope
            - ppt_rad_adj
            - radj_sppt
            - radj_wppt
            - radmax
            - tmax_index
        spatial_dim:
        time_dim:
        nsub:

    Returns: None
        ddsolrad parameters are added to the input parameter object

    """
    if spatial_dim == 'nhru':
        try:
            sdim_val = len(paramob.grid)
        except TypeError:
            msg = ("Dimension is set to 'nhru' but no grid is assigned to the parameter object, defaults will "
                   "be written to 'one' dimension only.")
            user_warning(
                msg,
                inspect.getframeinfo(
                    inspect.currentframe()
                ),
            )
            sdim_val = 1
            spatial_dim = 'one'
    elif spatial_dim == 'nsub':
        if nsub is None:
            msg = ("Dimension is set to 'nsub' but no nsub argument was specified, defaults will "
                   "be written to 'one' dimension only.")
            user_warning(
                msg,
                inspect.getframeinfo(
                    inspect.currentframe()
                ),
            )
            sdim_val = 1
            spatial_dim = 'one'
        else:
            sdim_val = nsub
    elif (spatial_dim == 'one') or (spatial_dim is None):
        sdim_val = 1
        spatial_dim = 'one'
    else:
        raise ValueError("The spatial dimension is not recognized as 'nhru', 'nsub', or 'one'")

    if time_dim & (spatial_dim != 'one'):
        paramob.add_parameter('dday_intcp', np.ones((12, sdim_val)) * -40.0, [12, sdim_val],
                              ['nmonths', spatial_dim])
        paramob.add_parameter('dday_slope', np.ones((12, sdim_val)) * 0.4, [12, sdim_val],
                              ['nmonths', spatial_dim])
        paramob.add_parameter('ppt_rad_adj', np.ones((12, sdim_val)) * 0.02, [12, sdim_val],
                              ['nmonths', spatial_dim])
        paramob.add_parameter('radadj_intcp', np.ones((12, sdim_val)) * 1.0, [12, sdim_val],
                              ['nmonths', spatial_dim])
        paramob.add_parameter('radadj_slope', np.ones((12, sdim_val)) * 0.0, [12, sdim_val],
                              ['nmonths', spatial_dim])
        paramob.add_parameter('radj_sppt', np.ones(sdim_val) * 0.44, sdim_val, spatial_dim)
        paramob.add_parameter('radj_wppt', np.ones(sdim_val) * 0.5, sdim_val, spatial_dim)
        paramob.add_parameter('radmax', np.ones((12, sdim_val)) * 0.8, [12, sdim_val],
                              ['nmonths', spatial_dim])
        paramob.add_parameter('tmax_index', np.ones((12, sdim_val)) * 50.0, [12, sdim_val],
                              ['nmonths', spatial_dim])
    elif time_dim & (spatial_dim == 'one'):
        paramob.add_parameter('dday_intcp',np.ones(12, dtype=float) * -40.0,12,'nmonths')
        paramob.add_parameter('dday_slope',np.ones(12, dtype=float) * 0.4,12,'nmonths')
        paramob.add_parameter('ppt_rad_adj',np.ones(12, dtype=float) * 0.02,12,'nmonths')
        paramob.add_parameter('radadj_intcp',np.ones(12, dtype=float) * 1.0,12,'nmonths')
        paramob.add_parameter('radadj_slope',np.ones(12, dtype=float) * 0.0,12,'nmonths')
        paramob.add_parameter('radj_sppt',np.array([0.44]),sdim_val,spatial_dim)
        paramob.add_parameter('radj_wppt',np.array([0.5]),sdim_val,spatial_dim)
        paramob.add_parameter('radmax',np.array([0.8]),sdim_val,spatial_dim)
        paramob.add_parameter('tmax_index',np.array([50.0]),sdim_val,spatial_dim)
    else:
        paramob.add_parameter('dday_intcp',np.ones(sdim_val) * -40.0, sdim_val, spatial_dim)
        paramob.add_parameter('dday_slope',np.ones(sdim_val) * 0.4,sdim_val, spatial_dim)
        paramob.add_parameter('ppt_rad_adj',np.ones(sdim_val) * 0.02,sdim_val, spatial_dim)
        paramob.add_parameter('radadj_intcp',np.ones(sdim_val) * 1.0,sdim_val, spatial_dim)
        paramob.add_parameter('radadj_slope',np.ones(sdim_val) * 0.0,sdim_val, spatial_dim)
        paramob.add_parameter('radj_sppt',np.ones(sdim_val) * 0.44,sdim_val, spatial_dim)
        paramob.add_parameter('radj_wppt',np.ones(sdim_val) * 0.5,sdim_val, spatial_dim)
        paramob.add_parameter('radmax',np.ones(sdim_val) * 0.8,sdim_val, spatial_dim)
        paramob.add_parameter('tmax_index',np.ones(sdim_val) * 50.0,sdim_val, spatial_dim)

def transp_tindex_and_intercept_from_rasters(paramob: PRMSParameters,
                                             veg_type: Union[str, Path, rio.DatasetReader, xr.DataArray, xr.Dataset],
                                             veg_cover: Union[str, Path, rio.DatasetReader, xr.DataArray, xr.Dataset],
                                             sand: Union[str, Path, rio.DatasetReader, xr.DataArray, xr.Dataset],
                                             clay: Union[str, Path, rio.DatasetReader, xr.DataArray, xr.Dataset],
                                             spatial_dim: Optional[str] = 'nhru',
                                             nsub: Optional[int] = None):
    """
    Assigns the transp_tindex and interception parameters from vegetation type, cover, and soils raster datasets. These
    datasets should come from LANDFIRE and SSURGO (or mimic their classification scheme) as the raster remaps used are
    specific to those sources. For all transp_tindex paramters that are not dependent on the raster inputs, the PRMS
    default values are returned for their maximum dimensions.

    Args:
        paramob: mihms.prep.prms.params.PRMSParameters
            A PRMSParamters object, must have grid assigned to use this function.
        veg_type: str | Path | rio.DatasetReader | xr.DataArray | xr.Dataset
            A vegetation-type raster. Must be from LANDFIRE dataset or be reclassified to mimic LANDFIRE integer
            classes because default remaps used in MIHMS are for LANDFIRE.
        veg_cover: str | Path | rio.DatasetReader | xr.DataArray | xr.Dataset
            A vegetation-cover raster. Must be from LANDFIRE dataset or be reclassified to mimic LANDFIRE integer
            classes because default remaps used in MIHMS are for LANDFIRE.
        sand: str | Path | rio.DatasetReader | xr.DataArray | xr.Dataset
            A percent sand content soil raster. This must be from SSURGO or represent proportion of sand as value
            between 0.0 and 1.0.
        clay: str | Path | rio.DatasetReader | xr.DataArray | xr.Dataset
            A percent clay content soil raster. This must be from SSURGO or represent proportion of sand as value
            between 0.0 and 1.0.
        spatial_dim:
        nsub:


    Returns: None
        Adds the relevant parameters to the parameter object given to the function. Parameters not delineated by raster
        data are assigned PRMS defaults at their maximum dimension.
    """
    if paramob.grid is None:
        raise ValueError("The paramater object does not have a grid assigned, grid must be assigned to perform raster"
                         "mapping of paramters.")

    mod_vtype = calc_zonal_stats(paramob.grid, veg_type,
                                    stats=[lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else np.nan],
                                    all_touched=True)
    mod_vtype = mod_vtype.values.ravel()
    mod_vcov = calc_zonal_stats(paramob.grid, veg_cover,
                                   stats=[lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else np.nan],
                                   all_touched=True)
    mod_vcov = mod_vcov.values.ravel()
    mod_clay = calc_zonal_stats(paramob.grid, clay, stats='median', all_touched=True)
    mod_clay = mod_clay.values.ravel()
    mod_sand = calc_zonal_stats(paramob.grid, sand, stats='median', all_touched=True)
    mod_sand = mod_sand.values.ravel()

    remaps = {}
    for r in (prep.root / prep.raster_remaps).glob('*.rmp'):
        remaps[r.stem] = bu.build_lut(r)

    covtype_lut = remaps[[k for k in remaps.keys() if 'covtype' in k][0]]
    covdensum_lut = remaps[[k for k in remaps.keys() if 'covdensum' in k][0]]
    covdenwin_lut = remaps[[k for k in remaps.keys() if 'covdenwin' in k][0]]
    snowintcp_lut = remaps[[k for k in remaps.keys() if 'snow_intcp' in k][0]]
    srainintcp_lut = remaps[[k for k in remaps.keys() if 'srain_intcp' in k][0]]

    # define raster based transp_tindex and interception params
    covtype = bu.covtype(mod_vtype, covtype_lut)
    covden_sum = bu.covden_sum(mod_vcov, covdensum_lut)
    covden_win = bu.covden_win(covtype.values, covdenwin_lut)
    rad_trncf = bu.rad_trncf(covden_win.values)
    snow_intcp = bu.snow_intcp(mod_vtype, snowintcp_lut)
    srain_intcp = bu.srain_intcp(mod_vtype, srainintcp_lut)
    wrain_intcp = bu.wrain_intcp(mod_vtype, snowintcp_lut)
    soil_type = bu.soil_type(mod_clay, mod_sand)

    paramob.add_pygsflow_record(covtype)
    paramob.add_pygsflow_record(covden_sum)
    paramob.add_pygsflow_record(covden_win)
    paramob.add_pygsflow_record(rad_trncf)
    paramob.add_pygsflow_record(snow_intcp)
    paramob.add_pygsflow_record(srain_intcp)
    paramob.add_pygsflow_record(wrain_intcp)
    paramob.add_pygsflow_record(soil_type)

    if spatial_dim == 'nhru':
            sdim_val = len(paramob.grid)
    elif spatial_dim == 'nsub':
        if nsub is None:
            msg = ("Dimension is set to 'nsub' but no nsub argument was specified, defaults will "
                   "be written to 'one' dimension only.")
            user_warning(
                msg,
                inspect.getframeinfo(
                    inspect.currentframe()
                ),
            )
            sdim_val = 1
            spatial_dim = 'one'
        else:
            sdim_val = nsub
    elif (spatial_dim == 'one') or (spatial_dim is None):
        sdim_val = 1
        spatial_dim = 'one'
    else:
        raise ValueError("The spatial dimension is not recognized as 'nhru', 'nsub', or 'one'")
    # remaining default params
    paramob.add_parameter('potet_sublim', np.ones(sdim_val, dtype=float) * 0.5, sdim_val, spatial_dim)
    paramob.add_parameter('transp_beg', np.ones(sdim_val, dtype=int), sdim_val, spatial_dim)
    paramob.add_parameter('transp_end', np.ones(sdim_val, dtype=int) * 13, sdim_val, spatial_dim)
    paramob.add_parameter('transp_tmax', np.ones(sdim_val, dtype=float), sdim_val, spatial_dim)

def soilzone_and_srunoff_smidx_from_rasters(paramob: PRMSParameters,
                                            veg_type: Union[str, Path, rio.DatasetReader, xr.DataArray, xr.Dataset],
                                            awc: Union[str, Path, rio.DatasetReader, xr.DataArray, xr.Dataset],
                                            impervious: Union[str, Path, rio.DatasetReader, xr.DataArray, xr.Dataset],
                                            spatial_dim: Optional[str] = 'nhru',
                                            nsub: Optional[int] = None):
    """

    Args:
        paramob:
        veg_type:
        awc:
        impervious:
        spatial_dim:
        nsub:

    Returns:

    """
    if paramob.grid is None:
        raise ValueError("The paramater object does not have a grid assigned, grid must be assigned to perform raster"
                         "mapping of paramters.")

    mod_vtype = calc_zonal_stats(paramob.grid, veg_type,
                                    stats=[lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else np.nan],
                                    all_touched=True)
    mod_vtype = mod_vtype.values.ravel()
    mod_awc = calc_zonal_stats(paramob.grid, awc, stats='median', all_touched=True)
    mod_awc = mod_awc.values.ravel()
    mod_imperv = calc_zonal_stats(paramob.grid, impervious, stats='mean', all_touched=True)
    mod_imperv = mod_imperv.values.ravel()

    remaps = {}
    for r in (prep.root / prep.raster_remaps).glob('*.rmp'):
        remaps[r.stem] = bu.build_lut(r)

    # soil_moist_max is the only thing set by rasters, vegtype -> root depth, root depth & awc used to estimate
    # soil_moist_max
    rtdepth_lut = remaps[[k for k in remaps.keys() if 'rtdepth' in k][0]]
    rtdepth = bu.root_depth(mod_vtype, rtdepth_lut)
    soil_moistmx = bu.soil_moist_max(mod_awc, rtdepth)
    paramob.add_pygsflow_record(soil_moistmx)
    # impervious area and carea_max from NLCD impervious raster for srunoff_smidx
    mod_imperv[np.isnan(mod_imperv)] = np.nanmean(mod_imperv)
    paramob.add_parameter("hru_percent_imperv",mod_imperv,len(mod_imperv),'nhru')
    paramob.add_parameter("carea_max",1 - mod_imperv,len(mod_imperv),'nhru')

    if spatial_dim == 'nhru':
            sdim_val = len(paramob.grid)
            ssr_dim = 'nssr'
    elif spatial_dim == 'nsub':
        if nsub is None:
            msg = ("Dimension is set to 'nsub' but no nsub argument was specified, defaults will "
                   "be written to 'one' dimension only.")
            user_warning(
                msg,
                inspect.getframeinfo(
                    inspect.currentframe()
                ),
            )
            sdim_val = 1
            spatial_dim = 'one'
            ssr_dim = 'one'
        else:
            sdim_val = nsub
            ssr_dim = 'nsub'
    elif (spatial_dim == 'one') or (spatial_dim is None):
        sdim_val = 1
        spatial_dim = 'one'
        ssr_dim = 'one'
    else:
        raise ValueError("The spatial dimension is not recognized as 'nhru', 'nsub', or 'one'")
    # defaults for other srunoff_smiddx params
    paramob.add_parameter('imperv_stor_max',np.ones(sdim_val, dtype=float) * 0.05, sdim_val, spatial_dim)
    paramob.add_parameter('smidx_coef',np.ones(sdim_val, dtype=float) * 0.005, sdim_val, spatial_dim)
    paramob.add_parameter('smidx_exp',np.ones(sdim_val, dtype=float) * 0.3, sdim_val, spatial_dim)
    paramob.add_parameter('snowinfil_max',np.ones(sdim_val, dtype=float) * 2.0, sdim_val, spatial_dim)
    # the rest of the soil zone and runoff params are defaults
    paramob.add_parameter('fastcoef_lin',np.ones(sdim_val) * 0.1, sdim_val, spatial_dim)
    paramob.add_parameter('fastcoef_sq',np.ones(sdim_val) * 0.8, sdim_val, spatial_dim)
    paramob.add_parameter('slowcoef_lin',np.ones(sdim_val) * 0.015, sdim_val, spatial_dim)
    paramob.add_parameter('slowcoef_sq',np.ones(sdim_val) * 0.1, sdim_val, spatial_dim)
    paramob.add_parameter('soil2gw_max',np.ones(sdim_val) * 0.0, sdim_val, spatial_dim)
    paramob.add_parameter('ssr2gw_exp',np.ones(sdim_val) * 1.0, sdim_val, ssr_dim)
    paramob.add_parameter('ssr2gw_rate',np.ones(sdim_val) * 0.1, sdim_val, ssr_dim)
    paramob.add_parameter('soil_rechr_max_frac',np.ones(sdim_val) * 1.0, sdim_val, spatial_dim)
    paramob.add_parameter('pref_flow_den',np.ones(sdim_val) * 0.0, sdim_val, spatial_dim)
    paramob.add_parameter('pref_flow_infil_frac',np.ones(sdim_val) * -1.0, sdim_val, spatial_dim)
    paramob.add_parameter('sat_threshold',np.ones(sdim_val) * 999.0, sdim_val, spatial_dim)
    paramob.add_parameter("soil_moist_init_frac",np.ones(sdim_val) * 0.1, sdim_val, spatial_dim)
    paramob.add_parameter("ssstor_init_frac",np.ones(sdim_val) * 0.1, sdim_val, ssr_dim)
    paramob.add_parameter("soil_rechr_init_frac",np.ones(sdim_val) * 0.1, sdim_val, spatial_dim)

def groundwater_flow_defaults(paramob: PRMSParameters,
                              spatial_dim: Optional[str] = 'ngw',
                              nsub: Optional[int] = None):
    """Adds the PRMS default groundwater paramters to a parameter object.

    This function adds PRMS groundwater flow parameter default values to an input PRMSParameters class.
    There are several options for mapping the values, with 3 different dimension options. The maximum dimension
    specifies the default value for every hydrologic response unit (hru, or in this case ngw). This option requires
    that a grid is specified with the PRMSParameters class. The nsub dimension maps to the number of sub-basins in
    the model domain, this options requires the use of the nsub argument to specify the number of subbasins in the
    model domain.

    Args:
        paramob:
            A PRMSParameters object to add the default parameter values for PRMS groundwater flow parameters to.
        spatial_dim:
            A string that determines which of the available dimension specifications is used. The maximum
            dimension space for groundwater flow parameters is ngw which is the size of the model domain. This is the
            default setting and requires a .grid property to be assigned to the input paramob object.
            Valid arguments are:
                - 'ngw'
                - 'nsub'
                - 'one'
        nsub:
            An integer that is the number of sub-basins in the model domain. Only used if dimensions = 'nsub'

    Returns: None
        The function adds the default parameters to the input PRMSParameters object.

    """
    if spatial_dim == 'ngw':
        try:
            sdim_val = len(paramob.grid)
        except TypeError:
            msg = ("Dimension is set to 'ngw' but no grid is assigned to the parameter object, defaults will "
                   "be written to 'one' dimension only.")
            user_warning(
                msg,
                inspect.getframeinfo(
                    inspect.currentframe()
                ),
            )
            sdim_val = 1
            spatial_dim = 'one'
    elif spatial_dim == 'nsub':
        if nsub is None:
            msg = ("Dimension is set to 'nsub' but no nsub argument was specified, defaults will "
                   "be written to 'one' dimension only.")
            user_warning(
                msg,
                inspect.getframeinfo(
                    inspect.currentframe()
                ),
            )
            sdim_val = 1
            spatial_dim = 'one'
        else:
            sdim_val = nsub
    elif (spatial_dim == 'one') or (spatial_dim is None):
        sdim_val = 1
        spatial_dim = 'one'
    else:
        raise ValueError("The spatial dimension is not recognized as 'ngw', 'nsub', or 'one'")

    paramob.add_parameter('gwflow_coef', np.ones(sdim_val) * 0.015, sdim_val, spatial_dim)
    paramob.add_parameter('gwsink_coef', np.ones(sdim_val) * 0.0, sdim_val, spatial_dim)
    paramob.add_parameter('gwstor_min', np.ones(sdim_val) * 0.0, sdim_val, spatial_dim)
    paramob.add_parameter("gwstor_init", np.ones(sdim_val) * 2.0, sdim_val, spatial_dim)

def routing_defaults(paramob: PRMSParameters,
                     nsegment: int,
                     segtype: Optional[np.ndarray] = None,
                     routing_module: str = 'muskingum'):
    """

    Args:
        paramob:
        nsegment:
        segtype:
        routing_module:

    Returns:

    """
    if segtype is None:
        try:
            segtype = paramob.parameters['segment_type'].values
        except KeyError:
            print("segtype argument was not specified and the 'segment_type' parameter does not exist in the parameter "
                  "object assigned.")

    paramob.add_parameter('obsin_segment', np.zeros(nsegment, dtype=int), nsegment, 'nsegment')
    paramob.add_parameter('obsout_segment', np.zeros(nsegment, dtype=int), nsegment, 'nsegment')
    paramob.add_parameter('segment_flow_init', np.zeros(nsegment, dtype=float), nsegment, 'nsegment')
    if routing_module == 'strmflow_in_out':
        pass
    elif routing_module == 'muskingum':
        k_coef = np.ones(nsegment, dtype=float)
        k_coef = np.where(segtype == 2, 24.0, k_coef)
        paramob.add_parameter('K_coef', k_coef, nsegment, 'nsegment')
        paramob.add_parameter('x_coef', np.ones(nsegment) * 0.2, nsegment, 'nsegment')
    else:
        raise NotImplementedError("The specified PRMS streamflow module is not supported yet.")

def lake_routing_defaults(paramob: PRMSParameters,
                          nlake: int,
                          lake_method: str = 'linear'):
    """

    Args:
        paramob:
        nlake:
        lake_method:

    Returns:

    """
    paramob.add_parameter('lake_evap_adj', np.ones(nlake, dtype=float), nlake, 'nlake')
    paramob.add_parameter('lake_qro', np.ones(nlake) * 0.1, nlake, 'nlake')
    paramob.add_parameter('obsout_lake', np.zeros(nlake, dtype=int), nlake, 'nlake')

    if lake_method == 'linear':
        paramob.add_parameter('lake_type', np.ones(nlake, dtype=int) * 2, nlake, 'nlake')
        paramob.add_parameter('lake_din1', np.ones(nlake) * 0.1, nlake, 'nlake')
        paramob.add_parameter('lake_init', np.zeros(nlake), nlake, 'nlake')
        paramob.add_parameter('lake_coef', np.ones(nlake) * 0.1, nlake, 'nlake')
    else:
        raise NotImplementedError("The specified PRMS lake routing method is not supported yet.")

def set_output_options(paramob: PRMSParameters,
                       print_freq: int = 3,
                       print_type: int = 1):
    """

    Args:
        paramob:
        print_freq:
        print_type:

    Returns:

    """
    paramob.add_parameter('print_freq', np.array([print_freq]), 1, 'one')
    paramob.add_parameter('print_type', np.array([print_type]), 1, 'one')

def snow_comp_defaults(paramob: PRMSParameters,
                       spatial_dim: Union[str, None] = 'nhru',
                       time_dim: bool = True,
                       nsub: Optional[int] = None):
    """

    Args:
        paramob:
        spatial_dim:
        time_dim:
        nsub:

    Returns:

    """
    # potet_sublim, den_init, den_max, melt_look, melt_force, hru_deplcrv, snarea_curve, albset_rnm,
    # albset_rna, albset_sna, albset_snm, emis_noppt, cecn_coef, tstorm_mo, snowpack_init, freeh20_cap,
    paramob.add_parameter('albset_rna', np.array([0.8]), 1, 'one')
    paramob.add_parameter('albset_rnm', np.array([0.6]), 1, 'one')
    paramob.add_parameter('albset_sna', np.array([0.05]), 1, 'one')
    paramob.add_parameter('albset_snm', np.array([0.2]), 1, 'one')
    # add 1 default snarea_curve for western US mountainous regions (assumed?)
    # This curve comes from the PRMS sagehen model distributed with the software
    paramob.parameters.coords['ndepl'] = (('ndepl',), np.array([1], dtype=int))
    paramob.add_parameter('snarea_curve', np.array(prep.snarea_curves['western'], dtype=float), 11, 'ndeplval')
    paramob.add_parameter('hru_deplcrv', np.array([1]), 1, 'one')

    paramob.add_parameter('snowpack_init', np.array([0.0]), 1, 'one')
    # another default pulled from the saghen model example, this is in time dimension only, 12 values for each month
    tstorm = np.array(prep.tstorm['western'])[:,None]

    if spatial_dim == 'nhru':
        try:
            sdim_val = len(paramob.grid)
        except TypeError:
            msg = ("Dimension is set to 'nhru' but no grid is assigned to the parameter object, defaults will "
                   "be written to 'one' dimension only.")
            user_warning(
                msg,
                inspect.getframeinfo(
                    inspect.currentframe()
                ),
            )
            sdim_val = 1
            spatial_dim = 'one'
    elif spatial_dim == 'nsub':
        if nsub is None:
            msg = ("Dimension is set to 'nsub' but no nsub argument was specified, defaults will "
                   "be written to 'one' dimension only.")
            user_warning(
                msg,
                inspect.getframeinfo(
                    inspect.currentframe()
                ),
            )
            sdim_val = 1
            spatial_dim = 'one'
        else:
            sdim_val = nsub
    elif (spatial_dim == 'one') or (spatial_dim is None):
        sdim_val = 1
        spatial_dim = 'one'
    else:
        raise ValueError("The spatial dimension is not recognized as 'nhru', 'nsub', or 'one'")

    paramob.add_parameter('den_init', np.ones(sdim_val) * 0.1, sdim_val, spatial_dim)
    paramob.add_parameter('den_max', np.ones(sdim_val) * 0.6, sdim_val, spatial_dim)
    paramob.add_parameter('emis_noppt', np.ones(sdim_val) * 0.757, sdim_val, spatial_dim)
    paramob.add_parameter('freeh2o_cap', np.ones(sdim_val) * 0.05, sdim_val, spatial_dim)
    paramob.add_parameter('melt_force', np.ones(sdim_val, dtype=int) * 140, sdim_val, spatial_dim)
    paramob.add_parameter('melt_look', np.ones(sdim_val, dtype=int) * 90, sdim_val, spatial_dim)
    paramob.add_parameter('settle_const', np.ones(sdim_val) * 0.1, sdim_val, spatial_dim)
    paramob.add_parameter('snarea_thresh', np.ones(sdim_val) * 50.0, sdim_val, spatial_dim)
    if time_dim & (spatial_dim != 'one'):
        paramob.add_parameter('cecn_coef', np.ones((12, sdim_val)) * 5.0, [12, sdim_val], ['nmonths', spatial_dim])
        paramob.add_parameter('tstorm_mo', tstorm.repeat(sdim_val, axis=1), [12, sdim_val],
                              ['nmonths', spatial_dim])
    elif time_dim & (spatial_dim == 'one'):
        paramob.add_parameter('cecn_coef', np.ones(12) * 5.0, 12, 'nmonths')
        paramob.add_parameter('tstorm_mo', tstorm.ravel().astype(int), 12, 'nmonths')
    else:
        paramob.add_parameter('cecn_coef', np.ones(sdim_val) * 5.0, sdim_val, spatial_dim)
        paramob.add_parameter('tstorm_mo', np.zeros(sdim_val), sdim_val,spatial_dim)
