from pathlib import Path
from typing import Union, Optional
import inspect
import datetime

import pandas as pd
import numpy as np
import xarray as xr

from mihms.prep.prms.control import ControlBase
from mihms.prep.utils import load_prms_statvar
from mihms.utils.io import user_warning

class PrmsOutput:

    def __init__(self,
                 controlfile: Union[str, Path, ControlBase],
                 obs_dset: Optional[Union[str, Path, xr.Dataset]] = None):

        if isinstance(controlfile, (str, Path)):
            self._control_file = ControlBase.load_from_prms_controlfile(controlfile)
        elif isinstance(controlfile, ControlBase):
            self._control_file = controlfile
        else:
            raise ValueError("The input control file is not a recognized type.")

        self._control_recs = [r.name for r in self._control_file.control_obj.records_list]

        if 'model_output_file' not in self._control_recs:
            raise ValueError("The input control file does not have a model_output_file specified.")
        else:
            self.out_file = self._control_file.model_wd / self._control_file.control_obj.get_values('model_output_file')[0]
            if self.out_file.exists():
                self._out = self._parse_outfile()

        if 'csv_output_file' not in self._control_recs:
            msg = ("An output, summary csv was not specified in the control volume. These results will not be "
                   "returned.")
            user_warning(
                msg,
                inspect.getframeinfo(
                    inspect.currentframe()
                ),
            )
            self.summary_file = None
            self._summarycsv = None
        else:
            self.summary_file = self._control_file.model_wd / self._control_file.control_obj.get_values('csv_output_file')[0]
            if self.summary_file.exists():
                self._summarycsv = self._load_csv_summary()
            else:
                self._summarycsv = None

        if 'statsON_OFF' in self._control_recs:
            if self._control_file.control_obj.get_values('statsON_OFF')[0] == 1:
                self.statvar_file = self._control_file.model_wd / self._control_file.control_obj.get_values('stat_var_file')[0]
                self._statvar = load_prms_statvar(self.statvar_file)
            else:
                self.statvar_file = None
                self._statvar = None
        else:
            self.statvar_file = None
            self._statvar = None

        self._control_recs = [r.name for r in self._control_file.control_obj.records_list]

        self._output_paths = self._get_out_paths()
        self.variables = self._get_out_variables()

        self._obs = obs_dset

    @property
    def run_info(self):
        return print(self._out)

    @property
    def summary_csv(self):
        return self._summarycsv

    @property
    def selected_variables(self):
        if self._statvar is None:
            print("No statvar file was enabled for the current model results.")
        else:
            return self._statvar

    @property
    def observation_data(self):
        return self._obs

    @observation_data.setter
    def observation_data(self, dset):
        c_list = ['lon', 'lat', 'hru', 'location', 'time', 'subbasin', 'segment']
        dims = ('time', 'location')
        var_list = ['observation_data']
        for c in list(dset.coords):
            if c in c_list:
                c_b = True
            else:
                c_b = False
        for v in list(dset.data_vars):
            if v in var_list:
                v_b = True
            else:
                v_b = False
        if v_b:
            if dims == dset.observation_data.dims:
                d_b = True
            else:
                d_b = False
        else:
            d_b = False

        dset_check = np.array([c_b, v_b, d_b])
        if dset_check.all():
            self._obs = dset
        else:
            raise ValueError("The input dataset does not fit the format of a observation dataset. Use the mihms "
                             "create_obs_dataset() function for a properly formatted dataset.")

    def get_basin_output(self, frequency: str = 'daily') -> pd.DataFrame:
        bvars = self.variables['basin']
        if bvars is None:
            raise ValueError("There are no model outputs for sub-basins specified in the control file.")

        b_dir = self._output_paths['basin']
        if b_dir is None:
            raise ValueError("There was not a valid sub-basin output file path specified in the control file.")

        if frequency == 'daily':
            b_flnm = b_dir / "basin_vars.csv"
        else:
            b_flnm = b_dir / f"basin_vars_{frequency}.csv"
        bsndf = pd.read_csv(b_flnm, index_col='Date', parse_dates=True)
        bsndf.columns = np.char.strip(bsndf.columns.values.astype(str))

        return bsndf

    def get_subbasin_output(self, variable: Union[str, list] = 'all', frequency: str = 'daily') -> xr.Dataset:

        subvars = self.variables['subbasin']
        if subvars is None:
            raise ValueError("There are no model outputs for sub-basins specified in the control file.")

        sub_dir = self._output_paths['subbasin']
        if sub_dir is None:
            raise ValueError("There was not a valid sub-basin output file path specified in the control file.")

        if isinstance(variable, str):
            if variable == 'all':
                vars = subvars
            else:
                vars = [variable]
        elif isinstance(variable, list):
            vars = variable
        else:
            raise ValueError("The variable argument is not recognized as a string or list of strings.")

        ds = xr.Dataset()
        for i, v in enumerate(vars):
            if frequency == 'daily':
                sub_flnm = sub_dir / f"subbasin_{v}.csv"
            else:
                sub_flnm = sub_dir / f"subbasin_{v}_{frequency}.csv"
            subdf = pd.read_csv(sub_flnm, index_col='Date', parse_dates=True)
            ds[v] = (('time', 'subbasin'), subdf.values)
            if i == 0:
                ds.coords['subbasin'] = np.char.strip(subdf.columns.values.astype(str)).astype(int)
                ds.coords['time'] = subdf.index.values

        return ds

    def get_segment_output(self, variable: Union[str, list] = 'all', frequency: str = 'daily') -> xr.Dataset:
        segvars = self.variables['segment']
        if segvars is None:
            raise ValueError("There are no model outputs for segments specified in the control file.")

        seg_dir = self._output_paths['segment']
        if seg_dir is None:
            raise ValueError("There was not a valid segments output file path specified in the control file.")

        if isinstance(variable, str):
            if variable == 'all':
                vars = segvars
            else:
                vars = [variable]
        elif isinstance(variable, list):
            vars = variable
        else:
            raise ValueError("The variable argument is not recognized as a string or list of strings.")

        ds = xr.Dataset()
        for i, v in enumerate(vars):
            if frequency == 'daily':
                seg_flnm = seg_dir / f"segment_{v}.csv"
            else:
                seg_flnm = seg_dir / f"segment_{v}_{frequency}.csv"
            segdf = pd.read_csv(seg_flnm, index_col='Date', parse_dates=True)
            ds[v] = (('time', 'segment'), segdf.values)
            if i == 0:
                ds.coords['segment'] = np.char.strip(segdf.columns.values.astype(str)).astype(int)
                ds.coords['time'] = segdf.index.values

        return ds

    def get_hru_output(self, variable: Union[str, list] = 'all', frequency: str = 'daily') -> xr.Dataset:
        hruvars = self.variables['hru']
        if hruvars is None:
            raise ValueError("There are no model outputs for hru's specified in the control file.")

        hru_dir = self._output_paths['hru']
        if hru_dir is None:
            raise ValueError("There was not a valid hru's output file path specified in the control file.")

        if isinstance(variable, str):
            if variable == 'all':
                vars = hruvars
            else:
                vars = [variable]
        elif isinstance(variable, list):
            vars = variable
        else:
            raise ValueError("The variable argument is not recognized as a string or list of strings.")

        ds = xr.Dataset()
        for i, v in enumerate(vars):
            if frequency == 'daily':
                hru_flnm = hru_dir / f"hru_{v}.csv"
            else:
                hru_flnm = hru_dir / f"hru_{v}_{frequency}.csv"
            hrudf = pd.read_csv(hru_flnm, index_col='Date', parse_dates=True)
            ds[v] = (('time', 'hru'), hrudf.values)
            if i == 0:
                ds.coords['hru'] = np.char.strip(hrudf.columns.values.astype(str)).astype(int)
                ds.coords['time'] = hrudf.index.values

        return ds

    def get_obs_compare(self) -> xr.Dataset:
        pass

    def save_to_zarr(self):
        pass

    def _get_out_variables(self) -> dict:
        out_vars = {}
        if 'basinOutVar_names' in self._control_recs:
            bsn = self._control_file.control_obj.get_values('basinOutVar_names').tolist()
        else:
            bsn = None

        if 'nsubOutVar_names' in self._control_recs:
            subbsn = self._control_file.control_obj.get_values('nsubOutVar_names').tolist()
        else:
            subbsn = None

        if 'nsegmentOutVar_names' in self._control_recs:
            seg = self._control_file.control_obj.get_values('nsegmentOutVar_names').tolist()
        else:
            seg = None

        if 'nhruOutVar_names' in self._control_recs:
            hru = self._control_file.control_obj.get_values('nhruOutVar_names').tolist()
        else:
            hru = None

        out_vars['basin'] = bsn
        out_vars['subbasin'] = subbsn
        out_vars['segment'] = seg
        out_vars['hru'] = hru

        return out_vars

    def _get_out_paths(self):
        out_pths = {}
        if 'basinOutBaseFileName' in self._control_recs:
            bsn = self._control_file.control_obj.get_values('basinOutBaseFileName')[0]
            out_pths['basin'] = self._control_file.model_wd / Path(bsn).parent
        else:
            bsn = None
            out_pths['basin'] = bsn

        if 'nsubOutBaseFileName' in self._control_recs:
            subbsn = self._control_file.control_obj.get_values('nsubOutBaseFileName')[0]
            out_pths['subbasin'] = self._control_file.model_wd / Path(subbsn).parent
        else:
            subbsn = None
            out_pths['subbasin'] = subbsn

        if 'nsegmentOutBaseFileName' in self._control_recs:
            seg = self._control_file.control_obj.get_values('nsegmentOutBaseFileName')[0]
            out_pths['segment'] = self._control_file.model_wd / Path(seg).parent
        else:
            seg = None
            out_pths['segment'] = seg

        if 'nhruOutBaseFileName' in self._control_recs:
            hru = self._control_file.control_obj.get_values('nhruOutBaseFileName')[0]
            out_pths['hru'] = self._control_file.model_wd / Path(hru).parent
        else:
            hru = None
            out_pths['hru'] = hru

        return out_pths

    def _parse_outfile(self) -> str:
        with open(self.out_file, "r") as fl:
            content = fl.read()

        return content

    def _load_csv_summary(self) -> pd.DataFrame:
        d = pd.read_csv(self.summary_file)
        multi_col = d.loc[0].to_dict()
        dsel = d.loc[1:]
        dtindx = pd.DatetimeIndex(dsel['Date'])
        dsel = dsel.drop(columns='Date')
        dsel.index = dtindx
        dsel.columns = pd.MultiIndex.from_tuples([(col, unit) for col, unit in multi_col.items() if col != "Date"])
        dsel = dsel.apply(pd.to_numeric)

        return dsel
