import os
import json
from subprocess import Popen, PIPE, STDOUT
from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
from gsflow.prms import PrmsData
from gsflow.control import ControlFile
from gsflow.output import StatVar

from mihms.prep.prms.params import PRMSParameters
from mihms.prep.prms.control import ControlBase
from mihms.prep.utils import write_prms_datafile
from mihms.utils.io import remove_all_files_in_directory

VAR_UNITS = {'ID': None,
             'runoff': 'cfs',
             'basin_tmin': 'deg C',
             'basin_tmax': 'deg C',
             'basin_ppt': 'inches',
             'basin_rain': 'inches',
             'basin_snow': 'inches',
             'basin_net_ppt': 'inches',
             'basin_intcp_stor': 'inches',
             'basin_potet': 'inches',
             'basin_actet': 'inches',
             'basin_pweqv': 'inches',  # SWE
             'basin_snowmelt': 'inches',
             'basin_snowcov': 'decimal fraction',
             'basin_sroff': 'inches',
             'basin_hortonian': 'inches',
             'basin_infil': 'inches',
             'basin_soil_moist': 'inches',
             'basin_recharge': 'inches',
             'basin_gwstor': 'inches',
             'basin_gwflow': 'inches',
             'basin_gwsink': 'inches',
             'basin_cms': 'cms',
             'basin_cfs': 'cfs',
             'basin_ssflow': 'inches',
             'basin_imperv_stor': 'inches',
             'basin_lake_stor': 'inches',
             'basin_ssstor': 'inches',
             'Date': 'datetime'}

OUTPUT_COLS = ['obs_q',
               'pred_q',
               'ppt',
               'actet',
               'gwstor',
               'rain',
               'snow',
               'pweqv',
               'swe_obs',
               'basin_tmin',
               'basin_tmax',
               'soil_moist',
               'infil',
               'recharge',
               'net_ppt',
               'potet',
               'snowmelt',
               'sroff',
               'hortonian',
               'intcp_stor',
               'gwflow',
               'gwsink',
               'ssflow',
               'imperv_stor',
               'lake_stor',
               'ssstor']


class HydroModel:
    def __init__(self):
        pass


class PrmsModel(HydroModel):

    def __init__(self, control_file: Union[str, Path, ControlBase]):
        super().__init__()
        if isinstance(control_file, ControlBase):
            self.control_file = control_file.model_wd / control_file.control_filename
            self.control = control_file
        else:
            self.control_file = control_file
            self.control = ControlBase.load_from_prms_controlfile(self.control_file)

        self.parameter_file = self.control.control_obj.get_values('param_file')
        self.parameters = PRMSParameters.load_paramfile(self.parameter_file)
        # apparently PrmsData.load_from_file() only accepts one input and cannot process multiple...datafiles which
        # is not consistent with how PRMS reads in this parameter from the control file (it can take multiple)
        # This isn't a big deal but should be corrected so it can mimic PRMS control parameter behavior
        self.data_file = self.control.control_obj.get_values('data_file')[0]
        self.data = PrmsData.load_from_file(self.data_file)

    def run_model(self,
                  stdout: Optional[Union[str, Path]]=None,
                  write_before: bool = False,
                  silent: bool = False,
                  report: bool = True,
                  clear_rundir: bool = True):

        if write_before:
            self.parameters.pygsflow_param_obj.write()
            self.control.save_controlfile()
            write_prms_datafile(self.data, self.data_file)

        if clear_rundir:
            remove_all_files_in_directory(self.control._out_pth, recursive=True)

        buff = []
        normal_msg = 'normal termination'

        argv = [self.control.control_obj.get_values('executable_model')[0], self.control_file]
        model_ws = os.path.dirname(self.control_file)
        proc = Popen(argv, stdout=PIPE, stderr=STDOUT, cwd=model_ws)

        success = None
        while True:
            line = proc.stdout.readline()
            c = line.decode('utf-8')
            if c != '':
                for msg in normal_msg:
                    if msg in c.lower():
                        success = True
                        break
                c = c.rstrip('\r\n')
                if not silent:
                    print('{}'.format(c))
                if report:
                    buff.append(c)
            else:
                break
        if stdout:
            with open(stdout, 'w') as fp:
                if report:
                    for line in buff:
                        fp.write(line + '\n')

        return success, buff


class RivSysModel:
    def __init__(self):
        pass


class RiverWare(RivSysModel):
    def __init__(self, control_file):
        super().__init__()

    def load_model(self):
        pass

    def init_dmi(self):
        pass

class HydraulicModel:
    def __init__(self):
        pass


class HecRas(HydraulicModel):

    def __init__(self):
        super().__init__()