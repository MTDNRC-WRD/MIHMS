import os
import json
from subprocess import Popen, PIPE, STDOUT

import numpy as np
import pandas as pd

from gsflow.prms import PrmsData, PrmsParameters
from gsflow.control import ControlFile
from gsflow.output import StatVar

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
    def __init__(selfs):
        pass


class MontanaPrmsModel(HydroModel):

    def __init__(self, control_file, parameter_file, data_file):
        super().__init__()
        self.control_file = control_file
        self.parameter_file = parameter_file
        self.data_file = data_file

        self.control = ControlFile.load_from_file(control_file)

        if str(self.control.get_record('param_file').values[0]) != self.parameter_file:
            self.control.param_file = [self.parameter_file]
            self.control.write()

        self.parameters = PrmsParameters.load_from_file(parameter_file)

        self.data = PrmsData.load_from_file(data_file)
        self.statvar = None

    def run_model(self, stdout=None):

        for obj_, var_ in [(self.control, 'control'),
                           (self.parameters, 'parameters'),
                           (self.data, 'data')]:
            if not obj_:
                raise TypeError('{} is not set, run "write_{}_file()"'.format(var_, var_))

        buff = []
        normal_msg = 'normal termination'
        report, silent = True, False

        argv = [self.control.get_values('executable_model')[0], self.control_file]
        model_ws = os.path.dirname(self.control_file)
        proc = Popen(argv, stdout=PIPE, stderr=STDOUT, cwd=model_ws)

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

    def get_statvar(self, snow_obs):

        self.statvar = StatVar.load_from_control_object(self.control)
        df = self.statvar.stat_df
        cols = [c.replace('_1', '') for c in df.columns]
        df.columns = cols
        df.drop(columns=['Hour', 'Minute', 'Second'], inplace=True)

        # cfs to cms per day
        if self.control.get_record('runoff_units').values[0] == 0:
            df['runoff'] = df['runoff'] / 0.028317

        df['runoff'][df['runoff'] < 0.0] = np.nan

        # try to get all the water balance components into MCMS per day
        df['obs_q'] = 60 * 60 * 24 * df['runoff'] / 1e6
        df['pred_q'] = 60 * 60 * 24 * df['basin_cms'] / 1e6

        # ppt in inches, hru_area in acres
        hru_area = self.parameters.get_values('hru_area')[0]
        hru_active = np.count_nonzero(self.parameters.get_values('hru_type'))
        # acres to m2
        basin_area = hru_active * hru_area.item() * 4046.856
        vols = []

        def inches_to_million_cubic_meters(col_str_):
            # inches to meters
            a = df[col_str_] / 39.3701
            _name = '_'.join(col_str_.split('_')[1:])
            # to million cubic meters
            df[_name] = basin_area * a / 1e6
            vols.append(_name)
            return None

        basin_vars = self.statvar.statvar_names

        [inches_to_million_cubic_meters(k) for k, v in VAR_UNITS.items() if k in basin_vars and v == 'inches']

        s, e = self.control.get_values('start_time'), self.control.get_values('end_time')
        try:
            df.index = pd.date_range('{}-{}-{}'.format(s[0], s[1], s[2]),
                                     '{}-{}-{}'.format(e[0], e[1], e[2]), freq='D')
            df.drop(columns=['Year', 'Month', 'Day'], inplace=True)
        except ValueError:
            pass

        with open(snow_obs, 'r') as fp:
            s = json.load(fp)

        s = [(k, v['0']) for k, v in s.items()]
        s = sorted(s, key=lambda x: x[0])
        dt = pd.DatetimeIndex([pd.to_datetime(d[0]) for d in s])
        s = [a[1] for a in s]
        s = np.array(s) * basin_area / 1e6
        s = pd.Series(index=dt, data=s, name='swe_obs')
        df = pd.concat([df, s], axis=1, ignore_index=False)

        # Agrimet data
        # 0.04184 mj m2-1 per langley

        df = df.loc['2017-01-01': '2017-12-31', OUTPUT_COLS]
        # df = df[OUTPUT_COLS]
        self.statvar.stat_df = df
        return self.statvar.stat_df


class prms(HydroModel):
    def __init__(self):
        super().__init__()


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