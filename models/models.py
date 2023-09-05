import os
from subprocess import Popen, PIPE, STDOUT

import numpy as np
import pandas as pd

from gsflow.prms import PrmsData, PrmsParameters
from gsflow.control import ControlFile
from gsflow.output import StatVar


class HydroModel():
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

    def get_statvar(self, snow=None):

        self.statvar = StatVar.load_from_control_object(self.control)
        df = self.statvar.stat_df
        df.drop(columns=['Hour', 'Minute', 'Second'], inplace=True)

        # cms to m3 per day
        df['obs_q_vol_m3'] = 60 * 60 * 24 * df['runoff_1'] / 1e6
        df['pred_q_vol_m3'] = 60 * 60 * 24 * df['basin_cms_1'] / 1e6

        # ppt in inches, hru_area in acres
        hru_area = self.parameters.get_values('hru_area')[0]
        hru_active = np.count_nonzero(self.parameters.get_values('hru_type'))
        basin_area = hru_active * hru_area * 43560.
        ppt_meters = df['basin_ppt_1'] / 39.3701
        df['ppt_vol_m3'] = basin_area * ppt_meters / 1e6

        s, e = self.control.get_values('start_time'), self.control.get_values('end_time')
        try:
            df.index = pd.date_range('{}-{}-{}'.format(s[0], s[1], s[2]),
                                     '{}-{}-{}'.format(e[0], e[1], e[2]), freq='D')
            df.drop(columns=['Year', 'Month', 'Day'], inplace=True)
        except ValueError:
            pass

        if snow:
            s = pd.read_csv(snow, parse_dates=True, index_col='Unnamed: 0', infer_datetime_format=True)
            s = s.loc[df.index]
            s /= 25.4
            s.rename(columns={'mean': 'swe'}, inplace=True)
            s = s['swe']
            df = pd.concat([df, s], axis=1, ignore_index=False)

        self.statvar.stat_df = df
        return self.statvar.stat_df


class prms(HydroModel):
    def __init__(self):
        super().__init__()


class rivsysmodel():
    def __init__(self):
        pass


class riverware(rivsysmodel):
    def __init__(self):
        super().__init__()
