import json
import os
import warnings
from datetime import datetime

import numpy as np
from gsflow.builder import builder_utils as bu
from gsflow.control import ControlRecord
from gsflow.prms import PrmsData
from gsflow.prms.prms_parameter import ParameterRecord
from pandas import DataFrame, date_range

from prep.datafile import write_basin_datafile
from prep.prms.standard_build import StandardPrmsBuild
from utils.bounds import GeoBounds
from utils.thredds import GridMet

warnings.simplefilter(action='ignore', category=DeprecationWarning)


class CbhruPrmsBuild(StandardPrmsBuild):

    def __init__(self, config):
        StandardPrmsBuild.__init__(self, config)
        self.data_file = os.path.join(self.cfg.data_folder,
                                      '{}_runoff.data'.format(self.proj_name_res))

    def write_datafile(self):

        gages = self.cfg.usgs_gages

        self.data_params = [ParameterRecord(name='nobs', values=[1, ], datatype=1)]
        self.data_params.append(ParameterRecord('nrain', values=[0], datatype=1))
        self.data_params.append(ParameterRecord('ntemp', values=[0], datatype=1))

        self.data_params.append(ParameterRecord('snowpack_init',
                                                np.zeros_like(self.ksat).ravel(),
                                                dimensions=[['nhru', self.nhru]],
                                                datatype=2))

        if not os.path.isfile(self.data_file):
            units = 'metric' if self.cfg.runoff_units == 1 else 'standard'
            write_basin_datafile(gages, self.data_file, station_json=self.cfg.prms_data_stations,
                                 units=units, ghcn_data=self.cfg.prms_data_ghcn,
                                 start=self.cfg.start_time, end=self.cfg.end_time)

        self.data = PrmsData.load_from_file(self.data_file)

    def write_day_files(self):
        s = datetime.strptime(self.cfg.start_time, '%Y-%m-%d')
        e = datetime.strptime(self.cfg.end_time, '%Y-%m-%d')

        dt_range = date_range(start=self.cfg.start_time,
                              end=self.cfg.end_time,
                              freq='D')

        bounds = GeoBounds(north=self.lat.max(),
                           west=self.lon.min(),
                           east=self.lon.max(),
                           south=self.lat.min())

        req_vars = {'pr': 'precip_day',
                    'tmmx': 'tmax_day',
                    'tmmn': 'tmin_day',
                    'rmin': None,
                    'rmax': None,
                    'pet': 'potet_day',
                    'srad': 'swrad_day',
                    'vs': 'windspeed_day'}

        day_files = [v for k, v in req_vars.items() if v is not None] + ['humidity_day']
        day_files = [os.path.join(self.cfg.data_folder, d.replace('_', '.')) for d in day_files]
        cont_recs = [ControlRecord(os.path.basename(_file).replace('.', '_'), [_file]) for _file in day_files]
        [self.control_records.append(cr) for cr in cont_recs]

        for var_, day in req_vars.items():

            if all([os.path.exists(f) for f in day_files]):
                break

            gridmet = GridMet(start=s,
                              end=e,
                              variable=var_,
                              target_profile=self.raster_meta,
                              bbox=bounds,
                              clip_feature=[self.study_area.envelope])
            vals = gridmet.subset_daily_tif()

            # UNITS: see pg. 10 of PRMS Tables 5.2.0
            # see https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET#bands
            # for Gridmet units in Earth Engine
            # All are hard-coded, except precip takes its value from 'precip_units' parameter.

            if var_ == 'rmin':
                rmin = np.copy(vals)
                unit_name = '%'
                continue

            if var_ in ['tmmn', 'tmmx']:
                vals = np.where(vals > 0, vals - 273.15, np.zeros_like(vals))
                vals = (vals * 9 / 5) + 32.

                # TODO: somehow allowing temps > ~95 leads to an error where
                # PRMS finds a temp exceeding MAXTEMP (200.0; see src/prms/prms_constants.f90)
                # vals = np.where(vals > 120.0, np.ones_like(vals) * 105., vals)
                vals = np.where(vals > 90.0, np.ones_like(vals) * 90., vals)

                vals = np.where(vals < -40.0, np.ones_like(vals) * -40., vals)
                unit_name = 'F'

            if var_ == 'pr':
                vals = np.where(vals > 30.0, np.ones_like(vals) * 29.9, vals)
                if self.cfg.precip_units == 0:
                    unit_name = 'inches'
                    vals = vals / 25.4
                else:
                    unit_name = 'mm'

            if var_ == 'vs':
                unit_name = 'm/s'
                pass

            if var_ == 'pet':
                vals = vals / 25.4
                unit_name = 'inches'

            if var_ == 'rmax':
                rmax = np.copy(vals)
                vals = (rmin + rmax) / 2.
                _name = 'humidity'
                _file = os.path.join(self.cfg.data_folder, '{}.day'.format(_name))
                unit_name = '%'

            if var_ == 'srad':
                vals = vals * 0.000239006
                unit_name = 'Langleys'

            if day is not None:
                _name = day.split('_')[0]
                _file = os.path.join(self.cfg.data_folder, '{}.day'.format(_name))

            if os.path.exists(_file):
                continue

            rng = [str(x) for x in range(vals.shape[1] * vals.shape[2])]
            df = DataFrame(index=dt_range, columns=rng,
                           data=vals.reshape(vals.shape[0], vals.shape[1] * vals.shape[2]))

            time_div = ['Year', 'Month', 'day', 'hr', 'min', 'sec']
            df['Year'] = [i.year for i in df.index]
            df['Month'] = [i.month for i in df.index]
            df['day'] = [i.day for i in df.index]
            for t_ in time_div[3:]:
                df[t_] = [0 for _ in df.index]

            df = df[time_div + rng]

            with open(_file, 'w') as f:
                f.write('Generated: {} \n'.format(datetime.now()))
                f.write('Units: {} \n'.format(unit_name))
                f.write('{}\t{}\n'.format(_name, self.nhru))
                f.write('########################################\n')
                df.to_csv(f, sep=' ', header=False, index=False, float_format='%.2f')
            print('write {}'.format(_file))

    def build_model(self):
        self._build_grid()
        self.write_datafile()
        self.write_day_files()
        self.build_parameters()
        self.build_controls()
        self.write_parameters()
        self.write_control()

    def write_parameters(self):

        rain_snow_adj = np.ones((self.nhru * self.nmonths), dtype=float)

        self.data_params.append(ParameterRecord('rain_cbh_adj', rain_snow_adj,
                                                dimensions=[['nhru', self.nhru], ['nmonths', self.nmonths]],
                                                datatype=2))

        self.data_params.append(ParameterRecord('snow_cbh_adj', rain_snow_adj,
                                                dimensions=[['nhru', self.nhru], ['nmonths', self.nmonths]],
                                                datatype=2))

        self.data_params.append(ParameterRecord('elev_units', [self.cfg.elev_units], datatype=4))
        self.data_params.append(ParameterRecord('precip_units', [self.cfg.precip_units], datatype=4))
        self.data_params.append(ParameterRecord('temp_units', [self.cfg.temp_units], datatype=4))

        [self.parameters.add_record_object(rec) for rec in self.data_params]

        self.parameters.write(self.parameter_file)

    def write_control(self):
        self.control.add_record('elev_units', [self.cfg.elev_units])
        self.control.add_record('precip_units', [self.cfg.precip_units])
        self.control.add_record('temp_units', [self.cfg.temp_units])

        self.control.add_record('runoff_units', [self.cfg.runoff_units])

        self.control.add_record('orad_flag', [0])
        self.control.add_record('cbh_check_flag', [1])
        self.control.add_record('parameter_check_flag', [2])
        self.control.add_record('print_debug', [1])

        self.control.precip_module = ['climate_hru']
        self.control.temp_module = ['climate_hru']
        self.control.et_module = ['climate_hru']
        self.control.solrad_module = ['climate_hru']

        if self.control_records is not None:
            [self.control.add_record(rec.name, rec.values) for rec in self.control_records]

        self.control.write(self.control_file)


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================
