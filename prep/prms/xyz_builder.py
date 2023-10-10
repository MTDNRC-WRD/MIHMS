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
from prep.met_data import met_zones_geometries, attribute_precip_zones, calculate_monthly_lapse_rates
from prep.prms.standard_build import StandardPrmsBuild
from utils.bounds import GeoBounds
from utils.thredds import GridMet

from prep.prms import NOT_NEEDED_XYZ

warnings.simplefilter(action='ignore', category=DeprecationWarning)


class XyzDistBuild(StandardPrmsBuild):

    def __init__(self, config):
        StandardPrmsBuild.__init__(self, config)
        self.data_file = os.path.join(self.cfg.data_folder,
                                      '{}_xyz.data'.format(self.proj_name_res))

    def write_datafile(self):

        self.nmonths = 12

        ghcn = self.cfg.prms_data_ghcn
        stations = self.cfg.prms_data_stations

        if not os.path.exists(stations):
            met_zones_geometries(self.cfg.all_stations, self.domain_shapefile, self.cfg.prms_data_stations,
                                 zones_out=self.cfg.prms_data_zones)

        with open(self.cfg.prms_data_gages, 'r') as js:
            gages = json.load(js)

        with open(stations, 'r') as js:
            stations = json.load(js)

        if self.cfg.selected_stations:
            stations = {k: stations[k] for k in self.cfg.selected_stations}
        if self.cfg.selected_gage:
            gages = {self.cfg.selected_gage: gages[self.cfg.selected_gage]}

        units = 'metric' if self.cfg.precip_units == '1' else 'standard'
        new_stations = write_basin_datafile(gages=gages, data_file=self.data_file, stations=stations,
                                            ghcn_data=ghcn, out_csv=None, units=units, return_modified=True)
        stations = {k: v for k, v in stations.items() if k in new_stations.keys()}

        sta_iter = sorted([(v['zone'], v) for k, v in stations.items()], key=lambda x: x[0])
        tsta_elev, tsta_nuse, tsta_x, tsta_y, psta_elev = [], [], [], [], []
        for _, val in sta_iter:

            if self.cfg.elev_units == '1':
                elev = val['elev'] / 0.3048
            else:
                elev = val['elev']

            tsta_elev.append(elev)
            tsta_nuse.append(1)
            tsta_x.append(val['proj_coords'][1])
            tsta_y.append(val['proj_coords'][0])
            psta_elev.append(elev)

        self.data_params = [ParameterRecord('nrain', values=[len(tsta_x)], datatype=1),

                            ParameterRecord('ntemp', values=[len(tsta_x)], datatype=1),

                            ParameterRecord('psta_elev', np.array(psta_elev, dtype=float).ravel(),
                                            dimensions=[['nrain', len(psta_elev)]], datatype=2),

                            ParameterRecord('psta_nuse', np.array(tsta_nuse, dtype=int).ravel(),
                                            dimensions=[['nrain', len(tsta_nuse)]], datatype=1),

                            # ParameterRecord(name='ndist_psta', values=[len(tsta_nuse), ], datatype=1),

                            ParameterRecord('psta_x', np.array(tsta_x, dtype=float).ravel(),
                                            dimensions=[['nrain', len(tsta_x)]], datatype=2),

                            ParameterRecord('psta_y', np.array(tsta_y, dtype=float).ravel(),
                                            dimensions=[['nrain', len(tsta_y)]], datatype=2),

                            ParameterRecord('tsta_elev', np.array(tsta_elev, dtype=float).ravel(),
                                            dimensions=[['ntemp', len(tsta_elev)]], datatype=2),

                            ParameterRecord('tsta_nuse', np.array(tsta_nuse, dtype=int).ravel(),
                                            dimensions=[['ntemp', len(tsta_nuse)]], datatype=1),

                            ParameterRecord(name='ndist_tsta', values=[len(tsta_nuse), ], datatype=1),

                            ParameterRecord('tsta_x', np.array(tsta_x, dtype=float).ravel(),
                                            dimensions=[['ntemp', len(tsta_x)]], datatype=2),

                            ParameterRecord('tsta_y', np.array(tsta_y, dtype=float).ravel(),
                                            dimensions=[['ntemp', len(tsta_y)]], datatype=2),

                            bu.tmax_adj(self.nhru),

                            bu.tmin_adj(self.nhru),

                            ParameterRecord(name='nobs', values=[1, ], datatype=1),
                            ]

        if self.cfg.temp_units == 1:
            allrain_max = np.ones((self.nhru * self.nmonths)) * 3.3
        else:
            allrain_max = np.ones((self.nhru * self.nmonths)) * 38.0

        self.data_params.append(ParameterRecord('tmax_allrain_sta', allrain_max,
                                                dimensions=[['nhru', self.nhru], ['nmonths', self.nmonths]],
                                                datatype=2))

        self.data_params.append(ParameterRecord('snowpack_init',
                                                np.ones_like(self.ksat).ravel(),
                                                dimensions=[['nhru', self.nhru]],
                                                datatype=2))

        self.data = PrmsData.load_from_file(self.data_file)

    def build_model(self):
        self._build_grid()
        self.build_parameters()
        self.write_build_shapefile()
        # data file depends on parameters being built already
        self.write_datafile()
        self.build_controls()
        self.write_parameters()
        self.write_control()

    def write_parameters(self):

        if self.data_params is not None:
            [self.parameters.add_record_object(rec) for rec in self.data_params]

        for p in NOT_NEEDED_XYZ:
            self.parameters.remove_record(p)

        self.parameters.write(self.parameter_file)

    def write_control(self):
        # 0: standard; 1: SI/metric

        self.control.add_record('elev_units', [self.parameters.elev_units.values[0]])
        self.control.add_record('precip_units', [self.parameters.precip_units.values[0]])
        self.control.add_record('temp_units', [self.parameters.temp_units.values[0]])
        self.control.add_record('runoff_units', [int(self.cfg.runoff_units)])

        self.control.precip_module = ['xyz_dist']
        self.control.temp_module = ['xyz_dist']
        self.control.et_module = ['potet_jh']
        self.control.solrad_module = ['ccsolrad']

        if self.control_records is not None:
            [self.control.add_record(rec) for rec in self.control_records]

        self.control.write(self.control_file)


if __name__ == '__main__':
    pass


# Username: david.ketchum@umt.edu
# Employee Email Address:  david.ketchum@mso.umt.edu
# ========================= EOF ====================================================================
