import os
from pprint import pprint


def get_params():
    """Dict of the form param: lower bound, upper bound, initial value"""
    params = {'basin_sum': {'outlet_sta': (None, None, 0)},

              'required': {
                  'fastcoef_lin': (0.0, 1.0, 0.1),
                  'fastcoef_sq': (0.0, 1.0, 0.8),
                  'pref_flow_den': (0.0, 0.5, 0.0),
                  'sat_threshold': (0.0001, 999.0, 999.0),
                  'slowcoef_lin': (0.0, 1.0, 0.015),
                  'slowcoef_sq': (0.0, 1.0, 0.1),
                  'soil_moist_max': (0.00001, 20.0, 2.0),
                  'soil2gw_max': (0.0, 5.0, 0.0),
                  'soil2gw_exp': (0.0, 3.0, 1.0),
                  'soil2gw_rate': (0.0001, 999.0, 0.1),
                  'gwflow_coef': (0.0, 0.5, 0.015),
                  'gwsink_coef': (0.0, 1.0, 0.01),
                  'gwstor_init': (0.0, 50.0, 5.0),
                  'gwstor_min': (0.0, 1.0, 0.01),

              },

              'ddsolrad': {'dday_intcp': (-60.0, 10, -40.0),
                           'dday_slope': (0.1, 1.4, 0.4),
                           'radadj_intcp': (0.0, 1.0, 1.0),
                           'radadj_slope': (0.0, 1.0, 0.0),
                           'tmax_index': (-10.0, 30.0, 17.0)},

              # 'intcp': {'epan_coef': None},

              # 'obs': {'rain_code': None, 'runoff_units': None},

              'potet_jh': {'jh_coef': (-0.5, 1.5, 0.014)},

              'soilzone': {
                  'ssstor_init': (0.0, 10.0, 0.0),
                  'soil_rechr_max_frac': (0.00001, 1.0, 1.0),
                  'soil_moist_init_frac': (0.0, 1.0, 0.0),
                  'soil_rechr_init_frac': (0.0, 1.0, 0.0),
                  'ssstor_init_frac': (0.0, 1.0, 0.0)
              },

              'xyz_dist': {'adjust_rain': (-0.5, 3.0, -0.4),
                           'adjust_snow': (-0.5, 3.0, -0.4),
                           'conv_flag': 0,
                           'max_lapse': (-100.0, 100.0, 0.0),
                           'min_lapse': (-100.0, 100.0, 0.0),

                           # 'nrain': None, # set in builder
                           # 'ntemp': None, # set in builder
                           # 'nlapse': None, # set in builder
                           # 'rain_code': (None, None, 2), # use default
                           # 'hru_x': None, # set in builder
                           # 'hru_y': None, # set in builder

                           'ppt_add': (-10., 10., 0.0),
                           'ppt_div': (-10., 10., 1.0),
                           # 'psta_elev': None, # set in builder
                           'ppt_lapse': (-10., 10., 0.0),
                           # 'psta_freq_nuse': None, # set in builder
                           'psta_month_ppt': (0.0, 20.0, 0.0),
                           # 'psta_x': None, # set in builder
                           # 'psta_y': None, # set in builder

                           # 'solrad_elev': None,  # TODO add solrad station?

                           'tmax_add': (-10., 10., 0.0),
                           'tmax_adj': (-10., 10., 0.0),
                           'tmax_allrain': (0.0, 3.0, 1.5),
                           'tmax_allrain_dist': (0.0, 3.0, 1.5),
                           'tmax_allsnow_dist': (-1.0, 2.0, 0.0),
                           'tmax_div': (-10., 10., 1.0),

                           'tmin_add': (-10., 10., 0.0),
                           'tmin_adj': (-10., 10., 0.0),
                           'tmin_div': (-10., 10., 1.0),

                           # 'tsta_month_max': None,  # TODO add climatology info
                           # 'tsta_month_min': None,  # TODO add climatology info

                           'x_add': (-1.0e7, 1.0e7, 0.0),
                           'x_div': (-1.0e7, 1.0e7, 1.0),
                           'y_add': (-1.0e7, 1.0e7, 0.0),
                           'y_div': (-1.0e7, 1.0e7, 1.0),
                           'z_add': (-1.0e7, 1.0e7, 0.0),
                           'z_div': (-1.0e7, 1.0e7, 1.0)}}
    return params


def collect_missing_params(stdout):
    missing = {}
    with open(stdout, 'r') as fp:
        for line in fp.readlines():
            if 'is used by module' in line:
                l = line.split(' ')
                param, mod = l[2], l[7]
                if mod not in missing.keys():
                    missing[mod] = {param: None}
                else:
                    missing[mod][param] = None

    pprint(missing)


if __name__ == '__main__':
    collect_missing_params('/home/dgketchum/PycharmProjects/MIHMS/example/'
                           'data/uyws_carter_3000/output/stdout.txt')
# ========================= EOF ====================================================================
