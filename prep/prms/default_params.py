import os
from pprint import pprint


def get_params():
    params = {'basin_sum': {'outlet_sta': 0},
              'ddsolrad': {'dday_intcp': -40.0,
                           'dday_slope': 0.4,
                           'radadj_intcp': 1.0,
                           'radadj_slope': 0.0,
                           'tmax_index': 50.0},
              'intcp': {'epan_coef': None},
              'obs': {'rain_code': None, 'runoff_units': None},
              'potet_jh': {'jh_coef': None},
              'soilzone': {'pref_flow_infil_frac': None, 'ssstor_init': None},
              'xyz_dist': {'adjust_rain': None,
                           'adjust_snow': None,
                           'conv_flag': None,
                           'max_lapse': None,
                           'min_lapse': None,
                           'ppt_add': None,
                           'ppt_div': None,
                           'ppt_lapse': None,
                           'psta_freq_nuse': None,
                           'psta_month_ppt': None,
                           'solrad_elev': None,
                           'tmax_add': None,
                           'tmax_allrain': None,
                           'tmax_allrain_dist': None,
                           'tmax_allsnow_dist': None,
                           'tmax_div': None,
                           'tmin_add': None,
                           'tmin_div': None,
                           'tsta_month_max': None,
                           'tsta_month_min': None,
                           'x_add': None,
                           'x_div': None,
                           'y_add': None,
                           'y_div': None,
                           'z_add': None,
                           'z_div': None}}
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
