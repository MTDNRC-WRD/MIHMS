import os
import warnings

from gsflow.prms import PrmsParameters

warnings.simplefilter(action='ignore', category=DeprecationWarning)


def read_calibration(params_dir):
    params_ = ['adjmix_rain',
               'tmax_allsnow',
               'srain_intcp',
               'wrain_intcp',
               'cecn_coef',
               'emis_noppt',
               'freeh2o_cap',
               'potet_sublim',
               'carea_max',
               'smidx_coef',
               'smidx_exp',
               'fastcoef_lin',
               'fastcoef_sq',
               'pref_flow_den',
               'sat_threshold',
               'slowcoef_lin',
               'slowcoef_sq',
               'soil_moist_max',
               'soil_rechr_max',
               'soil2gw_max',
               'ssr2gw_exp',
               'ssr2gw_rate',
               'transp_tmax',
               'gwflow_coef']

    dct = {k: None for k in params_}
    l = sorted([os.path.join(params_dir, x) for x in os.listdir(params_dir)])
    first = True
    for i, ll in enumerate(l):
        params = PrmsParameters.load_from_file(ll)
        if first:
            for p in params_:
                vals = params.get_values(p)
                dct[p] = vals.mean()
            first = False
            continue

        for p in params_:
            vals = params.get_values(p)
            new_val = vals.mean()
            delta = new_val - dct[p]
            print('{:.3f} {} delta'.format(delta, p))
            dct[p] = new_val
            if p == 'ssr2gw_exp':
                pass

        if i == len(l) - 1:
            print('final luca parameter values')
            for p in params_:
                vals = params.get_values(p)
                new_val = vals.mean()
                print('{:.3f} {} final values'.format(new_val, p))


if __name__ == '__main__':
    pass

# ========================= EOF ====================================================================
