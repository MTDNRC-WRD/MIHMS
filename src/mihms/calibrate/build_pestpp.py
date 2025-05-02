import os

import numpy as np
import pandas as pd
from pyemu.utils import PstFrom


def build_pest(model_dir, pest_dir, input_data, **kwargs):
    pest = PstFrom(model_dir, pest_dir, remove_existing=True)

    for k, v in kwargs['pars'].items():
        _file = v.pop('file')
        pest.add_parameters(_file, 'constant', **v)

    pest.add_observations(kwargs['obs']['file'], insfile=kwargs['obs']['insfile'])

    idf = pd.read_csv(input_data, index_col=None, parse_dates=True)
    idf['dummy_idx'] = ['obs_q_{}'.format(str(i).rjust(6, '0')) for i in range(idf.shape[0])]
    valid = [i for i, r in idf.iterrows() if r[idf.columns[0]]]
    valid = idf['dummy_idx'].loc[valid]

    d = pest.obs_dfs[0].copy()
    d['weight'] = 0.0
    d['weight'].loc[valid] = 1.0
    d['weight'].loc[np.isnan(d['obsval'])] = 0.0
    d['obsval'].loc[np.isnan(d['obsval'])] = -99.0
    pest.obs_dfs[0] = d

    pest.py_run_file = 'custom_forward_run.py'
    pest.mod_command = 'python custom_forward_run.py'

    pest.build_pst(write_py_file=False)


def params_dict_from_csv(_file):
    pdct = {'obs': {'file': 'obs_q.np',
                    'insfile': ins}}

    df = pd.read_csv(_file, header=None)
    df.columns = ['param', 'Desc', 'Module', 'lower_bound', 'upper_bound', 'initial_value']

    pars = {}

    for i, r in df.iterrows():
        p = r['param']
        pars[p] = {'file': _file,
                   'initial_value': r['initial_value'],
                   'lower_bound': r['lower_bound'],
                   'upper_bound': r['upper_bound'],
                   'pargp': p,
                   'index_cols': 0,
                   'use_cols': 5,
                   'use_rows': i}

    pdct.update({'pars': pars})
    return pdct


if __name__ == '__main__':
    project = 'smith_3000'
    root = '/home/dgketchum/PycharmProjects/MIHMS/example/data'
    d = os.path.join(root, '{}'.format(project))

    data = os.path.join(d, 'input')
    input_csv = os.path.join(data, 'inputs.csv')

    pp_dir = os.path.join(d, 'pest')

    ins = '{}.ins'.format(project)
    p_file = os.path.join(d, 'prms_params_rio_hondo.csv')

    # this just prints out the params dict for use below
    dct = params_dict_from_csv(p_file)

    build_pest(d, pp_dir, input_csv, **dct)
# ========================= EOF ====================================================================
