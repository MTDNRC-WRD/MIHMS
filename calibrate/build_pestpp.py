import os

import numpy as np
import pandas as pd
from pyemu import Pst, Matrix
from pyemu.utils import PstFrom


def build_pest(model_dir, pest_dir, input_data, **kwargs):
    pest = PstFrom(model_dir, pest_dir, remove_existing=True)

    for k, v in kwargs['pars'].items():
        _file = v.pop('file')
        if v['lower_bound'] <= 0.0:
            transform = 'none'
        else:
            transform = 'log'
        pest.add_parameters(_file, 'constant', transform=transform, alt_inst_str='{}_'.format(k), **v)

    # ======= Discharge Observations ==================
    obsnme_str = 'oname:obs_q_otype:arr_i:{}_j:0'

    pest.add_observations(kwargs['obs']['file'], insfile=kwargs['obs']['insfile'])

    qdf = pd.read_csv(input_data, index_col=None, parse_dates=True)
    qdf['dummy_idx'] = [obsnme_str.format(j) for j in range(qdf.shape[0])]
    valid = [i for i, r in qdf.iterrows() if r[qdf.columns[0]]]
    valid = qdf['dummy_idx'].loc[valid]

    d = pest.obs_dfs[0].copy()
    d['weight'] = 0.0
    d.loc['weight', valid] = 1.0
    d.loc['weight', np.isnan(d['obsval'])] = 0.0
    d.loc['obsval', np.isnan(d['obsval'])] = -99.0
    d['idx'] = d.index.map(lambda i: int(i.split(':')[3].split('_')[0]))
    d = d.sort_values(by='idx')
    d.drop(columns=['idx'], inplace=True)

    pest.obs_dfs[0] = d

    # ======= Snow Observations ==================
    obsnme_str = 'oname:obs_swe_otype:arr_i:{}_j:0'

    pest.add_observations(kwargs['swe_obs']['file'][j], insfile=kwargs['swe_obs']['insfile'][j])

    # only weight swe Nov - Apr
    swe_df = pd.read_csv(kwargs['inputs'][i], index_col=0, parse_dates=True)
    swe_df['dummy_idx'] = [obsnme_str.format(fid, j) for j in range(swe_df.shape[0])]
    valid = [ix for ix, r in swe_df.iterrows() if ix.month in [11, 12, 1, 2, 3, 4]]
    valid = swe_df['dummy_idx'].loc[valid]

    d = pest.obs_dfs[1].copy()
    d['weight'] = 0.0

    # TODO: adjust as needed for phi visibility of eta vs. swe
    d.loc['weight', valid] = 0.03
    d.loc['weight', np.isnan(d['obsval'])] = 0.0
    d.loc['obsval', np.isnan(d['obsval'])] = -99.0

    d['idx'] = d.index.map(lambda i: int(i.split(':')[3].split('_')[0]))
    d = d.sort_values(by='idx')
    d.drop(columns=['idx'], inplace=True)

    pest.obs_dfs[1] = d

    pest.py_run_file = 'custom_forward_run.py'
    pest.mod_command = 'python custom_forward_run.py'

    pest.build_pst()


def build_localizer(pst_file):
    et_params = ['aw', 'rew', 'tew', 'ndvi_alpha', 'ndvi_beta', 'mad']
    snow_params = ['swe_alpha', 'swe_beta']

    par_relation = {'eta': et_params, 'swe': snow_params}

    pst = Pst(pst_file)

    pdict = {}
    for i, r in pst.parameter_data.iterrows():
        if r['pargp'] not in pdict.keys():
            pdict[r['pargp']] = [r['parnme']]
        else:
            pdict[r['pargp']].append(r['parnme'])

    pnames = pst.parameter_data['parnme'].values

    df = Matrix.from_names(pst.nnz_obs_names, pnames).to_dataframe()

    localizer = df.copy()

    sites = list(set([i.split('_')[2] for i in df.index]))

    for s in sites:
        for ob_type, params in par_relation.items():
            idx = [i for i in df.index if '{}_{}'.format(ob_type, s) in i]
            cols = list(np.array([[c for c in df.columns if '{}_{}'.format(p, s) in c] for p in params]).flatten())
            localizer.loc[idx, cols] = 1.0

    mat_file = os.path.join(os.path.dirname(pst_file), 'loc.mat')
    Matrix.from_dataframe(localizer).to_ascii(mat_file)

    pst.pestpp_options["ies_localizer"] = "loc.mat"
    pst.pestpp_options["ies_num_reals"] = 100

    # pestpp-ies has a more rigorous pre-run testing functionality called when noptmax = -2
    pst.control_data.noptmax = -2

    pst.write(pst_file, version=2)


def params_dict_from_csv(_file):

    q_obs_file = 'obs/obs_q.np'
    swe_obs_file = 'obs/obs_swe.np'

    pdct = {'q_obs': {'file': q_obs_file,
                      'insfile': ins},
            'swe_obs': {'file': swe_obs_file,
                        'insfile': ins}
            }

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
    pest_file = os.path.join(pp_dir, '{}.pst'.format(project))

    ins = '{}.ins'.format(project)
    p_file = os.path.join(d, 'prms_params_rio_hondo.csv')

    # this just prints out the params dict for use below
    dct = params_dict_from_csv(p_file)

    build_pest(d, pp_dir, input_csv, **dct)

    build_localizer(pest_file)
# ========================= EOF ====================================================================
