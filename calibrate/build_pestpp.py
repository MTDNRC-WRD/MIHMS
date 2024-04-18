import os
import shutil

import numpy as np
import pandas as pd
from pyemu import Pst, Matrix
from pyemu.utils import PstFrom

from prep.prms.default_params import tuning_parameters, get_params


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

    pest.add_observations(kwargs['q_obs']['file'], insfile=kwargs['q_obs']['insfile'])

    qdf = pd.read_csv(input_data, index_col=None, parse_dates=True)
    swe_df = qdf.copy()
    qdf['dummy_idx'] = [obsnme_str.format(j) for j in range(qdf.shape[0])]
    valid = [i for i, r in qdf.iterrows() if r[qdf.columns[0]]]
    valid = qdf['dummy_idx'].loc[valid]

    d = pest.obs_dfs[0].copy()
    d['weight'] = 0.0
    d.loc[valid, 'weight'] = 1.0
    d.loc[np.isnan(d['obsval']), 'weight'] = 0.0
    d.loc[np.isnan(d['obsval']), 'obsval'] = -99.0
    d['idx'] = d.index.map(lambda i: int(i.split(':')[3].split('_')[0]))
    d = d.sort_values(by='idx')
    d.drop(columns=['idx'], inplace=True)

    pest.obs_dfs[0] = d

    # ======= Snow Observations ==================
    obsnme_str = 'oname:obs_swe_otype:arr_i:{}_j:0'

    pest.add_observations(kwargs['swe_obs']['file'], insfile=kwargs['swe_obs']['insfile'])

    # only weight swe Nov - Apr
    # reuse input df from above for it's time information
    # we forced the q and swe arrays into the same time in preproc.py
    swe_df['dummy_idx'] = [obsnme_str.format(j) for j in range(swe_df.shape[0])]
    swe_df.index = [pd.to_datetime(dt) for dt in swe_df['date']]
    valid = [ix for ix, r in swe_df.iterrows() if ix.month in [11, 12, 1, 2, 3, 4]]
    valid = swe_df['dummy_idx'].loc[valid]

    d = pest.obs_dfs[1].copy()
    d['weight'] = 0.0

    # TODO: adjust as needed for phi visibility of eta vs. swe
    d.loc[valid, 'weight'] = 0.03
    d.loc[np.isnan(d['obsval']), 'weight'] = 0.0
    d.loc[np.isnan(d['obsval']), 'obsval'] = -99.0

    d['idx'] = d.index.map(lambda i: int(i.split(':')[3].split('_')[0]))
    d = d.sort_values(by='idx')
    d.drop(columns=['idx'], inplace=True)
    pest.obs_dfs[1] = d

    ofiles = [str(x).replace('obs', 'pred') for x in pest.output_filenames]
    pest.output_filenames = ofiles
    os.makedirs(os.path.join(pest_dir, 'pred'))

    pest.py_run_file = 'custom_forward_run.py'
    pest.mod_command = 'python custom_forward_run.py'

    pest.build_pst(version=2)

    # the build function wrote a generic python runner that we replace with our own
    # with some work, pymeu build can do this for us
    auto_gen = os.path.join(pest_dir, 'custom_forward_run.py')
    runner = kwargs['python_script']
    shutil.copyfile(runner, auto_gen)

    # clean up the new pest directory
    for dd in ['master', 'workers']:
        try:
            shutil.rmtree(os.path.join(pest_dir, dd))
        except FileNotFoundError:
            continue

    # hack to write measurement std post-build, which if not included, 'weight' will be interpreted as std dev
    # this will be used to add noise to non-zero weighted obs data in e.g., tongue.obs+noise.csv
    # TODO: pre-compute observation ensembles, implement autocorrelated transient noise
    # see: github.com/gmdsi/GMDSI_notebooks/blob/main/tutorials/part2_02_obs_and_weights/freyberg_obs_and_weights.ipynb
    pst = Pst(os.path.join(pest.new_d, '{}.pst'.format(os.path.basename(model_dir))))
    obs = pst.observation_data
    obs['standard_deviation'] = np.nan
    obs.loc[[i for i in obs.index if 'q' in i], 'standard_deviation'] = obs['obsval'] * 0.05
    obs.loc[[i for i in obs.index if 'swe' in i], 'standard_deviation'] = obs['obsval'] * 0.02

    # add time information
    obs['time'] = [float(i.split(':')[3].split('_')[0]) for i in obs.index]

    pst.write(pst.filename, version=2)


def build_localizer(pst_file):

    tuning_params = tuning_parameters()

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

    obs_type = [i.split(':')[1].split('_')[1] for i in df.index]
    for param, obs in tuning_params.items():
        idx = [i for t, i in zip(obs_type, df.index) if t == obs]
        cols = [c for c in df.columns if param in c]
        localizer.loc[idx, cols] = 1.0

    mat_file = os.path.join(os.path.dirname(pst_file), 'loc.mat')
    Matrix.from_dataframe(localizer).to_ascii(mat_file)

    pst.write(pst_file, version=2)


def set_control_settings(pst_file):
    pst = Pst(pst_file)
    pst.pestpp_options["ies_localizer"] = "loc.mat"
    pst.pestpp_options["ies_num_reals"] = 100

    # pestpp-ies has a more rigorous pre-run testing functionality called when noptmax = -2
    pst.control_data.noptmax = -2

    pst.write(pst_file, version=2)


def params_dict_from_csv(_file):

    q_ins = 'q.ins'
    swe_ins = 'swe.ins'

    q_obs_file = 'obs/obs_q.np'
    swe_obs_file = 'obs/obs_swe.np'

    pdct = {'q_obs': {'file': q_obs_file,
                      'insfile': q_ins},
            'swe_obs': {'file': swe_obs_file,
                        'insfile': swe_ins},
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


def params_dict_from_defaults(dst_file):

    q_ins = 'q.ins'
    swe_ins = 'swe.ins'

    q_obs_file = 'obs/obs_q.np'
    swe_obs_file = 'obs/obs_swe.np'

    pdct = {'q_obs': {'file': q_obs_file,
                      'insfile': q_ins},
            'swe_obs': {'file': swe_obs_file,
                        'insfile': swe_ins},
            }

    pars = {}
    dct = get_params()
    tunable_params = tuning_parameters()

    columns = ['param', 'module', 'lower_bound', 'upper_bound', 'initial_value',
               'use_cols', 'use_rows', 'file']
    df = pd.DataFrame(columns=columns, index=list(range(1000)))

    ct = 0
    for k, v in dct.items():
        for kk, vv in v.items():

            if kk not in tunable_params.keys():
                continue

            try:
                pars[kk] = {'file': dst_file,
                            'initial_value': vv[2],
                            'lower_bound': vv[0],
                            'upper_bound': vv[1],
                            'pargp': kk,
                            'index_cols': 0,
                            'use_cols': 3,
                            'use_rows': ct}

                df.loc[ct] = pars[kk]
                df.loc[ct, 'param'] = kk
                df.loc[ct, 'module'] = k
                ct += 1

            except TypeError:
                print('Improper formatting, {}, {}'.format(k, kk))

        a = 1

    missing = []
    for k in tunable_params:
        if k not in pars.keys():
            missing.append(k)
    print('Consider adding tunable parameters to defaults: {}'.format(missing))

    pdct.update({'pars': pars})
    df.dropna(axis=0, how='all', inplace=True)
    df.to_csv(dst_file)
    return pdct


if __name__ == '__main__':
    project = 'smith_3000'
    src = '/home/dgketchum/PycharmProjects/MIHMS'
    root = os.path.join(src, 'example', 'data')
    d = os.path.join(root, '{}'.format(project))

    data = os.path.join(d, 'input')
    input_csv = os.path.join(data, 'inputs.csv')

    pp_dir = os.path.join(d, 'pest')
    pest_file = os.path.join(pp_dir, '{}.pst'.format(project))

    dst_parms_file = os.path.join(d, 'prms_params.csv')
    dct_ = params_dict_from_defaults(dst_parms_file)

    python_script = os.path.join(src, 'calibrate', 'custom_forward_run.py')
    # noinspection PyTypedDict
    dct_.update({'python_script': python_script})

    build_pest(d, pp_dir, input_csv, **dct_)

    build_localizer(pest_file)

    set_control_settings(pest_file)
# ========================= EOF ====================================================================
