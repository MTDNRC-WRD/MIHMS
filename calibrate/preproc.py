"""
Builds a PRMS model based on an initial set of parameters and writes the observations to a file accessible by PEST++.
"""

import os
import json

import pandas as pd
import numpy as np

from run.run import build_model

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def preproc(project_root, config, write_model=False, **kwargs):
    if write_model:
        build_model(config, return_model=False)

    in_file = os.path.join(project_root, 'input', 'inputs.csv')
    data = pd.read_csv(in_file, index_col='date', parse_dates=True)
    obs = [c for c in data.columns if 'runoff' in c][0]
    data['q'] = data[obs]
    q = data[['q']]
    print('preproc mean: {}'.format(np.nanmean(q.values)))
    obs_dir = os.path.join(project_root, 'obs')
    if not os.path.isdir(obs_dir):
        os.mkdir(obs_dir)
    _file = os.path.join(obs_dir, 'obs_q.np')
    np.savetxt(_file, q.values)
    print('Wrote obs to {}'.format(_file))

    # this can be modified to read in more than just swe
    if kwargs:
        for k, v in kwargs.items():
            with open(v, 'r') as fp:
                obs = json.load(fp)
            obs = {kk: vv['0'] for kk, vv in obs.items()}
            dt = pd.DatetimeIndex([pd.to_datetime(kk, format='%Y%m%d') for kk, vv in obs.items()], tz='UTC')
            obs = [vv for kk, vv in obs.items()]
            obs = pd.Series(data=obs, index=dt, name='swe')
            match = [i for i in obs.index if i in data.index]
            data.loc[match, 'swe'] = obs.loc[match]
            _file = os.path.join(obs_dir, 'obs_{}.np'.format(k))
            np.savetxt(_file, data['swe'].values)
            print('Wrote obs to {}'.format(_file))
    pass


if __name__ == '__main__':
    project = 'smith_3000'
    root = '/home/dgketchum/PycharmProjects/MIHMS/example'
    proj_dir = os.path.join(root, 'data', project)
    conf = os.path.join(proj_dir, 'smith_parameters.toml')

    swe_series = '/home/dgketchum/PycharmProjects/MIHMS/example/data/smith_3000/input/snodas.json'
    observations_ = {'swe': swe_series}
    preproc(proj_dir, conf, write_model=True, **observations_)
