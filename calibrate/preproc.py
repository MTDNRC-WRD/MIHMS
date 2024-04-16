"""
Builds a PRMS model based on an initial set of parameters and writes the observations to a file accessible by PEST++.
"""

import os
import json

import pandas as pd
import numpy as np

from example.run import build_model

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def preproc(project_root, config, write_model=False, **kwargs):
    if write_model:
        build_model(config, return_model=False)

    in_file = os.path.join(project_root, 'input', 'inputs.csv')
    data = pd.read_csv(in_file)
    data.index = list(range(data.shape[0]))
    obs = [c for c in data.columns if 'runoff' in c][0]
    data['q'] = data[obs]
    data = data[['q']]
    print('preproc mean: {}'.format(np.nanmean(data.values)))
    _file = os.path.join(project_root, 'obs_q.np')
    np.savetxt(_file, data.values)
    print('Wrote obs to {}'.format(_file))

    # since swe data is incomplete, we'll write it to a DataFrame with a DateTimeIndex
    if kwargs:
        for k, v in kwargs.items():
            with open(v, 'r') as fp:
                obs = json.load(fp)
            obs = {kk: vv['0'] for kk, vv in obs.items()}
            dt = pd.DatetimeIndex([pd.to_datetime(kk, format='%Y%m%d') for kk, vv in obs.items()])
            obs = [vv for kk, vv in obs.items()]
            obs = pd.DataFrame(data=obs, index=dt)
            _file = os.path.join(project_root, 'obs_{}.csv'.format(k))
            obs.to_csv(_file)
    pass


if __name__ == '__main__':
    project = 'smith_3000'
    root = '/home/dgketchum/PycharmProjects/MIHMS/example'
    proj_dir = os.path.join(root, 'data', project)
    conf = os.path.join(proj_dir, 'smith_parameters.toml')

    swe_series = '/home/dgketchum/PycharmProjects/MIHMS/example/data/smith_3000/input/snodas.json'
    observations_ = {'swe': swe_series}
    preproc(proj_dir, conf, **observations_)
