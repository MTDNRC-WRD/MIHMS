"""
Builds a PRMS model based on an initial set of parameters and writes the observations to a file accessible by PEST++.
"""

import os
import pandas as pd
import numpy as np

from example.run import build_model

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

project = 'smith_3000'


def preproc():
    project_ = 'smith'
    wspace = os.path.dirname(os.path.abspath(__file__))
    conf = os.path.join(wspace, '{}_parameters.toml'.format(project_))

    model = build_model(conf, return_model=True)
    in_file = os.path.join(model.cfg.data_folder, 'inputs.csv')
    data = pd.read_csv(in_file)
    data.index = list(range(data.shape[0]))
    obs = [c for c in data.columns if 'runoff' in c][0]
    data['q'] = data[obs]
    data = data[['q']]
    print('preproc mean: {}'.format(np.nanmean(data.values)))
    _file = os.path.join(wspace, 'obs_q.np')
    np.savetxt(_file, data.values)
    print('Wrote obs to {}'.format(_file))


preproc()
