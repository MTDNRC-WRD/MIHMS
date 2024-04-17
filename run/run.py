import os

import numpy as np
import pandas as pd

from models.models import MontanaPrmsModel
from prep.prms.xyz_builder import XyzDistBuild
from utils.plotting import plot_stats


def build_model(config, return_model=False):
    prms_build = XyzDistBuild(config)
    prms_build.build_model()

    if return_model:
        return prms_build


def run_model_priors(root, config, project_str, verbose=False):
    prms_build = XyzDistBuild(config)

    data = os.path.join(root, '../example/data')
    project = os.path.join(data, '{}_{}'.format(project_str, prms_build.cfg.hru_cellsize))
    prms = MontanaPrmsModel(prms_build.control_file,
                            prms_build.parameter_file,
                            prms_build.data_file)

    param_dict = {rn: prms.parameters.get_values(rn) for rn in prms.parameters.record_names}

    if verbose:
        stdout_ = os.path.join(project, 'output', 'stdout.txt')
    else:
        stdout_ = None

    prms.run_model(stdout_)


def run_model_posteriors(root, config, project_str):

    prms_build = XyzDistBuild(config)

    data = os.path.join(root, '../example/data')
    project = os.path.join(data, '{}_{}'.format(project_str, prms_build.cfg.hru_cellsize))
    prms = MontanaPrmsModel(prms_build.control_file,
                            prms_build.parameter_file,
                            prms_build.data_file)

    # update model params with PEST++ parameter value proposal
    pest_d = os.path.join(project, 'pest')
    param_store = os.path.join(pest_d, 'mult')
    pfiles = [os.path.join(param_store, f) for f in os.listdir(param_store)]

    for f in pfiles:
        with open(f, 'r') as fp:
            lines = fp.readlines()
            strp = lines[1].strip().split(',')
            k = strp[-3]
            old_val = prms.parameters.get_values(k)
            v = float(strp[-1])
            new_val = v * np.ones_like(old_val)
            prms.parameters.set_values(k, new_val)

            if old_val is None:
                continue

            if not isinstance(old_val, float):
                old_disp = old_val.mean().item()
            else:
                old_disp = old_val

            print('{} set: {:.2f} to {:.2f}'.format(k, old_disp, v))

    prms.parameters.write(prms_build.parameter_file)
    prms.run_model()

    statvar = prms.get_statvar()
    pred = statvar[['basin_cms']].values
    np.savetxt(os.path.join(pest_d, 'obs_q.np'), pred)

    idx = statvar['runoff'] >= 0.0
    valid = statvar.loc[idx]
    pred, obs = valid['runoff'].mean(), valid['basin_cms'].mean()

    print('Measured Dates, Mean Pred: {:.1f}; Obs {:.1f}'.format(pred, obs))


def read_output(config, snow):
    prms_build = XyzDistBuild(config)

    prms = MontanaPrmsModel(prms_build.control_file,
                            prms_build.parameter_file,
                            prms_build.data_file)

    stats_uncal = prms.get_statvar(snow)
    fig_ = os.path.join(prms_build.cfg.output_folder, 'hydrograph_uncal.png')
    plot_stats(stats_uncal, fig_)


def compare_parameters(config, csv):
    prms_build = XyzDistBuild(config)

    prms = MontanaPrmsModel(prms_build.control_file,
                            prms_build.parameter_file,
                            prms_build.data_file)

    df = pd.read_csv(csv)
    df = df.mean(axis=0)

    param_names = prms.parameters.record_names

    comp_params = [x for x in df.index if x in param_names]

    for p in comp_params:
        mp = prms.parameters.get_values(p).mean()
        gf = df[p]
        print('p: {}, actual: {:.3f}, geofabric: {:.3f}'.format(p, mp, gf))

    pass


if __name__ == '__main__':
    project_ = 'smith'
    res = 3000
    wspace = os.path.dirname(os.path.abspath(__file__))
    conf = os.path.join(wspace, '../example/data', '{}_{}'.format(project_, res), '{}_parameters.toml'.format(project_))
    # build_model(conf)
    # run_model(wspace, conf, project_, verbose=True)

    swe = '/home/dgketchum/PycharmProjects/MIHMS/example/data/smith_3000/input/snodas.json'
    read_output(conf, swe)
# ========================= EOF ====================================================================
