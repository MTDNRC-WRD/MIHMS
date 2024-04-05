import os

import pandas as pd

from models.models import MontanaPrmsModel
from prep.prms.xyz_builder import XyzDistBuild
from utils.plotting import plot_stats


def build_model(config, return_model=False):
    prms_build = XyzDistBuild(config)
    prms_build.build_model()

    if return_model:
        return prms_build


def run_model(root, config, project_str, verbose=False):
    prms_build = XyzDistBuild(config)

    data = os.path.join(root, 'data')
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
    conf = os.path.join(wspace, 'data', '{}_{}'.format(project_, res), '{}_parameters.toml'.format(project_))
    # build_model(conf)
    # run_model(wspace, conf, project_, verbose=True)

    swe = '/home/dgketchum/PycharmProjects/MIHMS/example/data/smith_3000/input/snodas.json'
    read_output(conf, swe)
# ========================= EOF ====================================================================
