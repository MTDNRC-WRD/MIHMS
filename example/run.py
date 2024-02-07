import os

import pandas as pd

from models.models import MontanaPrmsModel
from prep.prms.xyz_builder import XyzDistBuild
from utils.plotting import plot_stats


def run_model(root, config):

    prms_build = XyzDistBuild(config)
    prms_build.build_model()
    data = os.path.join(root, 'data')
    # matplotlib.use('TkAgg')

    project = os.path.join(data, '{}_{}'.format(prms_build.cfg.project_name, prms_build.cfg.hru_cellsize))
    luca_dir = os.path.join(project, 'input', 'luca')
    stdout_ = os.path.join(project, 'output', 'stdout.txt')
    # snodas = os.path.join(project, 'input', 'carter_basin_snodas.csv')

    # csv = os.path.join(data, 'prms_params_carter.csv')

    # luca_params = os.path.join(luca_dir, 'calib1_round3_step2.par')
    # read_calibration(luca_dir)

    prms = MontanaPrmsModel(prms_build.control_file,
                            prms_build.parameter_file,
                            prms_build.data_file)

    param_dict = {rn: prms.parameters.get_values(rn) for rn in prms.parameters.record_names}
    prms.run_model(stdout_)

    stats_uncal = prms.get_statvar()
    fig_ = os.path.join(project, 'output', 'hydrograph_uncal.png')
    plot_stats(stats_uncal, fig_)

    # prms = MontanaPrmsModel(prms_build.control_file,
    #                         luca_params,
    #                         prms_build.data_file)

    # compare_parameters(prms, csv)

    # prms.run_model()
    # stats_cal = prms.get_statvar()
    # fig_ = os.path.join(project, 'output', 'hydrograph_cal.png')
    # plot_stats(stats_cal, fig_)


def compare_parameters(model, csv):
    df = pd.read_csv(csv)
    df = df.mean(axis=0)

    param_names = model.parameters.record_names

    comp_params = [x for x in df.index if x in param_names]

    for p in comp_params:
        mp = prms.parameters.get_values(p).mean()
        gf = df[p]
        print('p: {}, actual: {:.3f}, geofabric: {:.3f}'.format(p, mp, gf))

    pass


if __name__ == '__main__':

    wspace = os.path.dirname(os.path.abspath(__file__))
    conf = os.path.join(wspace, 'smith_parameters.toml')

    # build_data(conf, overwrite=True)
    run_model(wspace, conf)
# ========================= EOF ====================================================================
