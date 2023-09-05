import os

import matplotlib
import pandas as pd

from models.models import MontanaPrmsModel
from prep.prms.model_builders import CbhruPrmsBuild
from prep.datafile import write_basin_datafile
from prep.flow_gages import get_gage_stations
from prep.met_data import get_ghcn_stations
from utils.plotting import plot_stats


def build_data(config, overwrite=False):
    """
    Build input data for MIHMs, starting with PRMS inputs.
    :param overwrite:
    :param config:
    :return:
    """

    prms_build = CbhruPrmsBuild(config)
    cfg = prms_build.cfg

    if not os.path.exists(cfg.prms_data_gages) or overwrite:
        get_gage_stations(cfg.study_area_path, cfg.usgs_gages, cfg.prms_data_gages)

    get_ghcn_stations(cfg.study_area_path, ghcn_shp=cfg.ghcn_stations, snotel_shp=cfg.snotel_stations,
                      out_json=cfg.prms_data_stations)

    write_basin_datafile(cfg.prms_data_gages, station_json=cfg.prms_data_stations,
                         data_file=prms_build.data_file, ghcn_data=cfg.prms_data_ghcn)


def run_model(config):
    root = '/media/research/IrrigationGIS/Montana/upper_yellowstone/gsflow_prep'
    matplotlib.use('TkAgg')

    project = os.path.join(root, 'uyws_carter_5000')
    luca_dir = os.path.join(project, 'input', 'luca')
    stdout_ = os.path.join(project, 'output', 'stdout.txt')
    snodas = os.path.join(project, 'input', 'carter_basin_snodas.csv')

    csv = '/media/research/IrrigationGIS/Montana/geospatial_fabric/prms_params_carter.csv'

    prms_build = CbhruPrmsBuild(config)
    prms_build.build_model()

    luca_params = os.path.join(luca_dir, 'calib1_round3_step2.par')
    # read_calibration(luca_dir)

    prms = MontanaPrmsModel(prms_build.control_file,
                            prms_build.parameter_file,
                            prms_build.data_file)

    prms.run_model(stdout_)
    stats_uncal = prms.get_statvar()
    fig_ = os.path.join(project, 'output', 'hydrograph_uncal.png')
    plot_stats(stats_uncal, fig_)

    prms = MontanaPrmsModel(prms_build.control_file,
                            luca_params,
                            prms_build.data_file)

    compare_parameters(prms, csv)

    prms.run_model()
    stats_cal = prms.get_statvar(snow=snodas)
    fig_ = os.path.join(project, 'output', 'hydrograph_cal.png')
    plot_stats(stats_cal, fig_)


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
    conf = '../models/uyws_parameters.ini'
    # build_data(conf, overwrite=False)
    run_model(conf)
# ========================= EOF ====================================================================
