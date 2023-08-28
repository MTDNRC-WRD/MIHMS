import os

from models.prms import CbhruPrmsBuild

from prep.datafile import write_basin_datafile
from prep.flow_gages import get_gage_stations
from prep.met_data import get_ghcn_stations


def build_data(config, overwrite=False):
    """
    Build input data for MIHMs, starting with PRMS inputs.
    :param overwrite:
    :param config:
    :return:
    """

    prms_build = CbhruPrmsBuild(conf)
    cfg = prms_build.cfg

    if not os.path.exists(cfg.prms_data_gages) or overwrite:
        get_gage_stations(cfg.study_area_path, cfg.usgs_gages, cfg.prms_data_gages)

    get_ghcn_stations(cfg.study_area_path, ghcn_shp=cfg.ghcn_stations, snotel_shp=cfg.snotel_stations,
                      out_json=cfg.prms_data_stations)

    write_basin_datafile(cfg.prms_data_gages, station_json=cfg.prms_data_stations,
                         data_file=prms_build.data_file, ghcn_data=cfg.prms_data_ghcn)


def run_model(config):
    pass


if __name__ == '__main__':
    conf = '../models/uyws_parameters.ini'
    build_data(conf, overwrite=False)
# ========================= EOF ====================================================================
