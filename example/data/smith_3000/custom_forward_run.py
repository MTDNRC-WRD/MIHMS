"""
Modify existing model with PEST++ parameter proposal and run the model.
"""
import os

from models.models import MontanaPrmsModel
from prep.prms.xyz_builder import XyzDistBuild


def run_model():

    # skip instantiating the build class, just use the paths
    # prms_build = XyzDistBuild(config)

    control_file = '/home/dgketchum/PycharmProjects/MIHMS/example/data/smith_3000/control/smith_3000.control'
    parameter_file = '/home/dgketchum/PycharmProjects/MIHMS/example/data/smith_3000/input/smith_3000.params'
    data_file = '/home/dgketchum/PycharmProjects/MIHMS/example/data/smith_3000/input/smith_3000_xyz.data'

    prms = MontanaPrmsModel(control_file,
                            parameter_file,
                            data_file)
    prms.run_model()


if __name__ == '__main__':
    run_model()
# ========================= EOF ====================================================================
