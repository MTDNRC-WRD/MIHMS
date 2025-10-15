"""Script to perform FAST Sensitivity analysis for JH ET module of PRMS.

author: Todd Blythe
MT DNRC - Water Sciences Bureau

"""
from pathlib import Path
import sys

from spotpy.parameter import Uniform, generate
import spotpy.objectivefunctions as of
import spotpy.algorithms as spotalg
import xarray as xr
import numpy as np
import pandas as pd


sys.path.append('C:/Users/CNB968/OneDrive - MT/GitHub/MIHMS/src')

import mihms.prep.prms as pprms
from mihms.models.models import PrmsModel
from mihms.output import PrmsOutput

m_wd = Path('D:/Modeling/GSFLOW/PRMS_Projects/Upper Yellowstone/et_calibration')
openet_pth = Path('D:/Modeling/GSFLOW/PRMS_Projects/Upper Yellowstone/calibration_data/OpenET_by_HRU_2020_2024.nc')

prmscont = pprms.CreateControlFile(Path('C:/Users/CNB968/OneDrive - MT/Modeling/GSFLOW/PRMS/prms_6.0.0/bin/prms.exe'),
                               '6.0.0',
                               '2020-01-01',
                               '2024-12-31',
                               m_wd / 'UY_1979_2024.data',
                               [m_wd / 'UY_params.param'],
                               m_wd,
                               file_name='UY_et.control')
prmscont.add_statvars('hru_actet', 25501)
prmscont.save_controlfile()


class SpotSetup(object):

    def __init__(self, obj_func=None):
        self.model = PrmsModel(m_wd / 'UY_et.control')
        self.params = [
            Uniform(name='jh_coef', low=-0.5, high=1.5),
            Uniform(name='potet_sublim', low=0.1, high=0.75),
            Uniform(name='imperv_stor_max', low=0.0, high=0.5),
            Uniform(name='transp_tmax', low=0.0, high=1000.0),
            Uniform(name='soil_rechr_max_frac', low=0.00001, high=1.0),
            Uniform(name='sat_threshold', low=0.0, high=999.0),
        ]
        self.param_archive = self.model.parameters.parameters.copy()
        openet = xr.open_dataset(openet_pth)
        openet_conv = openet.actual_et.sel(hru=25501) / 25.4
        hru_obs = openet_conv.values.tolist()
        self.obs_data = hru_obs
        self.obj_func = obj_func

    def parameters(self):
        return generate(self.params)

    def simulation(self, P):
        for i, prm in enumerate(P):
            self.model.parameters.change_parameter(self.params[i].name, prm)

        self.model.parameters.write_paramfile()
        self.model.run_model(silent=True, report=False)
        res = PrmsOutput(self.model.control)
        res_hru = res.selected_variables['hru_actet_25501']
        res_hru_mon = res_hru.resample('MS').sum()

        return res_hru_mon.to_list()

    def evaluation(self):
        return self.obs_data

    def objectivefunction(self, simulation, evaluation):
        if self.obj_func is None:
            obfun = of.rmse(evaluation, simulation)
        else:
            obfun = self.obj_func(evaluation, simulation)

        return obfun

if __name__ == '__main__':
    spotsetup = SpotSetup(of.rmse)
    sampler = spotalg.efast(spotsetup, dbname=(m_wd / 'FAST/PRMS_FAST_et').as_posix(), dbformat='csv')
    sampler.sample(3000)