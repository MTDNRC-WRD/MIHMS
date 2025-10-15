"""Script to perform FAST Sensitivity analysis for streamflow by subbasin in PRMS.

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
import geopandas as gpd


sys.path.append('C:/Users/CNB968/OneDrive - MT/GitHub/MIHMS/src')

import mihms.prep.prms as pprms
from mihms.models.models import PrmsModel
from mihms.output import PrmsOutput

m_wd = Path('D:/Modeling/GSFLOW/PRMS_Projects/Upper Yellowstone/streamflow_calibration')
openet_pth = Path('D:/Modeling/GSFLOW/PRMS_Projects/Upper Yellowstone/calibration_data/OpenET_by_HRU_2020_2024.nc')

prmscont = pprms.CreateControlFile(Path('C:/Users/CNB968/OneDrive - MT/Modeling/GSFLOW/PRMS/prms_6.0.0/bin/prms.exe'),
                               '6.0.0',
                               '2018-01-01',
                               '2024-12-31',
                               m_wd / 'UY_1979_2024.data',
                               [m_wd / 'UY_params.param'],
                               m_wd,
                               file_name='UY_streamflow.control')
prmscont.select_output_variables(['sub_cfs', 'sub_inq'])
prmscont.save_controlfile()


class SpotSetup(object):

    def __init__(self, subbasin, obj_func=None):
        self.model = PrmsModel(m_wd / 'UY_streamflow.control')
        self.params = [
            Uniform(name='adjmix_rain', low=0.0, high=3.0),
            Uniform(name='tmax_allsnow', low=-10.0, high=40.0),
            Uniform(name='tmax_allrain_offset', low=0.0, high=50.0),
            Uniform(name='imperv_stor_max', low=0.0, high=0.5),
            Uniform(name='potet_sublim', low=0.00001, high=0.75),
            Uniform(name='sat_threshold', low=0.0, high=999.0),
            Uniform(name='smidx_coef', low=0.0, high=1.0),
            Uniform(name='smidx_exp', low=0.0, high=5.0),
            Uniform(name='fastcoef_lin', low=0.0, high=1.5),
            Uniform(name='fastcoef_sq', low=0.0, high=1.0),
            Uniform(name='slowcoef_lin', low=0.0, high=1.0),
            Uniform(name='slowcoef_sq', low=0.0, high=1.0),
            Uniform(name='pref_flow_den', low=0.0, high=5.0),
            Uniform(name='pref_flow_infil_frac', low=-1.0, high=1.0),
            Uniform(name='soil_rechr_max_frac', low=0.00001, high=1.0),
            Uniform(name='soil2gw_max', low=0.0, high=5.0),
            Uniform(name='ssr2gw_exp', low=0.0, high=3.0),
            Uniform(name='ssr2gw_rate', low=0.00001, high=1.0),
            Uniform(name='gwflow_coef', low=0.0, high=5.0),
            Uniform(name='gwsink_coef', low=0.0, high=1.0),
            Uniform(name='gwstor_min', low=0.0, high=1.0),
            Uniform(name='freeh2o_cap', low=0.01, high=0.2),
            Uniform(name='settle_const', low=0.01, high=0.5),
            Uniform(name='snarea_thresh', low=0.0, high=200.0)
        ]
        self.param_archive = self.model.parameters.parameters.copy()
        obsdata = pd.read_csv('D:/Modeling/GSFLOW/PRMS_Projects/Upper Yellowstone/calibration_data/gage_observation_data.csv')
        mod_grid = gpd.read_file('D:/ArcGIS_Projects/Yellowstone/Upper Yellowstone/prms/UY_prms_grid.shp')
        obs_pnts = gpd.read_file('D:/ArcGIS_Projects/Yellowstone/Upper Yellowstone/Vector/prms_subbasin_pourpnts.shp')

        self.obs_data = pprms.create_obs_dataset(obsdata, obs_pnts.iloc[:-1,:], 'station_id', mod_grid)
        self.subbasin = subbasin
        self.obj_func = obj_func

    def parameters(self):
        return generate(self.params)

    def simulation(self, P):
        for i, prm in enumerate(P):
            self.model.parameters.change_parameter(self.params[i].name, prm)

        self.model.parameters.write_paramfile()
        self.model.run_model(silent=True, report=False)
        res = PrmsOutput(self.model.control)
        subout = res.get_subbasin_output()
        subout = subout.sel(time=slice('2020-01-01', '2024-12-31'))
        vals = subout.sub_inq.sel(subbasin=self.subbasin).values

        return vals.tolist()

    def evaluation(self):
        loc = self.obs_data.location[self.obs_data.subbasin == self.subbasin]
        vals = self.obs_data.observation_data.sel(location=loc.values).values.ravel()

        return vals.tolist()

    def objectivefunction(self, simulation, evaluation):
        if self.obj_func is None:
            obfun = of.rmse(evaluation, simulation)
        else:
            obfun = self.obj_func(evaluation, simulation)

        return obfun

if __name__ == '__main__':
    basin = 1
    spotsetup = SpotSetup(basin, of.nashsutcliffe)
    sampler = spotalg.efast(spotsetup, dbname=(m_wd / f'FAST/PRMS_FAST_streamflow_subbasin_{basin}').as_posix(), dbformat='csv')
    sampler.sample(5000)