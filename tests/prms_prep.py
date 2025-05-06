import tomli

from mihms.prep.prms.control import ControlBase
from mihms.config import prep

with open(prep.default_vars, 'rb') as vfile:
    varlists = tomli.load(vfile)

test = ControlBase('2020-01-01', '2024-12-31', 'UY_model.data', ['UY_params.param', 'UY_calib.param'], 'C:/example/Yellowstone')