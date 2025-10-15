import tomli
from pathlib import Path

from mihms.prep.prms.control import ControlBase
from mihms.utils.io import pathlike_as_string
from mihms.config import prep

with open(prep.default_vars, 'rb') as vfile:
    varlists = tomli.load(vfile)

test = ControlBase('2020-01-01', '2024-12-31', 'UY_model.data', ['UY_params.param', 'UY_calib.param'], 'C:/example/Yellowstone')

test_pth = r'C:\Users\CNB968\This\isnot\adirectory'
testppth = Path(test_pth)
test_pth2 = "C:/Users/Another/directory"
testppth2 = Path(test_pth2)

pathlike_as_string([test_pth, testppth, test_pth2, testppth2])
