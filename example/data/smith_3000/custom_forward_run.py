"""
Modify existing model with PEST++ parameter proposal and run the model.
"""
import os

import numpy as np

from mihms.models import MontanaPrmsModel


def run_model():
    # skip instantiating the build class, just use the paths
    # prms_build = XyzDistBuild(config)

    control_file = '/home/dgketchum/PycharmProjects/MIHMS/example/data/smith_3000/control/smith_3000.control'
    parameter_file = '/home/dgketchum/PycharmProjects/MIHMS/example/data/smith_3000/input/smith_3000.params'
    data_file = '/home/dgketchum/PycharmProjects/MIHMS/example/data/smith_3000/input/smith_3000_xyz.data'

    prms = MontanaPrmsModel(control_file,
                            parameter_file,
                            data_file)

    # update model params with PEST++ parameter value proposal
    pest_d = '/home/dgketchum/PycharmProjects/MIHMS/example/data/smith_3000/pest'
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

    prms.parameters.write(parameter_file)
    prms.run_model()

    statvar = prms.get_statvar()
    pred = statvar[['basin_cms']].values
    np.savetxt(os.path.join(pest_d, 'obs_q.np'), pred)

    idx = statvar['runoff'] >= 0.0
    valid = statvar.loc[idx]
    pred, obs = valid['runoff'].mean(), valid['basin_cms'].mean()

    print('Measured Dates, Mean Pred: {:.1f}; Obs {:.1f}'.format(pred, obs))


if __name__ == '__main__':
    run_model()
# ========================= EOF ====================================================================
