"""
Modify existing model with PEST++ parameter proposal and run the model.
"""
import os
import argparse

import numpy as np

from models.models import MontanaPrmsModel


def run_model(proj_dir, control, params, data, output_dir):

    prms = MontanaPrmsModel(control,
                            params,
                            data)

    # update model params with PEST++ parameter value proposal
    param_store = os.path.join(proj_dir, 'mult')
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

    prms.parameters.write(params)
    prms.run_model()

    args = {'output': {'pred_q': os.path.join(output_dir, 'pred_q.np'),
                       'basin_pweqv': os.path.join(output_dir, 'pred_swe.np')}}

    prms.get_statvar(snow_obs=None, return_df=False, **args)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--project_dir', required=True, help='Path to project directory')
    parser.add_argument('--control', required=True, help='Relative path to control file from project directory')
    parser.add_argument('--params', required=True, help='Relative path to params file from project directory')
    parser.add_argument('--data', required=True, help='Relative path to data file from project directory')
    parser.add_argument('--out_dir', required=True, help='Relative path to output directory from project directory')

    args = parser.parse_args()

    control = os.path.join(args.project_dir, args.control)
    params = os.path.join(args.project_dir, args.params)
    data = os.path.join(args.project_dir, args.data)
    out_dir = os.path.join(args.project_dir, args.out_dir)

    run_model(proj_dir=args.project_dir,
              control=control,
              params=params,
              data=data,
              output_dir=out_dir)


if __name__ == '__main__':
    main()
# ========================= EOF ====================================================================
