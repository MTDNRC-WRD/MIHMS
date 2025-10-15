import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spotpy
import tqdm

from mihms.calibrate.utils import calc_cdf
from mihms.calibrate.database import MultiDimDb, MemDb

# TODO: need to check this and make sure it is multidimensional capable (ie can accept the
#   parameter as a single entry but then randomly sample across the full dimension space
def LHS(a, N):
    """

    Parameters
    ----------
    a: A numpy 1-D array of values (include only min and max for uniform distribution)

    N: number of equi-probable intervals to sample cdf, equivalent to number of
    model runs/parameter sets

    Returns - Stratified Random Sample of parameter values of size N for the input array
    -------

    """
    Is = np.linspace(0, 1, N+1)
    Ie = np.roll(Is, -1)
    Is = Is[:-1].copy()
    Ie = Ie[:-1].copy()
    srs = []
    for i in range(len(Is)):
        rs_p = np.random.uniform(Is[i], Ie[i], 1)
        rs_v = np.quantile(a, rs_p[0])
        srs.append(rs_v)
    return np.array(srs)


# TODO - This needs lots of improvements to generalize to iterative database writing (like for zarr)
# TODO - There also needs to be functionality for using an existing distribution's percentiles versus remaking the distribution based only on the bounds (traditional LHS)
def LHS_md(params: list[spotpy.parameter.Base], repetitions: int, dbase: MultiDimDb, use_distribution: bool = False):
    """

    Args:
        params:
        repetitions:
        dbase_name:
        dbase_format:

    """
    database = copy.deepcopy(dbase)
    param_dims = {}
    param_dim_names = {}
    for p in params:
        ndim = {p.name: p.dim}
        ndimname = {p.name: p.dim_name}
        param_dims.update(ndim)
        param_dim_names.update(ndimname)

    database.dims = param_dims
    database.dim_names = param_dim_names

    if use_distribution:
        Is = np.linspace(0, 1, repetitions + 1)
        Ie = np.roll(Is, -1)
        Is = Is[:-1].copy()
        Ie = Ie[:-1].copy()
        paramdict = {}
        for p in params:
            if p.rndfunctype == 'List':
                val_arr = p.values
                if len(val_arr.shape) != 2:
                    raise ValueError("For multidimensional LHS, a 2D paramter array is required for 'List' type.")
            else:
                vals = np.linspace(p.minbound, p.maxbound, repetitions)
                val_arr = np.repeat(vals[:, None], p.dim, axis=1)
            print(f"Sampling {repetitions} repetitions of the {p.name} parameter...")
            samples = []
            for i in tqdm.tqdm(range(len(Is)), desc=f'{p.name} samples', leave=False):
                rs_p = np.random.uniform(Is[i], Ie[i], p.dim)
                rs_v = np.nanquantile(val_arr, rs_p, axis=0)
                p_samps = rs_v[np.arange(p.dim), np.arange(p.dim)]
                samples.append(p_samps)

            paramdict.update({p.name: np.array(samples)})
    else:
        segment = 1 / float(repetitions)
        paramdict = {}
        for p in params:
            parmin = p.minbound
            parmax = p.maxbound

            if isinstance(parmin, float):
                parmin = np.repeat(parmin, p.dim)
            if isinstance(parmax, float):
                parmax = np.repeat(parmax, p.dim)

            matrix = np.empty((repetitions, p.dim))
            print(f"Sampling {repetitions} repetitions of the {p.name} parameter...")
            for i in tqdm.tqdm(range(repetitions), desc=f"{p.name} samples", leave=False):
                segmentMin = i * segment
                pointInSegment = segmentMin + (np.random.random() * segment)
                parset = pointInSegment * (parmax - parmin) + parmin
                matrix[i, :] = parset
            paramdict.update({p.name: matrix})

    # "Shuffle" or randomize the parameter sets (or combinations)
    for k, v in paramdict.items():
        np.random.shuffle(v)
        paramdict[k] = v

    database.save(param_dict=paramdict)

    if database.format == 'memory':
        if paramdict != database.parameter_samples:
            raise ValueError("The sampled parameter dictionary does not match the database parameter records...")
        else:
            paramdict = copy.deepcopy(database.parameter_samples)
        return paramdict, database
    else:
        return paramdict