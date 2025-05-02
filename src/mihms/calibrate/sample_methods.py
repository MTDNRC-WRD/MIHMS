import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from util_funcs import calc_cdf

tst = np.array([0.02, 3.55])
tst = np.linspace(0.02, 3.55, 100)
tst = np.random.uniform(100)
cdf = calc_cdf(tst)
p = [0.01, 0.25, 0.5, 0.75, 0.99]
smp = np.quantile(tst, p)
plt.plot(cdf[0], cdf[1], 'r-', smp, p, 'ko')

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
