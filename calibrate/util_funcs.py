import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def calc_cdf(a):
    """

    Parameters
    ----------
    a: A numpy 1-D array of values

    Returns - cdf; 2-D numpy array where ndarray[0] = sorted input (a) values
    and ndarray[1] = cumulative probability for each value
    -------

    """
    srt = np.sort(a)
    N = len(a)
    rnk = np.arange(1, N+1)
    P = rnk / (float(N)+1.0)
    return np.array([srt, P])

def sample_cdf(a, p):
    """

    Parameters
    ----------
    a: A numpy 1-D array of values

    p: a single value or array-like input of cumulative probabilities (quantiles) where
    corresponding parameter values are desired

    Returns - values for input probabilities
    -------

    """
    smple = np.quantile(a, p)
    return smple

