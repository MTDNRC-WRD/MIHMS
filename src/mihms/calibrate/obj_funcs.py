import pandas as pd
import numpy as np


def NSE(x1, x2):
    """

    Parameters
    ----------
    x1: observed or measured dataset, ordered (time-series) list or array-like
    x2: modeled/comparison dataset, ordered (time-series) list or array-like

    Returns - Nash-Sutcliffe Efficiency calculated between a measured and modeled data series
    -------

    """
    nse = 1 - ((np.sum((x1 - x2)**2))/np.sum((x1 - np.mean(x1))**2))
    return nse

def RMSE(x1, x2):
    """

    Parameters
    ----------
    x1: observed or measured dataset, ordered (time-series) list or array-like
    x2: modeled/comparison dataset, ordered (time-series) list or array-like

    Returns - Root Mean Squared Error between observed and modeled data series
    -------

    """
    rmse = np.sqrt((np.sum((x1 - x2)**2)/len(x1)))
    return rmse

def iqr_NRMSE(x1, x2):
    """

    Parameters
    ----------
    x1: observed or measured dataset, ordered (time-series) list or array-like
    x2: modeled/comparison dataset, ordered (time-series) list or array-like

    Returns - Normalized Root Mean Squared Error (NRMSE: normalized by the IQR)
    between observed and modeled data series
    -------

    """
    rmse = np.sqrt((np.sum((x1 - x2) ** 2) / len(x1)))
    nrmse = rmse / (np.quantile(x1, 0.75) - np.quantile(x1, 0.25))
    return nrmse

