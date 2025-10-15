import pandas as pd
import numpy as np

obj_func_direction = {
    'mse': 'minimize',
    'rmse': 'minimize',
    'nrmse': 'minimize',
    'nse': 'maximize',
    'pbias': 'minimize'
}

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


def mse_md(observation: np.ndarray, simulation: np.ndarray, return_dict: bool = False, axis=0):
    if observation.shape[axis] == simulation.shape[axis]:
        mse = np.nanmean((observation - simulation) ** 2, axis=axis)
        if return_dict:
            return {'mse': mse}
        else:
            return mse
    else:
        raise ValueError("evaluation and simulation data do not have the same length.")


def rmse_md(observation: np.ndarray, simulation: np.ndarray, return_dict: bool = False, axis=0):
    if observation.shape[axis] == simulation.shape[axis]:
        mse = mse_md(observation, simulation, axis=axis)
        rmse = np.sqrt(mse)
        if return_dict:
            return {'rmse': rmse}
        else:
            return rmse
    else:
        raise ValueError("evaluation and simulation data do not have the same length.")


def nrmse_md(observation: np.ndarray, simulation: np.ndarray, return_dict: bool = False, axis=0):
    if observation.shape[axis] == simulation.shape[axis]:
        nrmse = rmse_md(observation, simulation, axis=axis) / np.nanmean(observation, axis=axis)
        if return_dict:
            return {'nrmse': nrmse}
        else:
            return nrmse
    else:
        raise ValueError("evaluation and simulation data do not have the same length.")


def nse_md(observation: np.ndarray, simulation: np.ndarray, return_dict: bool = False, axis=0):
    if observation.shape[axis] == simulation.shape[axis]:
        mean_observed = np.nanmean(observation, axis=axis)
        # compute numerator and denominator
        numerator = np.nansum((observation - simulation) ** 2, axis=axis)
        denominator = np.nansum((observation - mean_observed) ** 2, axis=axis)
        # compute coefficient
        nse = 1 - (numerator / denominator)
        if return_dict:
            return {'nse': nse}
        else:
            return nse
    else:
        raise ValueError("evaluation and simulation data do not have the same length.")


def pbias_md(observation: np.ndarray, simulation: np.ndarray, return_dict: bool = False, axis=0):
    if observation.shape[axis] == simulation.shape[axis]:
        pbias = np.nansum(simulation - observation, axis=axis) / np.nansum(observation, axis=axis)
        if return_dict:
            return {'pbias': pbias}
        else:
            return pbias
    else:
        raise ValueError("evaluation and simulation data do not have the same length.")