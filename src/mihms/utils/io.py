import warnings
import inspect
import os
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def user_warning(msg, frame, wtype=UserWarning):
    """
    Method to standardize the warning output and avoid
    absolute file paths in warning messages - copied from pygsflow repository

    Parameters
    ----------
    msg : str
        error message
    frame : named tuple
        from inspect.getframeinfo
    wtype :
        warning type to be displayed defaults to UserWarning

    """
    module = os.path.split(frame.filename)[-1]
    warnings.warn_explicit(msg, wtype, module, frame.lineno)