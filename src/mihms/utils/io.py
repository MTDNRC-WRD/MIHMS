import warnings
import inspect
import os
import json
from typing import Union
from pathlib import Path

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

def pathlike_as_string(pathstring: Union[str, Path, list, np.ndarray]) -> Union[str, list]:
    """
    Standardizes a pathlike string to a posix string representation.
    Args:
        pathstring: str | pathlib.Path | list
            A string path or pathlib.Path object.

    Returns: str
        A posix-like string or list of strings of the path name(s).

    """
    if isinstance(pathstring, str):
        pth = Path(pathstring)
        return pth.as_posix()
    elif isinstance(pathstring, Path):
        return pathstring.as_posix()
    elif isinstance(pathstring, (list, np.ndarray)):
        pths = []
        for i in pathstring:
            if isinstance(i, str):
                pth = Path(i)
            elif isinstance(i, Path):
                pth = i
            else:
                raise ValueError("One of the path strings is neither string type nor pathlib.Path type.")
            pths.append(pth.as_posix())

        return pths
    else:
        raise ValueError("The path is not recognized as a string, pathlib.Path object, or list of paths.")


def remove_all_files_in_directory(directory_path: Union[str, Path], recursive: bool = False):
    """
    Removes all files within the specified directory.
    Subdirectories and their contents are not removed.

    Args:
        directory_path:
            A string path or pathlib.Path object.
        recursive:
            Boolean to determine if files in subdirectories are also removed recursively (True) or just the top level
            directory (False)

    Returns:
        None
    """
    target_dir = Path(directory_path)

    if not target_dir.is_dir():
        print(f"Error: '{directory_path}' is not a valid directory.")
        return

    if recursive:
        for item in target_dir.rglob('*'):
            if item.is_file():
                try:
                    item.unlink()  # Deletes the file
                except OSError as e:
                    print(f"Error deleting file {item}: {e}")
    else:
        for item in target_dir.iterdir():
            try:
                item.unlink()  # Deletes the file
            except OSError as e:
                print(f"Error deleting file {item}: {e}")