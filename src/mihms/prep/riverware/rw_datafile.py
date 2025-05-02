# This script creates RiverWare compatible data files or parses RiverWare output files.

import sqlite3
from pathlib import Path
import pandas as pd


def extract_series_from_db(dbase_pth, db_map_dict):
    """
    :param dbase_pth: pathname to model sqlite database
    :param db_map_dict: dictionary database map -- decode riverware slots to database query
    :return: pandas dataframe of slot data extracted from sqlite database
    """

    # currently only accepts values for one where statement in the SQL query
    col, val = [v for v in db_map_dict['where'].items()][0]

    with sqlite3.connect(dbase_pth) as con:
        q = pd.read_sql("""
                    SELECT * FROM {0}
                    WHERE {1} = '{2}';
                    """.format(db_map_dict['table'], col, val),
                        con=con,
                        parse_dates=['datetime'],
                        index_col=['datetime'])

    return {db_map_dict['slot']: q}


def format_input_file(series_dict, dmi_data_dir):
    """
    :param series_dict: dictionary of pandas series where RiverWare slot name is the dict key
    :param dmi_data_dir: str -- path to data directory constructed via config file
    :return: none -- writes output directly to RiverWare compatible text file
    """

    flname = list(series_dict.keys())[0]
    vals = series_dict[flname]

    with open(dmi_data_dir / flname, 'w') as f:
        f.write("start_date: {0}\n".format(vals.index[0]))
        f.write("end_date: {0}\n".format(vals.index[-1]))
        f.write("timestep: 1 DAY\n")
        for l in vals.tolist():
            f.write("{0}\n".format(l))
