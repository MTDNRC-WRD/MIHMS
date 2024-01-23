import os

import pandas as pd

from utils.thredds import GridMet

PARAMS = {'precip': 'pr', 'tmin': 'tmmn', 'tmax': 'tmmx'}


def gridmet_infill(df, *variables, lat, lon, units):

    df = df.tz_convert(None)
    for v in list(variables):
        p = PARAMS[v]
        g = GridMet(p, lat=lat, lon=lon, start=df.index[0], end=df.index[-1],)
        s = g.get_point_timeseries()
        if p.startswith('t'):
            if units == 'metric':
                s[p] -= 273.15
            else:
                s[p] = ((s[p] - 273.15) * 9/5) + 32.

        else:
            if units == 'standard':
                s[p] *= 0.03937008

        df[v].fillna(s[p], inplace=True)

    df.index = pd.DatetimeIndex(df.index, tz='UTC')
    return df


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
