import os
import json
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
from pandas import read_csv, date_range, to_datetime, isna, DataFrame

from utils.hydrograph import get_station_flows
from prep.met_data import download_ghcn


def write_basin_datafile(gages, data_file,
                         stations, ghcn_data, out_csv=None, start='1990-01-01', end='2021-12-31',
                         units='metric', nodata_value=-999, overwrite=False, return_modified=False):
    dt_index = date_range(start=start, end=end, tz='UTC')

    if not isinstance(gages, dict):
        with open(gages, 'r') as fp:
            gages = json.load(fp)

    for k, v in gages.items():

        s, e = v['start'], v['end']
        df = get_station_flows(s, e, k, overwrite=overwrite)
        df.columns = ['runoff']

        if to_datetime(start) > to_datetime(s):
            df = df.loc[start:]

        df = df.reindex(dt_index)

        if units == 'metric':
            df = df * 0.0283168

        v['data'] = df

        if not isinstance(stations, dict):
            with open(stations, 'r') as fp:
                stations = json.load(fp)

    invalid_stations = 0

    dropped = []

    for k, v in stations.items():

        _file = os.path.join(ghcn_data, '{}.csv'.format(k))

        if not os.path.exists(_file):
            df = download_ghcn(k, _file, start)
            if not isinstance(df, pd.DataFrame):
                continue
        else:
            df = read_csv(_file, parse_dates=True, infer_datetime_format=True, index_col=0)
            df.index = pd.DatetimeIndex(df.index, tz='UTC')

        s = v['start']

        if to_datetime(start) > to_datetime(s):
            df = df.loc[start:]
            if df.empty or (df.shape[0] / len(dt_index)) < 0.7:
                print(k, 'insuf records in date range')
                invalid_stations += 1
                dropped.append(k)
                continue

        else:
            print(k, '{} records'.format(df.shape[0]))

        df = df.reindex(dt_index)

        try:
            df = df[['TMAX', 'TMIN', 'PRCP']]

            df['tmax'] = df['TMAX'] / 10.
            df[df['tmax'] > 43.0] = np.nan
            df[df['tmax'] < -40.0] = np.nan

            df['tmin'] = df['TMIN'] / 10.
            df[df['tmin'] > 43.0] = np.nan
            df[df['tmin'] < -40.0] = np.nan

            df['precip'] = df['PRCP'] / 10.

            df = df[['tmax', 'tmin', 'precip']]

            if units != 'metric':
                df['tmax'] = (df['tmax'] * 9 / 5) + 32.
                df['tmin'] = (df['tmin'] * 9 / 5) + 32.
                df['precip'] = df['precip'] / 25.4

            if df.empty or df.shape[0] < 1000:
                print(k, 'insuf records in date range')
                invalid_stations += 1

        except KeyError as e:
            print(k, 'incomplete', e)
            invalid_stations += 1
            dropped.append(k)
            continue

        print(k, df.shape[0])
        df.dropna(axis=0, how='any', inplace=True)
        df['precip'] = df[['precip']].fillna(0.0)
        stations[k]['data'] = df

    input_dct = {k: v for k, v in stations.items() if 'data' in v.keys()}

    [input_dct.update({k: v}) for k, v in gages.items()]

    dt_now = datetime.now().strftime('%Y-%m-%d %H:%M')

    with open(data_file, 'w') as f:

        df = DataFrame(index=dt_index)

        time_div = ['Year', 'Month', 'day', 'hr', 'min', 'sec']
        df['Year'] = [i.year for i in df.index]
        df['Month'] = [i.month for i in df.index]
        df['day'] = [i.day for i in df.index]
        for t_ in time_div[3:]:
            df[t_] = [0 for _ in df.index]

        [f.write('{}\n'.format(item)) for item in ['PRMS Datafile: {}\n'.format(dt_now),
                                                   '// ',
                                                   '    '.join(
                                                       ['// ID  ',
                                                        '    Site Name{}'.format(' '.join([' ' for _ in range(14)])),
                                                        'Type',
                                                        'Lat',
                                                        'Lon',
                                                        'Elev (m)',
                                                        'Units'])]]

        counts = OrderedDict([('runoff', 0), ('tmax', 0), ('tmin', 0), ('precip', 0)])
        vars_ = ['runoff', 'tmax', 'tmin', 'precip']

        if units == 'metric':
            units_ = ['cms', 'C', 'C', 'mm']
        else:
            units_ = ['cfs', 'F', 'F', 'in']

        for var, unit in zip(vars_, units_):
            for k, v in input_dct.items():
                try:
                    d = v['data']
                except KeyError:
                    continue

                if var in d.columns:
                    line_ = ' '.join(['// ', k.rjust(11, '0'),
                                      v['name'].ljust(40, ' ')[:40],
                                      var,
                                      '{:.3f}'.format(v['lat']),
                                      '{:.3f}'.format(v['lon']),
                                      '{:.1f}'.format(v['elev']),
                                      unit]) + '\n'
                    f.write(line_)
                    df['{}_{}'.format(k, var)] = d[var]
                    counts[var] += 1

        f.write('// \n')
        for k, v in counts.items():
            f.write('{} {}\n'.format(k, v))
        f.write('######################## \n')

        df[isna(df)] = nodata_value

        df = df.round(2)
        df = df.astype(str)
        # df = df.replace('-999.00', '-999', regex=True)
        df.to_csv(f, sep=' ', header=False, index=False, lineterminator='\n')

        if out_csv:
            #  save dataframe to normal csv for use elsewhere
            df = df[[c for c in df.columns if c not in time_div]]
            df['date'] = df.index
            df[df.values == nodata_value] = np.nan
            df.to_csv(out_csv, sep=' ', float_format='%.2f')

    if return_modified:
        input_dct = {k: v for k, v in input_dct.items() if k not in dropped}
        return input_dct


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
