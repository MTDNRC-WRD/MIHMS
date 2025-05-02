import os

import urllib
import requests
import numpy as np
import geopandas as gpd

url = r'https://epqs.nationalmap.gov/v1/json?'


def elevation_from_coordinate(lat, lon):
    params = {
        'output': 'json',
        'x': lon,
        'y': lat,
        'units': 'Meters'
    }

    result = requests.get((url + urllib.parse.urlencode(params)))
    elev = float(result.json()['value'])
    return elev


def append_elevations(shp, out_shp):
    gdf = gpd.read_file(shp)
    out_gdf = gdf.copy()
    out_gdf['ELEV'] = [-9999 for x in range(gdf.shape[0])]
    for i, r in gdf.iterrows():
        try:
            elev = elevation_from_coordinate(r['LAT'], r['LON'])
            elev = np.round(elev, decimals=2)
            out_gdf.loc[i, 'ELEV'] = elev
        except Exception as e:
            print(r['STAID'], e)
        print(elev, r['STANAME'])

    out_gdf.to_file(out_shp)


if __name__ == '__main__':
    d = '/home/dgketchum/PycharmProjects/MIHMS/example/data/gages'
    i = 'usgs_gages_30yr.shp'
    o = 'usgs_gages_wELEV_30yr.shp'
    append_elevations(os.path.join(d, i), os.path.join(d, o))
# ========================= EOF ====================================================================
