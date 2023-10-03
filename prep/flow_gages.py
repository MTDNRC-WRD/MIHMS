import json
from time import sleep

import fiona
from shapely.geometry import shape
from utils.elevation import elevation_from_coordinate


def get_gage_stations(basin_shp, gages_shp, out_json):
    """Select USGS gages within the bounds of basin_shp write to .json"""

    _crs = [fiona.open(shp).meta['crs'] for shp in [basin_shp, gages_shp]]
    assert all(x == _crs[0] for x in _crs)
    with fiona.open(basin_shp, 'r') as basn:
        basin_geo = shape([f['geometry'] for f in basn][0])

    stations = {}

    with fiona.open(gages_shp, 'r') as src:
        for f in src:
            geo = shape(f['geometry'])
            if not geo.intersects(basin_geo):
                continue
            sta = f['properties']['STAID']
            s, e = f['properties']['start'], f['properties']['end']
            elev = elevation_from_coordinate(f['properties']['LAT'], f['properties']['LON'])
            print(sta, elev)
            # sleep(2)

            stations[sta] = {'start': s, 'end': e,
                             'lat': f['properties']['LAT'],
                             'lon': f['properties']['LON'],
                             'elev': elev,
                             'type': 'usgs',
                             'name': f['properties']['STANAME'],
                             'proj_coords': (geo.y, geo.x),
                             'snotel': False}

    with open(out_json, 'w') as dst:
        dst.write(json.dumps(stations))


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
