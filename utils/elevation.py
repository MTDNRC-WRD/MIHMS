import requests
import urllib

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


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
