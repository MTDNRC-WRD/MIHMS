import os
import json

import geopandas as gpd

from rasterstats import zonal_stats


def snodas_zonal_stats(in_shp, raster_dir, out_js, targets=None):
    """Find mean watershed SWU in meters depth, daily"""
    df = gpd.read_file(in_shp)

    if targets:
        df = df.loc[targets]

    geo, fids = list(df['geometry']), [0]

    l = sorted([os.path.join(raster_dir, x) for x in os.listdir(raster_dir) if x.endswith('.tif')])

    dct = {}

    for r in l:
        dts = os.path.basename(r).replace('.tif', '').split('_')[-1]
        dct[dts] = {}
        stats = zonal_stats(geo, r, stats=['mean'])
        for fid, s in zip(fids, stats):
            if s['mean']:
                dct[dts][fid] = float(s['mean']) / 1000.
            else:
                dct[dts][fid] = 0.0
        print(os.path.basename(r), dct[dts][0])

    with open(out_js, 'w') as fp:
        json.dump(dct, fp, indent=4)

    print('wrote', out_js)


if __name__ == '__main__':

    s_dir = '/data/hdd1/snodas/processed/swe'
    if not os.path.isdir(s_dir):
        s_dir = '/media/research/IrrigationGIS/climate/snodas/processed/swe'

    shp_ = '/home/dgketchum/PycharmProjects/MIHMS/example/data/smith_3000/domain/smith_basin_wgs.shp'
    out_js_ = '/home/dgketchum/PycharmProjects/MIHMS/example/data/smith_3000/input/snodas.json'
    snodas_zonal_stats(shp_, s_dir, out_js_)
# ========================= EOF ====================================================================
