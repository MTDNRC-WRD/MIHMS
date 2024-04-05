import os
import sys
import json
from calendar import monthrange

import numpy as np
import ee

sys.path.insert(0, os.path.abspath('..'))
sys.setrecursionlimit(5000)

RF_ASSET = 'projects/ee-dgketchum/assets/IrrMapper/IrrMapperComp'
UMRB_CLIP = 'users/dgketchum/boundaries/umrb_ylstn_clip'
CMBRB_CLIP = 'users/dgketchum/boundaries/CMB_RB_CLIP'
CORB_CLIP = 'users/dgketchum/boundaries/CO_RB'
WESTERN_11_STATES = 'users/dgketchum/boundaries/western_11_union'


def export_gridded_data(tables, bucket, years, description, features=None, min_years=0, debug=False):
    """
    Reduce Regions, i.e. zonal stats: takes a statistic from a raster within the bounds of a vector.
    Use this to get e.g. irrigated area within a county, HUC, or state. This can mask based on Crop Data Layer,
    and can mask data where the sum of irrigated years is less than min_years. This will output a .csv to
    GCS wudr bucket.
    :param features:
    :param bucket:
    :param tables: vector data over which to take raster statistics
    :param years: years over which to run the stats
    :param description: export name append str
    :param cdl_mask:
    :param min_years:
    :return:
    """
    is_authorized()
    fc = ee.FeatureCollection(tables)
    if features:
        fc = fc.filter(ee.Filter.inList('STAID', features))
    cmb_clip = ee.FeatureCollection(CMBRB_CLIP)
    umrb_clip = ee.FeatureCollection(UMRB_CLIP)
    corb_clip = ee.FeatureCollection(CORB_CLIP)

    eff_ppt_coll = ee.ImageCollection('users/dgketchum/expansion/ept')
    eff_ppt_coll = eff_ppt_coll.map(lambda x: x.rename('eff_ppt'))

    irr_coll = ee.ImageCollection(RF_ASSET)
    coll = irr_coll.filterDate('1987-01-01', '2021-12-31').select('classification')
    remap = coll.map(lambda img: img.lt(1))
    irr_min_yr_mask = remap.sum().gte(min_years)
    irr_mask = irr_min_yr_mask

    for yr in years:
        for month in range(1, 13):
            s = '{}-{}-01'.format(yr, str(month).rjust(2, '0'))
            end_day = monthrange(yr, month)[1]
            e = '{}-{}-{}'.format(yr, str(month).rjust(2, '0'), end_day)

            irr = irr_coll.filterDate('{}-01-01'.format(yr), '{}-12-31'.format(yr)).select('classification').mosaic()
            irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))

            annual_coll = ee.ImageCollection('users/dgketchum/ssebop/cmbrb').merge(
                ee.ImageCollection('users/hoylmanecohydro2/ssebop/cmbrb'))
            et_coll = annual_coll.filter(ee.Filter.date(s, e))
            et_cmb = et_coll.sum().multiply(0.00001).clip(cmb_clip.geometry())

            annual_coll = ee.ImageCollection('users/kelseyjencso/ssebop/corb').merge(
                ee.ImageCollection('users/dgketchum/ssebop/corb')).merge(
                ee.ImageCollection('users/dpendergraph/ssebop/corb'))
            et_coll = annual_coll.filter(ee.Filter.date(s, e))
            et_corb = et_coll.sum().multiply(0.00001).clip(corb_clip.geometry())

            annual_coll_ = ee.ImageCollection('projects/usgs-ssebop/et/umrb')
            et_coll = annual_coll_.filter(ee.Filter.date(s, e))
            et_umrb = et_coll.sum().multiply(0.00001).clip(umrb_clip.geometry())

            et_sum = ee.ImageCollection([et_cmb, et_corb, et_umrb]).mosaic()

            eff_ppt = eff_ppt_coll.filterDate(s, e).select('eff_ppt').mosaic()

            ppt, etr = extract_gridmet_monthly(yr, month)
            ietr = extract_corrected_etr(yr, month)

            area = ee.Image.pixelArea()

            irr_mask = irr_min_yr_mask.updateMask(irr.lt(1))
            et = et_sum.mask(irr_mask)
            eff_ppt = eff_ppt.mask(irr_mask).rename('eff_ppt')
            ietr = ietr.mask(irr_mask)
            irr_mask = irr_mask.reproject(crs='EPSG:5070', scale=30)
            irr = irr_mask.multiply(area).rename('irr')

            et = et.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('et')
            eff_ppt = eff_ppt.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('eff_ppt')
            ppt = ppt.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('ppt')
            etr = etr.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('etr')
            ietr = ietr.reproject(crs='EPSG:5070', scale=30).resample('bilinear').rename('ietr')

            cc = et.subtract(eff_ppt).rename('cc')

            et = et.multiply(area)
            eff_ppt = eff_ppt.multiply(area)
            cc = cc.multiply(area)
            ppt = ppt.multiply(area)
            etr = etr.multiply(area)
            ietr = ietr.multiply(area)

            if yr > 1986 and month in range(4, 11):
                bands = irr.addBands([et, cc, ppt, etr, eff_ppt, ietr])
                select_ = ['STAID', 'irr', 'et', 'cc', 'ppt', 'etr', 'eff_ppt', 'ietr']

            else:
                bands = ppt.addBands([etr])
                select_ = ['STAID', 'ppt', 'etr']

            if debug:
                pt = bands.sample(region=get_geomteries()[2],
                                  numPixels=1,
                                  scale=30)
                p = pt.first().getInfo()['properties']
                print('propeteries {}'.format(p))

            data = bands.reduceRegions(collection=fc,
                                       reducer=ee.Reducer.sum(),
                                       scale=30)

            out_desc = '{}_{}_{}'.format(description, yr, month)
            task = ee.batch.Export.table.toCloudStorage(
                data,
                description=out_desc,
                bucket=bucket,
                fileNamePrefix=out_desc,
                fileFormat='CSV',
                selectors=select_)

            task.start()
            print(out_desc)


def is_authorized():
    ee.Authenticate()
    ee.Initialize(project='ee-dgketchum')


def get_geomteries():
    bozeman = ee.Geometry.Polygon([[-111.19206055457778, 45.587493372544984],
                                   [-110.91946228797622, 45.587493372544984],
                                   [-110.91946228797622, 45.754947053477565],
                                   [-111.19206055457778, 45.754947053477565],
                                   [-111.19206055457778, 45.587493372544984]])

    navajo = ee.Geometry.Polygon([[-108.50192867920967, 36.38701227276218],
                                  [-107.92995297120186, 36.38701227276218],
                                  [-107.92995297120186, 36.78068624960868],
                                  [-108.50192867920967, 36.78068624960868],
                                  [-108.50192867920967, 36.38701227276218]])

    test_point = ee.Geometry.Point(-110.644980, 45.970375)

    western_us = ee.Geometry.Polygon([[-127.00073221292574, 30.011505140554807],
                                      [-100.63354471292574, 30.011505140554807],
                                      [-100.63354471292574, 49.908396143431744],
                                      [-127.00073221292574, 49.908396143431744],
                                      [-127.00073221292574, 30.011505140554807]])

    return bozeman, navajo, test_point, western_us


def extract_gridmet_monthly(year, month):
    m_str, m_str_next = str(month).rjust(2, '0'), str(month + 1).rjust(2, '0')
    if month == 12:
        dataset = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filterDate('{}-{}-01'.format(year, m_str),
                                                                        '{}-{}-31'.format(year, m_str))
    else:
        dataset = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET').filterDate('{}-{}-01'.format(year, m_str),
                                                                        '{}-{}-01'.format(year, m_str_next))
    pet = dataset.select('etr').sum().multiply(0.001).rename('gm_etr')
    ppt = dataset.select('pr').sum().multiply(0.001).rename('gm_ppt')
    return ppt, pet


def extract_corrected_etr(year, month):
    m_str = str(month).rjust(2, '0')
    end_day = monthrange(year, month)[1]
    ic = ee.ImageCollection('projects/openet/reference_et/gridmet/monthly')
    band = ic.filterDate('{}-{}-01'.format(year, m_str), '{}-{}-{}'.format(year, m_str, end_day)).select('etr').first()
    return band.multiply(0.001)


if __name__ == '__main__':
    start_yr, end_yr = 1995, 2021

    basins = 'users/dgketchum/gages/gage_basins'
    bucket = 'wudr'
    selected_g = ['06076690']

    gages_metadata = '/home/dgketchum/PycharmProjects/MIHMS/example/data/irrigated_gage_metadata_res_ibt.json'

    with open(gages_metadata, 'r') as fp:
        stations = json.load(fp)

    extract_years = np.arange(start_yr - 5, end_yr + 1)
    basin_ids = [station_id for station_id, _ in stations.items() if station_id in selected_g]
    export_gridded_data(basins, bucket, extract_years, features=basin_ids,
                        min_years=5, description='smith', debug=False)
# ========================= EOF ====================================================================
