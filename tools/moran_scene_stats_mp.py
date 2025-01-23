import argparse
import json
import logging
import multiprocessing
import os
import pprint
import random

import ee
import pandas as pd

import openet.core.utils as utils

"""
Upload the files to the bucket
    gsutil -m cp -r -n "./stats_moran/*" gs://openet_temp/moran_scene_stats/

Download the files from the bucket
    gsutil -m cp -r -n "gs://openet_temp/moran_scene_stats/*" ./stats_moran/

Download the stats files from the bucket
    gsutil -m cp -r "gs://openet_temp/scene_stats/*" ./stats/
    gsutil -m cp -r "gs://openet_temp/scene_stats/2024/*" ./stats/2024/
    
"""

ee.Initialize(
    project='ee-cmorton',
    opt_url='https://earthengine-highvolume.googleapis.com',
)


def main(years):
    #count_threshold_pct_min = 0
    #count_threshold_pct_min = 50
    count_threshold_pct_min = 95
    count_threshold_pct_max = 101

    years = sorted([
        year for year_str in years
        for year in utils.str_ranges_2_list(year_str)
    ])

    start_month = 1
    end_month = 12
    overwrite_flag = False

    stats_csv_ws = os.path.join(os.getcwd(), 'stats')
    output_ws = os.path.join(os.getcwd(), 'stats_moran')

    
    # # Use the OpenET ssebop collection for building the WRS2 list for now
    # wrs2_list = sorted(
    #     # ee.ImageCollection('projects/openet/assets/ssebop/conus/gridmet/landsat/c02')
    #     # ee.ImageCollection('projects/openet/assets/intercomparison/ssebop/landsat/c02/v0p2p6')
    #     ee.ImageCollection('projects/usgs-gee-nhm-ssebop/assets/ssebop/landsat/c02')
    #     .filterDate('2020-01-01', '2024-01-01')
    #     .aggregate_histogram('wrs2_tile').keys().getInfo(),
    #     reverse=True
    # )
    # wrs2_list = wrs2_list + ['p018r028']

    # Process all WRS2 tiles in the study area
    wrs2_list = sorted(
        ee.FeatureCollection('projects/openet/assets/features/wrs2/custom')
        .filterBounds(ee.Geometry.BBox(-124, 26, -68, 50))
        .filter(ee.Filter.inList('wrs2_tile', ['p10r030']).Not())
        .aggregate_histogram('wrs2_tile').keys().getInfo(),
        reverse=True
    )
    print(len(wrs2_list))
    
    # print('WRS2 tile count: {}'.format(len(wrs2_list)))
    
    print('\nReading skip lists')
    scene_skip_url = '../../scene-skip-list/v2p1.csv'
    # scene_skip_url = 'https://raw.githubusercontent.com/cgmorton/scene-skip-list/main/v2p1.csv'
    scene_skip_df = pd.read_csv(scene_skip_url)
    scene_skip_list = list(scene_skip_df['SCENE_ID'].values)
    print(f'  Skip list images: {len(scene_skip_list)}')
    
    scene_cloudscore_url = '../../scene-skip-list/v2p1_cloudscore.csv'
    # scene_cloudscore_url = 'https://raw.githubusercontent.com/cgmorton/scene-skip-list/main/v2p1_cloudscore.csv'
    scene_cloudscore_list = list(pd.read_csv(scene_cloudscore_url)['SCENE_ID'].values)
    print(f'  Skip cloudscore images: {len(scene_cloudscore_list)}')

    
    # Get the list of existing files
    print('\nReading Moran jsons')
    files = []
    for year in years:
        year_ws = os.path.join(output_ws, f'{year}')
        for month in range(start_month, end_month+1):
            month_ws = os.path.join(output_ws, f'{year}', f'{month:02d}')
            if not os.path.isdir(month_ws):
                continue
            for item in os.listdir(month_ws):
                if item.endswith('json'):
                    files.append(item)
    files = set(files)
    print('  JSON count: {}'.format(len(files)))
    
    
    print('\nBuilding scene list from stats files')
    stats_df_list = []
    for year in years:
        for wrs2_tile in wrs2_list:
            wrs2_stats_path = os.path.join(stats_csv_ws, f'{year}', f'{wrs2_tile}_{year}.csv')
            if not os.path.isfile(wrs2_stats_path):
                continue
            try:
                wrs2_stats_df = pd.read_csv(wrs2_stats_path)
            except Exception as e:
                print(f'  {wrs2_tile}_{year} - Error reading CSV, skipping')
                continue
            if wrs2_stats_df.empty:
                continue
            try:
                wrs2_stats_df.drop(columns=['system:index', '.geo'], inplace=True)
            except:
                pass
            stats_df_list.append(wrs2_stats_df)
            wrs2_stats_df = None
    stats_df = pd.concat(stats_df_list)
    stats_df = stats_df[stats_df['CLOUD_COVER_LAND'] < 71]
    #stats_df = stats_df[stats_df['CLOUD_COVER_LAND'] >= 0]
    #stats_df = stats_df[~stats_df['SCENE_ID'].isin(scene_skip_list)]

    stats_df['MASKED_PIXELS'] = (
        stats_df['CLOUD_PIXELS'] + stats_df['CIRRUS_PIXELS'] + stats_df['DILATE_PIXELS']
        + stats_df['SHADOW_PIXELS']
        + stats_df['SNOW_PIXELS']
        # + stats_df['WATER_PIXELS']
        + stats_df['ACCA_PIXELS']
        # + stats_df['SATURATED_PIXELS']
    )
    stats_df['CLOUD_COUNT_RATIO'] = stats_df['MASKED_PIXELS'] / stats_df['TOTAL_PIXELS']
    stats_df = stats_df[stats_df['CLOUD_COUNT_RATIO'] < (count_threshold_pct_max / 100)]
    stats_df = stats_df[stats_df['CLOUD_COUNT_RATIO'] >= (count_threshold_pct_min / 100)]

    scene_id_list = [row["SCENE_ID"].upper() for i, row in stats_df.iterrows()]
    random.shuffle(scene_id_list)
    print('  Scene count: {}'.format(len(scene_id_list)))

    
    # # Build the initial image ID list from the collections
    # print('\nBuilding scene ID list')
    # scene_id_list = []
    # for year in sorted(years, reverse=False):
    #     print(f'  {year} - {start_month:>2d} {end_month>2d}')
    #     scene_id_list.extend(get_scene_ids(
    #         year, wrs2_list, start_month, end_month, cloud_cover_min=68, cloud_cover_max=71
    #     ))
    # print('Image count: {}'.format(len(scene_id_list)))


    # Build the input list
    # Check if the corresponding json file already exists
    inputs = []
    index = 0
    for scene_id in scene_id_list:
        file_name = f'{scene_id.lower()}.json'
        if not overwrite_flag and (file_name in files):
            # print(f'  {scene_id} - skipping')
            continue
        year = int(scene_id[12:16])
        month = int(scene_id[16:18])
        file_path = os.path.join(output_ws, f'{year}', f'{month:02d}', file_name)
        inputs.append([index, scene_id, file_path])
        index += 1
    print('\nInput count: {}'.format(len(inputs)))
    # input('ENTER')

    
    # Build the output month folders if needed
    for month_ws in set(os.path.dirname(file_path) for [index, scene_id, file_path] in inputs):
        if not os.path.isdir(month_ws):
            os.makedirs(month_ws)

    # Compute
    if inputs:
        pool = multiprocessing.Pool(20)
        pool.starmap(write_json, inputs)
        pool.close()
        pool.join()
        print('  Closing pool')


def get_scene_ids(year, wrs2_tiles, start_month=1, end_month=12, cloud_cover_min=0, cloud_cover_max=101):
    """"""
    start_date = ee.Date.fromYMD(year, start_month, 1)
    end_date = ee.Date.fromYMD(year, end_month, 1).advance(1, 'month')
    
    l4_sr_coll = (
        ee.ImageCollection('LANDSAT/LT04/C02/T1_L2')
        .filterDate(start_date, end_date)
        #.filter(ee.Filter.calendarRange(year, year, 'year'))
        #.filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
        .filterBounds(ee.Geometry.BBox(-124, 25, -65, 50))
        .filter(ee.Filter.gte('CLOUD_COVER_LAND', cloud_cover_min))
        .filter(ee.Filter.lt('CLOUD_COVER_LAND', cloud_cover_max))
    )
    l5_sr_coll = (
        ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
        .filterDate(start_date, end_date)
        #.filter(ee.Filter.calendarRange(year, year, 'year'))
        #.filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
        .filterBounds(ee.Geometry.BBox(-124, 25, -65, 50))
        .filter(ee.Filter.gte('CLOUD_COVER_LAND', cloud_cover_min))
        .filter(ee.Filter.lt('CLOUD_COVER_LAND', cloud_cover_max))
    )
    l7_sr_coll = (
        ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
        .filterDate(start_date, end_date)
        #.filter(ee.Filter.calendarRange(year, year, 'year'))
        #.filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
        .filterBounds(ee.Geometry.BBox(-124, 25, -65, 50))
        .filter(ee.Filter.gte('CLOUD_COVER_LAND', cloud_cover_min))
        .filter(ee.Filter.lt('CLOUD_COVER_LAND', cloud_cover_max))
    )
    l8_sr_coll = (
        ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        .filterDate(start_date, end_date)
        #.filter(ee.Filter.calendarRange(year, year, 'year'))
        #.filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
        .filterBounds(ee.Geometry.BBox(-124, 25, -65, 50))
        .filter(ee.Filter.gte('CLOUD_COVER_LAND', cloud_cover_min))
        .filter(ee.Filter.lt('CLOUD_COVER_LAND', cloud_cover_max))
    )
    l9_sr_coll = (
        ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
        .filterDate(start_date, end_date)
        #.filter(ee.Filter.calendarRange(year, year, 'year'))
        #.filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
        .filterBounds(ee.Geometry.BBox(-124, 25, -65, 50))
        .filter(ee.Filter.gte('CLOUD_COVER_LAND', cloud_cover_min))
        .filter(ee.Filter.lt('CLOUD_COVER_LAND', cloud_cover_max))
    )
    if year < 1993:
        landsat_coll = l5_sr_coll.merge(l4_sr_coll)
    elif year in range(1993, 1999):
        landsat_coll = l5_sr_coll
    elif year in range(1999, 2012):
        landsat_coll = l5_sr_coll.merge(l7_sr_coll)
    elif year in [2012]:
        landsat_coll = l7_sr_coll
    elif year in range(2013, 2023):
        landsat_coll = l8_sr_coll.merge(l7_sr_coll)
    elif year >= 2023:
        landsat_coll = l9_sr_coll.merge(l8_sr_coll)

    l4_toa_coll = (
        ee.ImageCollection('LANDSAT/LT04/C02/T1_TOA')
        .filterDate(start_date, end_date)
        #.filter(ee.Filter.calendarRange(year, year, 'year'))
        #.filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
        .filterBounds(ee.Geometry.BBox(-125, 25, -65, 50))
    )
    l5_toa_coll = (
        ee.ImageCollection('LANDSAT/LT05/C02/T1_TOA')
        .filterDate(start_date, end_date)
        #.filter(ee.Filter.calendarRange(year, year, 'year'))
        #.filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
        .filterBounds(ee.Geometry.BBox(-125, 25, -65, 50))
    )
    l7_toa_coll = (
        ee.ImageCollection('LANDSAT/LE07/C02/T1_TOA')
        .filterDate(start_date, end_date)
        #.filter(ee.Filter.calendarRange(year, year, 'year'))
        #.filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
        .filterBounds(ee.Geometry.BBox(-125, 25, -65, 50))
    )
    l8_toa_coll = (
        ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
        .filterDate(start_date, end_date)
        #.filter(ee.Filter.calendarRange(year, year, 'year'))
        #.filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
        .filterBounds(ee.Geometry.BBox(-125, 25, -65, 50))
    )
    l9_toa_coll = (
        ee.ImageCollection('LANDSAT/LC09/C02/T1_TOA')
        .filterDate(start_date, end_date)
        #.filter(ee.Filter.calendarRange(year, year, 'year'))
        #.filter(ee.Filter.calendarRange(start_month, end_month, 'month'))
        .filterBounds(ee.Geometry.BBox(-125, 25, -65, 50))
    )

    if year < 1993:
        landsat_toa_coll = l5_toa_coll.merge(l4_toa_coll)
    elif year in range(1993, 1999):
        landsat_toa_coll = l5_toa_coll
    elif year in range(1999, 2012):
        landsat_toa_coll = l5_toa_coll.merge(l7_toa_coll)
    elif year in [2012]:
        landsat_toa_coll = l7_toa_coll
    elif year in range(2013, 2023):
        landsat_toa_coll = l8_toa_coll.merge(l7_toa_coll)
    elif year >= 2023:
        landsat_toa_coll = l9_toa_coll.merge(l8_toa_coll)

    # Joining so that we only process images that have SR and TOA available
    # This may not be needed though
    filter = ee.Filter.equals(leftField='system:index', rightField='system:index')
    landsat_coll = ee.Join.simple().apply(landsat_coll, landsat_toa_coll, filter)

    # Apply the WRS2 tile list filtering after getting the image ID list
    # Recompute the scene_id in case it was from a merged collection
    wrs2_tile_set = {f'{wrs2_tile[1:4]}{wrs2_tile[5:8]}' for wrs2_tile in wrs2_tiles}
    scene_id_list = [
        "_".join(scene_id.split("_")[-3:]).upper()
        for scene_id in landsat_coll.aggregate_array('system:index').getInfo()
        if scene_id.split('_')[-2] in wrs2_tile_set
    ]
    
    return scene_id_list


def compute_moran_stats(scene_id, scale):
    landsat_type = scene_id.upper()[0:4]
    image_id = f'LANDSAT/{landsat_type}/C02/T1_L2/{scene_id.upper()}'

    # The Ocean mask is True for water, so flip it for updateMask call so that land pixels are 1
    land_mask = ee.Image('projects/openet/assets/features/water_mask').Not()
    # Apply the NLCD/NALCMS water mask (anywhere it is water, set the ocean mask 
    land_mask = land_mask.where(ee.Image("USGS/NLCD_RELEASES/2020_REL/NALCMS").unmask(18).eq(18), 0)
    # land_mask = land_mask.And(ee.Image("USGS/NLCD_RELEASES/2020_REL/NALCMS").unmask(18).neq(18))

    if scale == 480:
        scale_str = '0K'
    elif scale == 960:
        scale_str = '1K'
    elif scale == 1920:
        scale_str = '2K'
    elif scale == 3840:
        scale_str = '4K'
    elif scale == 7680:
        scale_str = '8K'
    else:
        raise ValueError('unsupported scale')
    
    default_stats = ee.Dictionary({f'MORAN_{scale_str}': -9999})

    # Get the cloud mask (including the snow mask for now)
    qa_img = ee.Image(image_id).select(['QA_PIXEL'])
    cloud_mask = qa_img.rightShift(3).bitwiseAnd(1).neq(0)
    fmask_mask = qa_img.rightShift(3).bitwiseAnd(1).neq(0)
    
    # if cirrus_flag:
    cirrus_mask = qa_img.rightShift(2).bitwiseAnd(1).neq(0).And(fmask_mask.Not())
    fmask_mask = fmask_mask.Or(cirrus_mask)
    
    # if dilate_flag:
    dilate_mask = qa_img.rightShift(1).bitwiseAnd(1).neq(0).And(fmask_mask.Not())
    fmask_mask = fmask_mask.Or(dilate_mask)

    # if shadow_flag:
    shadow_mask = qa_img.rightShift(4).bitwiseAnd(1).neq(0).And(fmask_mask.Not())
    fmask_mask = fmask_mask.Or(shadow_mask)
    
    # if snow_flag:
    snow_mask = qa_img.rightShift(5).bitwiseAnd(1).neq(0).And(fmask_mask.Not())
    fmask_mask = fmask_mask.Or(snow_mask)

    # if water_flag:
    #water_mask = qa_img.rightShift(7).bitwiseAnd(1).neq(0).And(fmask_mask.Not())


    # Compute Moran I Calculation on the reduced resolution version of the mask
    # https://groups.google.com/g/google-earth-engine-developers/c/HX1pMp_OLZQ/m/u88cbGpVBQAJ
    mask_img = fmask_mask.updateMask(land_mask).rename('mask')
    grid_geo = [scale, 0, 15, 0, -scale, 15]
    rr_args = {'reducer': ee.Reducer.mean().unweighted(), 'maxPixels': 30000, 'bestEffort': True}
    grid_mask = (
        mask_img
        .reduceResolution(**rr_args)
        .reproject(**{'crs': qa_img.projection().crs(), 'crsTransform': grid_geo})
    )
    grid_mask = grid_mask.updateMask(grid_mask.mask().gt(0.5))
    
    # Create a list of weights for a 9x9 kernel
    row = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    # The center of the kernel is zero.
    centerRow = [1, 1, 1, 1, 0, 1, 1, 1, 1]
    # Assemble a list of lists: the 9x9 kernel weights as a 2-D matrix
    rows = [row, row, row, row, centerRow, row, row, row, row]
    # Create the kernel from the weights
    # Non-zero weights represent the spatial neighborhood.
    kernel = ee.Kernel.fixed(9, 9, rows, -4, -4, False)
    
    # regreass point on neighbor
    rn_args = {'reducer': ee.Reducer.mean(), 'kernel': kernel}
    grid_fit = (
        grid_mask
        .addBands(grid_mask.reduceNeighborhood(**rn_args).rename("mask_lag"))
        .reduceRegion(**{
            'reducer': ee.Reducer.linearFit(), 'geometry': qa_img.geometry(), 
            # 'scale': grid_2k_geo[0], 'crs': qa_img.projection().crs(), 'bestEffort': True, 
            'crs': qa_img.projection().crs(), 'crsTransform': grid_geo, 'bestEffort': False, 
            # 'tileScale': 4,
        })
    )

    grid_stats = ee.Dictionary({
        f'MORAN_{scale_str}': ee.Number(grid_fit.get("scale")),
    })

    # Combine the statistic dictionaries
    output_stats = default_stats.combine(grid_stats, overwrite=True)
    
    return ee.Feature(None, output_stats)


def write_json(index, scene_id, file_name):
    """Request the image statistics and write to the json file"""
    print(f'{index:>6d} - {scene_id} - started')

    image_stats = {'SCENE_ID': scene_id}
    # try:
    #     image_stats = compute_image_stats(scene_id).getInfo()['properties']
    # except Exception as e:
    #     print(f'{index:>6d} - {scene_id} - {e}')
    #     return
    # #pprint.pprint(image_stats)

    try:
        moran_1k_stats = compute_moran_stats(scene_id, scale=960).getInfo()['properties']
    except Exception as e:
        print(f'{index:>6d} - {scene_id} - {e}')
        return
    #pprint.pprint(moran_1k_stats)

    try:
        moran_2k_stats = compute_moran_stats(scene_id, scale=1920).getInfo()['properties']
    except Exception as e:
        print(f'{index:>6d} - {scene_id} - {e}')
        return
    #pprint.pprint(moran_2k_stats)

    try:
        moran_4k_stats = compute_moran_stats(scene_id, scale=3840).getInfo()['properties']
    except Exception as e:
        print(f'{index:>6d} - {scene_id} - {e}')
        return
    #pprint.pprint(moran_4k_stats)

    try:
        moran_8k_stats = compute_moran_stats(scene_id, scale=7680).getInfo()['properties']
    except Exception as e:
        print(f'{index:>6d} - {scene_id} - {e}')
        return
    #pprint.pprint(moran_8k_stats)
    
    image_stats.update(moran_1k_stats) 
    image_stats.update(moran_2k_stats) 
    image_stats.update(moran_4k_stats) 
    image_stats.update(moran_8k_stats) 
    output_stats = json.dumps(image_stats)
    
    try:
        with open(file_name, 'w') as out_file:
            out_file.write(output_stats)
    except Exception as e:
        print(f'{index:>6d} - {scene_id} - {e}')
        return
        
    # print(f'{index:>6d} - {scene_id} - done')


# def compute_image_stats(scene_id):
#     landsat_type = scene_id.upper()[0:4]
#     image_id = f'LANDSAT/{landsat_type}/C02/T1_L2/{scene_id.upper()}'

#     # The Ocean mask is True for water, so flip it for updateMask call so that land pixels are 1
#     land_mask = ee.Image('projects/openet/assets/features/water_mask').Not()
#     # Apply the NLCD/NALCMS water mask (anywhere it is water, set the ocean mask 
#     land_mask = land_mask.where(ee.Image("USGS/NLCD_RELEASES/2020_REL/NALCMS").unmask(18).eq(18), 0)
#     # land_mask = land_mask.And(ee.Image("USGS/NLCD_RELEASES/2020_REL/NALCMS").unmask(18).neq(18))
    
#     refl_sr_bands_dict = ee.Dictionary({
#         'LT04': ['SR_B3', 'SR_B2', 'SR_B1', 'QA_PIXEL', 'QA_RADSAT'],
#         'LT05': ['SR_B3', 'SR_B2', 'SR_B1', 'QA_PIXEL', 'QA_RADSAT'],
#         'LE07': ['SR_B3', 'SR_B2', 'SR_B1', 'QA_PIXEL', 'QA_RADSAT'],
#         'LC08': ['SR_B4', 'SR_B3', 'SR_B2', 'QA_PIXEL', 'QA_RADSAT'],
#         'LC09': ['SR_B4', 'SR_B3', 'SR_B2', 'QA_PIXEL', 'QA_RADSAT'],
#     })
#     refl_toa_bands_dict = ee.Dictionary({
#         'LT04': ['B3', 'B2', 'B1'],
#         'LT05': ['B3', 'B2', 'B1'],
#         'LE07': ['B3', 'B2', 'B1'],
#         'LC08': ['B4', 'B3', 'B2'],
#         'LC09': ['B4', 'B3', 'B2'],
#     })

#     landsat_sr_img = (
#         ee.Image(image_id)
#         .select(refl_sr_bands_dict.get('LE07'), ['SR_RED', 'SR_GREEN', 'SR_BLUE', 'QA_PIXEL', 'QA_RADSAT'])
#     )
#     # Note, we can't rename the TOA bands here since the simple cloud score is expecting a raw TOA image
#     landsat_toa_img = ee.Image(image_id.replace('T1_L2', 'T1_TOA'))

#     default_stats = ee.Dictionary({
#         'UNMASKED_PIXELS': -1, 'TOTAL_PIXELS': -1,
#         'CLOUD_PIXELS': -1, 'CIRRUS_PIXELS': -1, 'DILATE_PIXELS': -1, 
#         'SHADOW_PIXELS': -1, 'SNOW_PIXELS': -1, 'WATER_PIXELS': -1,
#         'SR_RED': -1, 'SR_GREEN': -1, 'SR_BLUE': -1,
#         'UNMASKED_SR_RED': -1, 'UNMASKED_SR_GREEN': -1, 'UNMASKED_SR_BLUE': -1,
#         'TOA_RED': -1, 'TOA_GREEN': -1, 'TOA_BLUE': -1,
#         'UNMASKED_TOA_RED': -1, 'UNMASKED_TOA_GREEN': -1, 'UNMASKED_TOA_BLUE': -1,
#         'CLOUD_COVER_LAND': landsat_sr_img.get('CLOUD_COVER_LAND'), 
#         'SCENE_ID': scene_id.upper(),
#     })

#     # Get the cloud mask (including the snow mask for now)
#     qa_img = ee.Image(landsat_sr_img.select(['QA_PIXEL']))
#     cloud_mask = qa_img.rightShift(3).bitwiseAnd(1).neq(0)
#     fmask_mask = qa_img.rightShift(3).bitwiseAnd(1).neq(0)
    
#     # if cirrus_flag:
#     cirrus_mask = qa_img.rightShift(2).bitwiseAnd(1).neq(0).And(fmask_mask.Not())
#     fmask_mask = fmask_mask.Or(cirrus_mask)
    
#     # if dilate_flag:
#     dilate_mask = qa_img.rightShift(1).bitwiseAnd(1).neq(0).And(fmask_mask.Not())
#     fmask_mask = fmask_mask.Or(dilate_mask)

#     # if shadow_flag:
#     shadow_mask = qa_img.rightShift(4).bitwiseAnd(1).neq(0).And(fmask_mask.Not())
#     fmask_mask = fmask_mask.Or(shadow_mask)
    
#     # if snow_flag:
#     snow_mask = qa_img.rightShift(5).bitwiseAnd(1).neq(0).And(fmask_mask.Not())
#     fmask_mask = fmask_mask.Or(snow_mask)

#     # if water_flag:
#     water_mask = qa_img.rightShift(7).bitwiseAnd(1).neq(0).And(fmask_mask.Not())

#     # Saturated mask (only keep unmasked saturated pixels)
#     # Flag as saturated if any of the RGB bands are saturated
#     #   or change .gt(0) to .gt(7) to flag if all RGB bands are saturated
#     # Comment out rightShift line to flag if saturated in any band
#     bitshift = ee.Dictionary({'LT04': 0, 'LT05': 0, 'LE07': 0, 'LC08': 1, 'LC09': 1});
#     saturated_mask = (
#         landsat_sr_img.select('QA_RADSAT')
#         .rightShift(ee.Number(bitshift.get(landsat_type))).bitwiseAnd(7)
#         .gt(0)
#     )
#     saturated_mask = saturated_mask.where(fmask_mask, 0)

#     # Simple cloud score (ACCA)
#     # Only keep unmasked ACCA pixels
#     acca_mask = ee.Algorithms.Landsat.simpleCloudScore(landsat_toa_img).select(['cloud']).gte(100)
#     acca_mask = acca_mask.where(fmask_mask, 0)

#     # Flip to set cloudy pixels to 0 and clear to 1
#     fmask_update_mask = fmask_mask.Not()

#     rr_mean_params = {
#         'reducer': ee.Reducer.mean().unweighted(),
#         'geometry': qa_img.geometry(), 
#         'crs': qa_img.projection().crs(), 
#         'crsTransform': [30, 0, 15, 0, -30, 15],
#         'bestEffort': False,
#         'maxPixels': 1E12,
#     }
#     rr_count_params = {
#         'reducer': ee.Reducer.count().unweighted(),
#         'geometry': qa_img.geometry(), 
#         'crs': qa_img.projection().crs(), 
#         'crsTransform': [30, 0, 15, 0, -30, 15],
#         'bestEffort': False,
#         'maxPixels': 1E12,
#     }
    
#     tile_scale = 1
#     if tile_scale > 1:
#         rr_mean_params['tileScale'] = tile_scale
#         rr_count_params['tileScale'] = tile_scale

#     refl_sr_nomask_bands = (
#         landsat_sr_img.select(['SR_RED', 'SR_GREEN', 'SR_BLUE'])
#         .multiply([0.0000275]).add([-0.2]).clamp(0, 1)
#     )
#     refl_sr_masked_bands = (
#         landsat_sr_img.select(
#             ['SR_RED', 'SR_GREEN', 'SR_BLUE'], 
#             ['UNMASKED_SR_RED', 'UNMASKED_SR_GREEN', 'UNMASKED_SR_BLUE']
#         )
#         .multiply([0.0000275]).add([-0.2]).clamp(0, 1)
#         .updateMask(fmask_update_mask)
#     )
#     refl_toa_nomask_bands = landsat_toa_img.select(
#         refl_toa_bands_dict.get(landsat_type), ['TOA_RED', 'TOA_GREEN', 'TOA_BLUE']
#     )
#     refl_toa_masked_bands = (
#         landsat_toa_img.select(
#             refl_toa_bands_dict.get(landsat_type), 
#             ['UNMASKED_TOA_RED', 'UNMASKED_TOA_GREEN', 'UNMASKED_TOA_BLUE']
#         )
#         .updateMask(fmask_update_mask)
#     )
#     refl_mean_stats = (
#         refl_sr_nomask_bands
#         .addBands(refl_sr_masked_bands)
#         .addBands(refl_toa_nomask_bands)
#         .addBands(refl_toa_masked_bands)
#         .updateMask(land_mask)
#         .reduceRegion(**rr_mean_params)
#     )

#     # Compute the masked count stats (these may be the same, not sure yet)
#     # If they are, then it may make more sense to compute the masked and unmasked count
#     count_stats = (
#         landsat_sr_img.select(['SR_RED'], ['UNMASKED_PIXELS']).updateMask(fmask_update_mask)
#         .addBands([
#             landsat_sr_img.select(['SR_RED'], ['TOTAL_PIXELS']),
#             cloud_mask.selfMask().rename(['CLOUD_PIXELS']),
#             cirrus_mask.selfMask().rename(['CIRRUS_PIXELS']),
#             dilate_mask.selfMask().rename(['DILATE_PIXELS']),
#             shadow_mask.selfMask().rename(['SHADOW_PIXELS']),
#             snow_mask.selfMask().rename(['SNOW_PIXELS']),
#             water_mask.selfMask().rename(['WATER_PIXELS']),
#             saturated_mask.selfMask().rename(['SATURATED_PIXELS']),
#             acca_mask.selfMask().rename(['ACCA_PIXELS']),
#         ])
#         .updateMask(land_mask)
#         .reduceRegion(**rr_count_params)
#     )

#     # Combine the statistic dictionaries
#     output_stats = (
#         default_stats
#         .combine(refl_mean_stats, overwrite=True)
#         .combine(count_stats, overwrite=True)
#     )
    
#     return ee.Feature(None, output_stats)

def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Moran I Scene Stats',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--years', nargs='+', help='Comma separated list and/or range of years')
    # parser.add_argument(
    #     '--key', type=utils.arg_valid_file,
    #     help='JSON key file', metavar='FILE')
    # parser.add_argument(
    #     '--project', default=None,
    #     help='Google cloud project ID to use for GEE authentication')
    # parser.add_argument(
    #     '--regex', help='Regular expression for filtering task IDs')
    # parser.add_argument(
    #     '--reverse', default=False, action='store_true',
    #     help='Process WRS2 tiles in reverse order')
    parser.add_argument(
        '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action='store_const', dest='loglevel')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = arg_parse()
    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.getLogger('googleapiclient').setLevel(logging.ERROR)

    main(years=args.years)
