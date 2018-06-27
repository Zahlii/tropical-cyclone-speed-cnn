import math

import numpy as np
import os
import pandas as pd
from datetime import datetime
from glob import glob
from tqdm import tqdm

from backend.db import TBL_STORMS
from settings import DATA_DIR

files = glob(DATA_DIR + '/**/*.jpg', recursive=True)

data = {
    'year': [],
    'basin': [],
    'storm': [],
    'time': [],
    'wind': [],
    'pressure': [],
    'file': [],
    'lat': [],
    'lng': []
}

# print('Start parsing')

for f in tqdm(files, total=len(files)):
    try:
        if '..' in f:
            continue

        if os.path.getsize(f) < 5:
            continue

        year, basin, name, type, filename = f.split('data/')[1].split('/')
        parts = filename.split('.', 7)
        date, time, sat, _, type_detail, storm, position_and_data, __ = parts

        wind, pressure, lat, lng = position_and_data.split('-')

        pressure = None if 'NA' in pressure else int(pressure.replace('mb', ''))

        wind = None if 'NA' in wind else int(wind.replace('kts', ''))

        if wind is None:
            continue

        timestamp = datetime.strptime(date + ' ' + time, '%Y%m%d %H%M')

        if 'N' in lat:
            lat = int(lat.replace('N', '')) / 10
        elif 'S' in lat:
            lat = -int(lat.replace('S', '')) / 10
        else:
            lat = int(lat) / 10

        if 'E' in lng:
            lng = int(lng.replace('E', '')) / 10
        elif 'W' in lng:
            lng = -int(lng.replace('W', '')) / 10
        else:
            lng = int(lng) / 10

        data['storm'].append(name)
        data['time'].append(timestamp)
        data['wind'].append(wind)
        data['pressure'].append(pressure)
        data['basin'].append(basin)
        data['year'].append(year)
        data['file'].append(f)
        data['lat'].append(lat)
        data['lng'].append(lng)
    except Exception as e:
        # print(f, e)
        continue

print('Finished parsing image files.')

pd.set_option('expand_frame_repr', False)

df = pd.DataFrame.from_dict(data)[['year', 'basin', 'storm', 'time', 'wind', 'pressure', 'file', 'lat', 'lng']]

df = df.sort_values(['year', 'basin', 'storm', 'time'])

df = df.groupby(['year', 'basin', 'storm', 'time', 'wind', 'pressure']).agg({
    'file': ['first', 'last', 'count'],
    'lat': 'first',
    'lng': 'first'
})

df = df.reset_index(drop=False)

df.columns = ['year', 'basin', 'storm', 'time', 'wind', 'pressure', 'file_ir', 'file_wv', 'file_cnt', 'lat', 'lng']

df = df[df['file_cnt'] >= 2]

print('Total pairs', len(df))

df['n_images'] = 1

pd.to_pickle(df, 'data/all_obs.pkl', compression='gzip')

exit()

per_storm = df.groupby(['year', 'storm'], as_index=False).agg({
    'basin': 'first',
    'wind': max,
    'pressure': min,
    'time': [min, max],
    'n_images': len
})

per_storm = per_storm.reset_index(drop=True)
per_storm.columns = ['year', 'storm', 'basin', 'max_speed', 'min_pressure', 'start_time', 'end_time', 'n_images']

per_storm = per_storm.sort_values(['year', 'basin', 'start_time'])


def date_handler(obj):
    return obj.isoformat() if isinstance(obj, (datetime, datetime.date)) else None


def get_category(basin, speed):
    # kts
    # speed /= 1.852
    ## SSHS
    if speed >= 137:
        return '5'
    elif speed >= 113:
        return '4'
    elif speed >= 96:
        return '3'
    elif speed >= 83:
        return '2'
    elif speed >= 64:
        return '1'
    elif speed >= 34:
        return 'TS'
    else:
        return 'TD'


# TBL_STORMS.ensure_index()

for __, s in tqdm(per_storm.iterrows(), total=len(per_storm), desc='Generating Storm DB entries'):
    n = s['storm']
    y = s['year']
    b = s['basin']

    c = get_category(b, s['max_speed'])
    k = y + ' - ' + b + ' - ' + n

    if TBL_STORMS.find_one({'key': k}):
        print('Skipping, %s - already added' % k)
        continue

    sub_df = df[(df['storm'] == n) & (df['year'] == y)]
    sub_df = sub_df.sort_values('time')

    data = []
    last_time = None

    last_set_wind = None
    all_coords = []

    ace_s = 0
    last_lat = None
    last_lng = None

    for _, r in sub_df.iterrows():
        wind = r['wind']
        pressure = r['pressure']
        time = r['time']

        if math.isnan(wind) or math.isnan(pressure):
            continue

        ace = 0
        if last_time is None or time - last_time >= pd.Timedelta('6 hours'):
            last_time = time
            ace = wind * wind / 10000
            ace_s += ace

        coords = [float(r['lng']), float(r['lat'])]
        lng, lat = coords

        if (last_lat is None or lat != last_lat) or (last_lng is None or lng != last_lng):
            last_lat = lat
            last_lng = lng
            all_coords.append(coords)

        data.append({
            'obs_time': time,
            'wind': wind,
            'pressure': pressure,
            'wv_file': r['file_wv'],
            'ir_file': r['file_ir'],
            'ace': ace,
            'ace_cum': ace_s,
            'category': get_category(b, wind),
            'loc': {'type': 'Point', 'coordinates': coords}
        })

    if len(all_coords) > 1:
        geo_type = 'LineString'
    else:
        geo_type = 'Point'
        all_coords = all_coords[0]

    data_dict = {
        'key': k,
        'name': n,
        'year': int(y),
        'basin': b,
        'total_ace': ace_s,
        'max_category': c,
        'min_pressure': s['min_pressure'],
        'max_speed': s['max_speed'],
        'start_time': s['start_time'],
        'end_time': s['end_time'],
        'n_observations': len(data),
        # 'duration': s['end_time'] - s['start_time'],
        'observations': data,
        'track': {'type': geo_type, 'coordinates': all_coords}
    }

    TBL_STORMS.insert_one(data_dict, bypass_document_validation=True)


def n_maj(v):
    return len(np.where((v == '5') | (v == '4') | (v == '3'))[0])


def n_hur(v):
    return len(np.where((v == '5') | (v == '4') | (v == '3') | (v == '2') | (v == '1'))[0])

"""
db.getCollection('storms').aggregate([{'$group': {'_id': {'year': '$year', 'basin': '$basin'},
                                                  'total_ace': {'$sum': '$total_ace'},
                                                  'max_ace': {'$max': '$total_ace'},
                                                  'min_pressure': {'$min': '$min_pressure'},
                                                  'max_speed': {'$max': '$max_speed'}}}])
"""