import _pickle as pkl
import json
import math
import os
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm

data_dir = os.path.abspath('D:/Dokumente/NOAA/')

files = glob(data_dir + '\**\*.jpg',recursive=True)


data = {
    'year': [],
    'basin': [],
    'storm': [],
    'time': [],
    'wind': [],
    'pressure': [],
    'file': []
}

# print('Start parsing')

for f in tqdm(files, total=len(files)):
    try:
        year, basin, name, type, filename = f.split('NOAA\\')[1].split('\\')
        parts = filename.split('.', 7)
        date, time, sat, _, type_detail, storm, position_and_data, __ = parts

        wind, pressure, lat, lng = position_and_data.split('-')

        pressure = None if 'NA' in pressure else int(pressure.replace('mb', ''))

        wind = None if 'NA' in wind else int(wind.replace('kts', ''))

        if wind is None:
            continue

        timestamp = datetime.strptime(date + ' ' + time, '%Y%m%d %H%S')

        data['storm'].append(name)
        data['time'].append(timestamp)
        data['wind'].append(wind)
        data['pressure'].append(pressure)
        data['basin'].append(basin)
        data['year'].append(year)
        data['file'].append(f)
    except Exception as e:
        #print(f, e)
        continue

# print(len(data['storm']))
#print('Finished')

pd.set_option('expand_frame_repr', False)

df = pd.DataFrame.from_dict(data)[['year','basin','storm','time','wind','pressure','file']]

df = df.sort_values(['year', 'basin', 'storm', 'time'])



df['n_images'] = 1

with open('all_obs.pkl', 'wb') as f:
    pkl.dump(file=f, obj=df, protocol=-1)

print('Saved all obs')



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


data_dict = {}
date_handler = lambda obj: (
    obj.isoformat()
    if isinstance(obj, (datetime, datetime.date))
    else None
)

ace = []
cat = []


def get_category(basin, speed):
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


for __, s in tqdm(per_storm.iterrows(), total=len(per_storm)):
    n = s['storm']
    y = s['year']
    b = s['basin']

    c = get_category(b, s['max_speed'])
    k = y + ' - ' + b + ' - ' + n

    sub_df = df[(df['storm'] == n) & (df['year'] == y)]
    sub_df = sub_df.sort_values('time')

    data = []
    last_wind = None
    last_pressure = None
    last_time = None
    ace_s = 0
    for _, r in sub_df.iterrows():
        wind = r['wind']
        pressure = r['pressure']
        time = r['time']

        if math.isnan(wind) or math.isnan(pressure):
            continue

        if last_time is None or time - last_time >= pd.Timedelta('6 hours'):
            last_time = time
            ace_s += wind * wind / 10000

        if last_wind is None or last_wind != wind or last_pressure is None or last_pressure != pressure:
            data.append([r['time'], wind, pressure])
            last_wind = wind
            last_pressure = pressure

    ace_s = round(ace_s,1)

    data_dict[k] = {
        'name': n,
        'year': y,
        'basin': b,
        'ace': ace_s,
        'category':c,
        'min_pressure': s['min_pressure'],
        'max_speed': s['max_speed'],
        'start_time': s['start_time'],
        'end_time': s['end_time'],
        'data': data
    }

    ace.append(ace_s)
    cat.append(c)

per_storm['ace'] = ace
per_storm['category'] = cat

print(per_storm.head(15))

with open('per_storm.pkl', 'wb') as f:
    pkl.dump(file=f, obj=per_storm, protocol=-1)

print('Saved per storm')

with open('storms.json', 'w') as f:
    json.dump(fp=f, obj=data_dict, default=date_handler)

print('Saved JSON')


def n_maj(v):
    return len(np.where((v == '5') | (v == '4') | (v == '3'))[0])


def n_hur(v):
    return len(np.where((v == '5') | (v == '4') | (v == '3') | (v == '2') | (v == '1'))[0])


per_year = per_storm.groupby(['year', 'basin'], as_index=False).agg({
    'max_speed': max,
    'min_pressure': min,
    'start_time': min,
    'end_time': max,
    'ace': [max, sum],
    'n_images': [len, sum],
    'category': [n_maj, n_hur]
})

per_year = per_year.reset_index(drop=True)
per_year.columns = ['year', 'basin', 'max_speed', 'min_pressure', 'start_time', 'end_time', 'max ace', 'sum ace',
                    'n_storms', 'n_images', 'n_maj', 'n_hur']
per_year = per_year.sort_values(['year', 'basin', 'start_time'])

print(per_year.head(15))

with open('per_year.pkl', 'wb') as f:
    pkl.dump(file=f, obj=per_year, protocol=-1)

print('Saved per year')


def to_md(df, file):
    if 'file' in df:
        del df['file']
    cols = df.columns
    df2 = pd.DataFrame([['---', ] * len(cols)], columns=cols)
    df3 = pd.concat([df2, df])
    # Save as markdown
    df3 = df3.round(1)
    df3.to_csv(file, sep="|", index=False, float_format='%.1f')


to_md(df.head(10), 'all.md')
to_md(per_year.head(10), 'per_year.md')
to_md(per_storm.head(10), 'per_storm.md')
