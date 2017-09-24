import _pickle as pkl
import os
from datetime import datetime
from glob import glob

import pandas as pd

data_dir = os.path.abspath('D:/Dokumente/NOAA/')

files = glob(data_dir + '\**\*.jpg',recursive=True)

print(len(files))

data = {
    'year': [],
    'basin': [],
    'storm': [],
    'time': [],
    'wind': [],
    'pressure': [],
    'file': []
}

for f in files:
    try:
        year, basin, name, type, filename = f.split('NOAA\\')[1].split('\\')
        parts = filename.split('.', 7)
        date, time, sat, _, type_detail, storm, position_and_data, __ = parts

        wind, pressure, lat, lng = position_and_data.split('-')

        pressure = None if 'NA' in pressure else int(pressure.replace('mb', ''))

        wind = None if 'NA' in wind else int(wind.replace('kts', ''))

        timestamp = datetime.strptime(date + ' ' + time, '%Y%m%d %H%S')

        data['storm'].append(name)
        data['time'].append(timestamp)
        data['wind'].append(wind)
        data['pressure'].append(pressure)
        data['basin'].append(basin)
        data['year'].append(year)
        data['file'].append(f)
    except Exception as e:
        print(f, e)
        continue

pd.set_option('expand_frame_repr', False)

df = pd.DataFrame.from_dict(data)[['year','basin','storm','time','wind','pressure','file']]

df = df.sort_values(['year', 'basin', 'storm', 'time'])

print(df.head(5))

df['n_images'] = 1

with open('all_obs.pkl', 'wb') as f:
    pkl.dump(file=f, obj=df, protocol=-1)

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

print(per_storm.head(5))

with open('per_storm.pkl', 'wb') as f:
    pkl.dump(file=f, obj=per_storm, protocol=-1)

per_year = df.groupby(['year', 'basin'], as_index=False).agg({
    'wind': max,
    'pressure': min,
    'time': [min, max],
    'n_images': len
})

per_year = per_year.reset_index(drop=True)
per_year.columns = ['year', 'basin', 'max_speed', 'min_pressure', 'start_time', 'end_time', 'n_images']
per_year = per_year.sort_values(['year', 'basin', 'start_time'])

print(per_year.head(50))

with open('per_year.pkl', 'wb') as f:
    pkl.dump(file=f, obj=per_year, protocol=-1)
