from glob import glob
import os
from datetime import datetime
import pandas as pd
import _pickle as pkl

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
    year, basin, name, type, filename = f.split('NOAA\\')[1].split('\\')
    date, time, sat, _, type_detail, storm, position_and_data, __ , ___ = filename.split('.')

    wind, pressure, lat, lng = position_and_data.split('-')

    pressure = None if 'NA' in pressure else int(pressure.replace('mb',''))

    wind = None if 'NA' in wind else int(wind.replace('kts', ''))

    try:
        timestamp = datetime.strptime(date + ' ' + time, '%Y%m%d %H%S')
    except Exception as e:
        print(f,e)
        continue

    data['storm'].append(name)
    data['time'].append(timestamp)
    data['wind'].append(wind)
    data['pressure'].append(pressure)
    data['basin'].append(basin)
    data['year'].append(year)
    data['file'].append(f)


df = pd.DataFrame.from_dict(data)[['year','basin','storm','time','wind','pressure','file']]

df.sort_values(['year','basin','storm','time'])

print(df.head(20))

with open('images.pkl','wb') as f:
    pkl.dump(file=f, obj=df, protocol=-1)


