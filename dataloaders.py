from pathlib import Path
import pandas as pd
import datetime
import numpy as np
import pyproj
from fastdtw import fastdtw
import os

def fix_time_globalsat(df):
    '''something went wrong in the globalsats time registration, here I retrieve the recording time'''
    date = datetime.date(1, 1, 1)
    df['time'] = df['timestamp'].dt.time

    df['datetime_temp'] = df.time.apply(lambda x: datetime.datetime.combine(date, x))
    df = df.sort_values(by='datetime_temp').reset_index()
    duration = df.datetime_temp.apply(lambda x: x - df.datetime_temp[0]).dt.total_seconds()
    df['duration_local'] = duration    
    df = df.drop(columns = 'datetime_temp')
    return df


def haversine_distance(p1, p2):
    lon1 = p1[0] * np.pi/180
    lat1 = p1[1] * np.pi/180
    lon2 = p2[0] * np.pi/180
    lat2 = p2[1] * np.pi/180
    
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    
    R = 6371e3
    a = (np.sin(d_lat/2) * np.sin(d_lat/2) + 
        np.cos(lat1) * np.cos(lat2) *
         np.sin(d_lon/2) * np.sin(d_lon/2))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d


def haversine_speed(df, lon_col = 'lon', lat_col = 'lat', t_col = 'duration_local', max_speed=1e99):
    speed_calc = [np.nan] * len(df)
    for i in range(len(df)):
        if i>0:
            p1 = (df[lon_col].iloc[i-1], df[lat_col].iloc[i-1])
            p2 = (df[lon_col].iloc[i], df[lat_col].iloc[i])
            dist = haversine_distance(p1, p2)
            
                      
            dt = df[t_col].iloc[i] - df[t_col].iloc[i-1]
            
                
            if dt > 0:
                speed = dist/dt
                if speed > max_speed:
                    speed = np.nan
            else:
                speed = np.nan
            speed_calc[i] = speed
    return pd.Series(speed_calc)


def append_haversine_speed(gps_data):
    for device in gps_data.keys():
        for trip_id in gps_data[device].keys():
            df = gps_data[device][trip_id]
            df['speed_haversine'] = haversine_speed(df, max_speed = 230/3.6)
    return gps_data
    

def load_gps_data_unsynced(gps_dir, calculate_speed = False):
    gps_data = {'GoPro':{},'S9': {}, 'P20': {}, 'GlobalSat': {}}
    for trip_nr in [1, 2, 3]:
        for device in ['GoPro', 'S9', 'P20', 'GlobalSat']:
            
            if device == 'GoPro':
                df = pd.read_csv(gps_dir / f'0{trip_nr} GPS {device}.csv', parse_dates=['date'])
                df.speed = df.speed/3.6
                gps_data[device][trip_nr] = df
            
            if device == 'S9':
                    df = pd.read_csv(gps_dir / f'0{trip_nr} GPS {device}.csv', parse_dates = ['time'], index_col=0)
                    df['duration_local'] = (df.time - df.time[0]).dt.total_seconds()
                    gps_data[device][trip_nr] = df
                    

            elif device == 'P20':
                df = pd.read_csv(gps_dir / f'0{trip_nr} GPS {device}.csv', parse_dates =['time'])
                if trip_nr == 1:
                    df.time = df.time.str.split(':').apply(lambda x: x[0]+':'+x[1]+':'+x[2]+'.'+x[3])
                    time_temp = pd.to_datetime(df.time)
                    df['duration_local'] = (time_temp - time_temp[0]).dt.total_seconds()
                else:
                    df['duration_local'] = (df.time - df.time[0]).dt.total_seconds()
                gps_data[device][trip_nr] = df
                
            elif device == 'GlobalSat': 
                df = pd.read_csv(gps_dir / f'0{trip_nr} GPS {device}.csv', parse_dates = ['timestamp'])
                df = fix_time_globalsat(df)
                df = df.dropna()
                df['speed'] = df['speed (mph)'] * 1.60934
                gps_data[device][trip_nr] = df

    # old trip 1 export by ultra gps logger
    df = pd.read_csv(gps_dir / f'01 GPS S9 old.csv', parse_dates = ['Time'])
    df['duration_local'] = (df.Time - df.Time[0]).dt.total_seconds()
    df = df.rename(columns={"Longitude": "lon", "Latitude": "lat"})
    gps_data['S9']['trip1_ultra'] = df

    if calculate_speed:
        gps_data = append_haversine_speed(gps_data)
                
    return gps_data


def sync_gps_data(gps_data, sync_col_gopro = 'lon',sync_col_others = 'lon'):

    sync_col_s1 = sync_col_gopro
    sync_col_s2 = sync_col_others
    for device in ['S9', 'P20', 'GlobalSat']:
        for trip_nr in [1,2,3]:
            s1 = gps_data['GoPro'][trip_nr]
            s2 = gps_data[device][trip_nr]

            x = np.array(s1[sync_col_s1].fillna(0))
            y = np.array(s2[sync_col_s2].fillna(0))

            distance, path = fastdtw(x, y)

            result = []
            for i in range(0,len(path)):
                result.append([s1['duration_local'].iloc[path[i][0]],
                               s2['duration_local'].iloc[path[i][1]],
                s1[sync_col_s1].iloc[path[i][0]],
                s2[sync_col_s2].iloc[path[i][1]]])
            df_sync = pd.DataFrame(data=result,columns=['duration_local s1', 'duration_local s2',f'{sync_col_s1} s1', f'{sync_col_s2} s2']).dropna()
            df_sync = df_sync.drop_duplicates(subset=['duration_local s1'])
            df_sync = df_sync.sort_values(by='duration_local s1')
            df_sync = df_sync.reset_index(drop=True)
            lag = (df_sync['duration_local s2'] - df_sync['duration_local s1']).median()
            s2['duration_synced'] = s2['duration_local'] - lag
    return gps_data

def load_gps_data_synced(gps_dir):

    gps_data = {'GoPro':[],'S9':[], 'P20':[], 'GlobalSat':[] }
    for trip_nr in range(3):
        for device in ['GoPro', 'S9', 'P20', 'GlobalSat']:
            df = pd.read_csv(gps_dir/f'0{trip_nr+1} GPS {device}.csv')
            if device == 'GoPro':
                df = df.rename({'duration_local': 't'}, axis=1)
                df['date'] = pd.to_datetime(df['date'])
            else:
                df = df.rename({'duration_synced': 't'}, axis=1)
                df.time = pd.to_datetime(df.time)
            
            df.index = pd.to_timedelta(df.t, unit = 'S')
            
            gps_data[device].append(df)
    return gps_data



def save_gps_data(gps_data, dir_out):
    dir_out = Path(dir_out)
    for trip_nr in [1, 2, 3]:
        for device in ['GoPro', 'S9', 'P20', 'GlobalSat']:
            gps_data[device][trip_nr].to_csv(dir_out / f'0{trip_nr} GPS {device}.csv',index=False)
       
def distance_bearing(lon_series, lat_series):
    '''calculate distance and bearing between subsequent points'''
    
    lon_series = lon_series.reset_index(drop=True)
    lat_series = lat_series.reset_index(drop=True)
    
    geod = pyproj.Geod(ellps = 'WGS84')
    
    _, bearing, distance = geod.inv(lon_series, lat_series, lon_series.shift(1), lat_series.shift(1))
    
    return distance, bearing


def batch_load_csv(csv_dir, sep_trips = True):
    '''load all csvs from dir'''
    
    csv_dir = Path(csv_dir)
    all_trips = []
    all_trip_names = []
    for trip_nr in range(3):
        trip_dfs = []
        trip_names = []
        filenames = [x for x in os.listdir(csv_dir) if x[-3:]=='csv' and f'0{trip_nr+1}' in x]
        for filename in filenames:
            trip_dfs.append(pd.read_csv(csv_dir / filename))
            trip_names.append(filename)
        all_trips.append(trip_dfs)
        all_trip_names.append(trip_names)

    return all_trips, all_trip_names