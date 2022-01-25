import pandas as pd
from pathlib import Path
import os

def load_data():
    # lane change annotations
    lca_dfs = []
    for i in range(3):
        lca_dfs.append(pd.read_csv(
            f'data/lane change annotations/lane change annotations with sections trip {i+1}.csv'))

    # trips
    '''load all csvs from dir'''

    csv_dir = Path('data/gps')
    all_trips = []
    all_trip_names = []
    for trip_nr in range(3):
        trip_dfs = []
        trip_names = []
        filenames = [x for x in os.listdir(
            csv_dir) if x[-3:] == 'csv' and f'0{trip_nr+1}' in x]
        for filename in filenames:
            trip_dfs.append(pd.read_csv(csv_dir / filename))
            trip_names.append(filename)
        all_trips.append(trip_dfs)
        all_trip_names.append(trip_names)

    # roads
    roads = []
    for i in range(3):
        roads.append(pd.read_csv(f'data/roads/route {i+1} mindist10.csv'))

    return lca_dfs, all_trips, all_trip_names, roads
