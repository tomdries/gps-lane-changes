from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def angular_diff(h1, h2):
    a = h1-h2
    a = (a + 180) % 360 - 180
    return a

def find_nearest_road_index(gps_row, road):
    road.bearing = (road.bearing + 360)
    candidates = road[abs(angular_diff(road.bearing, gps_row.bearing_gps)) < 45] # remove far removed directions
    if len(candidates) > 0:
        nearest_index = np.sqrt((gps_row.x_gps-candidates.x)**2 + (gps_row.y_gps - candidates.y)**2).idxmin()
    else:
        nearest_index = None
        
    return nearest_index

def offset(bearing_road, x_road, y_road, x_gps,y_gps):
    '''Lateral offset from road calculation. See vector rejection second example https://en.wikipedia.org/wiki/Vector_projection'''
    b = [np.cos((90-bearing_road)*np.pi/180), np.sin((90-bearing_road)*np.pi/180)]
    a = [x_gps-x_road, y_gps-y_road]
    dist_vec = a - np.dot(np.dot(a,b) / np.dot(b, b), b)
    d = np.linalg.norm(dist_vec)
    sign = np.sign(np.cross(dist_vec, b))
    offset = d * sign
    return offset

def batch_append_offset(trip_dfs, trip_names, roads, save_path = None):
    for trip_nr in range(3):
        road = roads[trip_nr]
        i_device = 0
        for trip in trip_dfs[trip_nr]:
            filename = trip_names[trip_nr][i_device]

            trip['road_index'] = trip.apply(lambda row: find_nearest_road_index(row, road), axis=1) # find nearest road index 
            trip_matched = pd.merge(trip, road, how='left', left_on = 'road_index', right_index=True, suffixes = ("_gps", "_road"))
            trip_matched['offset'] = trip_matched.apply(lambda r: offset(r.bearing_road, r.x_road, r.y_road, r.x_gps, r.y_gps),axis=1)

            if save_path:
                trip_matched.to_csv( save_path / filename, index=False)
            i_device +=1

    return trip_dfs