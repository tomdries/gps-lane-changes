import matplotlib.pyplot as plt
import numpy as np

def plot_lane_changes(lca_df, left_color='C0', right_color='C1'):
    for timestamp in lca_df[lca_df.direction=='left']['t_lc']:
        plt.axvline(timestamp, color=left_color)

    for timestamp in lca_df[lca_df.direction=='right']['t_lc']:
        plt.axvline(timestamp, color=right_color)
        
def derivative(t, y):
    dy = np.diff(y, prepend=np.nan)
    dt = np.diff(t,prepend=np.nan)
    dydt = dy/dt
    
    return dydt

def selector(trip_dfs, lca_dfs, trip_names, trip_nr, device_nr, sections = 'all', road_type = 'all'):
    trip_df = trip_dfs[trip_nr][device_nr].copy()
    trip_name = trip_names[trip_nr][device_nr]
    lca_df = lca_dfs[trip_nr].copy()
    
    if sections != 'all':
        lca_df = lca_df[lca_df.section.isin(sections)]
        trip_df = trip_df[trip_df.section.isin(sections)]
    
    if road_type != 'all':
        lca_df = lca_df[lca_df.section.isin(sections)]
        trip_df = trip_df[trip_df.section.isin(sections)]
    
    return trip_df, lca_df, trip_name
