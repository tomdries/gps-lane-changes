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
    dt =    np.where(dt==0, np.nan, dt)
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


def calculate_signals_trip(trip_df, params):

    trip_df['lat_vel'] = derivative(trip_df.t, trip_df['offset'])
    trip_df['offset_ewmfilt'] = trip_df.offset.ewm(span=params['EWM_SPAN']).mean()
    trip_df['lat_vel_ewmfilt'] = derivative(trip_df.t, trip_df['offset_ewmfilt'])

    trip_df['bearing_diff'] = (trip_df.bearing_gps- trip_df.bearing_road) * np.pi/180
    trip_df['proj_lat_dist'] = np.sin(trip_df.bearing_diff)*trip_df.distance_gps

    trip_df['bearing_diff_ewmfilt'] = trip_df.bearing_diff.ewm(span=params['EWM_SPAN']).mean()
    trip_df['proj_lat_dist_ewmfilt'] = np.sin(trip_df.bearing_diff_ewmfilt) * trip_df.distance_gps
    trip_df['proj_lat_vel_ewmfilt'] = derivative(trip_df.t,trip_df.proj_lat_dist_ewmfilt)

    trip_df['offset_medfilt'] = trip_df.offset.rolling(params['MEDFILT_SIZE']).median()
    trip_df['lat_vel_medfilt'] = derivative(trip_df.t, trip_df['offset_medfilt'])

    trip_df['bearing_diff_medfilt'] = trip_df.bearing_diff.rolling(params['MEDFILT_SIZE']).median()
    trip_df['proj_lat_dist_medfilt'] = np.sin(trip_df.bearing_diff_medfilt) * trip_df.distance_gps
    

    # lookback offset
    for col in ['offset', 'offset_ewmfilt']:
        lookback_offset = [np.nan]*len(trip_df)
        for i in range(len(trip_df)-params['LOOKBACK_DISTDIFF']):
            under = trip_df[col].iloc[i]
            upper = trip_df[col].iloc[i+params['LOOKBACK_DISTDIFF']]
            lookback_offset[i+params['LOOKBACK_DISTDIFF']]=upper-under
        trip_df[f'lookback_{col}'] = lookback_offset

    return trip_df

def calculate_signals_alltrips(trip_dfs, params):
    for trip_nr in range(3):
        for dev_nr, trip_df in enumerate(trip_dfs[trip_nr]):
            trip_df = calculate_signals_trip(trip_df, params)
            trip_dfs[trip_nr][dev_nr] = trip_df
    return trip_dfs
    

def extract_baseline_fragments_trip(signal_col, trip_df, lca_df, window_size, min_dist):
    
    t_start_lc = lca_df.t_lc - min_dist 
    t_end_lc = lca_df.t_lc + min_dist 

    # What are intervals without lane change annotations
    intervals_without_lane_change = [(trip_df.t.iloc[0], t_start_lc.iloc[0])]
    for i in range(len(lca_df)-1):
        intervals_without_lane_change.append((t_end_lc.iloc[i], t_start_lc.iloc[i+1]))
    intervals_without_lane_change.append((t_end_lc.iloc[-1], trip_df.t.iloc[-1]))

    # Remove intervals that last shorter than window_size
    intervals_without_lane_change = [x for x in intervals_without_lane_change if x[1]-x[0]>window_size]

    # split the intervals up in non-overlapping fragments of size window_size
    fragments_without_lane_change = []
    for interval in intervals_without_lane_change:
        interval_duration = interval[1] - interval[0]
        fragment_bounds = np.arange(interval[0], interval[1], window_size)
        for i in range(len(fragment_bounds) - 1):
            fragments_without_lane_change.append((fragment_bounds[i], fragment_bounds[i+1]))
    
    
    # remove fragments that contain no data
    t_fragments = fragments_without_lane_change
    for i, t_fragment in enumerate(t_fragments):
        fragment = trip_df[trip_df.t.between(t_fragment[0], t_fragment[1])]
        if len(fragment)<2:
            t_fragments[i] = 0
    t_fragments = [x for x in t_fragments if x!=0] 
    
    # get trip fragments
    fragment_dfs = [np.nan]* len(t_fragments)
    for i, t_fragment in enumerate(t_fragments):
        fragment_df = trip_df[trip_df.t.between(t_fragment[0], t_fragment[1])].copy()

        fragment_df['t_fragment'] = fragment_df.t - (fragment_df.t.iloc[0] + fragment_df.t.iloc[-1])/2#fragment_df.t - (fragment_df.t.iloc[0] + window_size/2)
        fragment_df['relative_signal'] = fragment_df[signal_col] - fragment_df.loc[abs(fragment_df.t_fragment).idxmin(), signal_col]

        fragment_dfs[i] = fragment_df
    
    return fragment_dfs


def extract_baseline_fragments_alltrips(signal_col, trip_dfs, trip_names, lca_dfs, dev_nr, sections, window_size, min_dist):
    # extract fragments for all 3 trips and dump in one list
    baseline_fragments_alltrips = [] #for one device
    for trip_nr in range(3):
        trip_df, lca_df, trip_name = selector(trip_dfs, lca_dfs, trip_names, trip_nr, dev_nr, sections)
        fragment_dfs = extract_baseline_fragments_trip(signal_col, trip_df, lca_df, window_size, min_dist)
        baseline_fragments_alltrips.extend(fragment_dfs)
    return baseline_fragments_alltrips


def extract_lanechange_fragments_trip(signal_col, trip_df, lca_df, window_size):
    lca_df['t0_w'] = lca_df.t_lc-(window_size/2) # boundaries for all windows
    lca_df['t_end_w'] = lca_df.t_lc+(window_size/2)

    fragment_dfs = [[],[]] #left, right

    for i, lca_df_row in lca_df.iterrows(): # for lane changes in lca_df
        if lca_df_row.direction == 'left':
            fragment_df = trip_df[trip_df.t.between(lca_df_row.t0_w, lca_df_row.t_end_w)].copy()
            fragment_df['t_fragment'] = fragment_df.t - lca_df_row.t_lc#(fragment_df.t.iloc[0] + fragment_df.t.iloc[-1])/2
            fragment_df['relative_signal'] = fragment_df[signal_col] - fragment_df.loc[abs(fragment_df.t_fragment).idxmin(), signal_col]
            fragment_dfs[0].append(fragment_df)

        elif lca_df_row.direction == 'right':
            fragment_df = trip_df[trip_df.t.between(lca_df_row.t0_w, lca_df_row.t_end_w)].copy()
            if len(fragment_df) > 0:
                fragment_df['t_fragment'] = fragment_df.t - lca_df_row.t_lc # (fragment_df.t.iloc[0] + fragment_df.t.iloc[-1])/2# lca_df_row.t_lc
                fragment_df['relative_signal'] = fragment_df[signal_col] - fragment_df.loc[abs(fragment_df.t_fragment).idxmin(), signal_col]
                fragment_dfs[1].append(fragment_df)
            else: 
                print('ignored_fragment')                
    return fragment_dfs


def extract_lanechange_fragments_alltrips(signal_col, trip_dfs, trip_names, lca_dfs, dev_nr, sections, window_size):
    fragments_all_trips = [[],[]]

    for trip_nr in range(3):
        trip_df, lca_df, trip_name = selector(trip_dfs, lca_dfs, trip_names, trip_nr, dev_nr, sections)
        fragment_dfs = extract_lanechange_fragments_trip(signal_col, trip_df, lca_df, window_size=window_size)
        fragments_all_trips[0].extend(fragment_dfs[0])
        fragments_all_trips[1].extend(fragment_dfs[1])
    return fragments_all_trips

def make_lanechange_fragments_unidirectional(fragments, signal_col):
    '''reverse the direction of the left lane changes so they can be treated as one class'''
    fragments_left=fragments[0]
    fragments_right = fragments[1]

    fragments_positive = [np.nan]*(len(fragments_left)+len(fragments_right))

    # "reverse" offset of left fragments
    for i,fragment in enumerate(fragments_left): 
        fragment['offset_pos'] = -fragment[signal_col]
        fragments_positive[i] = fragment
    for i,fragment in enumerate(fragments_right): 
        fragment['offset_pos'] = fragment[signal_col]
        fragments_positive[i+len(fragments_left)] = fragment

    return fragments_positive


def plot_fragments(fragment_dfs, plot_col, params):
    fig, axs = plt.subplots(1,2, figsize=(8,2), sharey=True)
    
    for fragment_df in fragment_dfs[0]: #left
        axs[0].plot(fragment_df.t_fragment, fragment_df[plot_col], color='black', alpha=params['alpha'])
    for fragment_df in fragment_dfs[1]: #right
        axs[1].plot(fragment_df.t_fragment, fragment_df[plot_col], color='black', alpha=params['alpha'])
        
    if params['ylims']: 
        axs[0].set_ylim(params['ylims'][0], params['ylims'][1])
        axs[1].set_ylim(params['ylims'][0], params['ylims'][1])
    axs[0].title.set_text('left')
    axs[1].title.set_text('right')
    axs[0].set_ylabel(params['ylabel'])
    
    n_left = len(fragment_dfs[0])
    n_right = len(fragment_dfs[1])

    axs[0].text(-4,-2, f"n={n_left}", size=15)
    axs[1].text(-4,1.5, f"n={n_right}", size=15)

    plt.tight_layout() 

def get_results(fragments_positive, fragments_negative, threshold):
    
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    y_true = []
    y_score = []

    for fragment_pos in fragments_positive:
        y_true.append(1)
        offset_diff = fragment_pos.offset_pos.iloc[-1] - fragment_pos.offset_pos.iloc[0]
        if offset_diff > threshold: #if positive class is assigned:
            TP+=1
            y_score.append(1)
        else:
            FN+=1
            y_score.append(0)

    for fragment_neg in fragments_negative:
        y_true.append(0)
        offset_diff = fragment_neg.offset.iloc[-1] - fragment_neg.offset.iloc[0]
        if offset_diff > threshold: #if positive class is assigned:
            FP+=1
            y_score.append(1)
        else:
            TN +=1
            y_score.append(0)
            
    accuracy = (TP+TN)/(TP+FP+TN+FN)
    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    return y_true, y_score, TP, FP, TN, FN, accuracy, TPR, FPR

def plot_fragments_offset(fragment_dfs, plot_col, params):
    fig, axs = plt.subplots(1,2, figsize=(8,2), sharey=True)
    
    for fragment_df in fragment_dfs[0]: #left
        axs[0].plot(fragment_df.t_fragment, fragment_df[plot_col], color='black', alpha=params['alpha'])
    for fragment_df in fragment_dfs[1]: #right
        axs[1].plot(fragment_df.t_fragment, fragment_df[plot_col], color='black', alpha=params['alpha'])
        
    if params['ylims']: 
        axs[0].set_ylim(params['ylims'][0], params['ylims'][1])
        axs[1].set_ylim(params['ylims'][0], params['ylims'][1])
        axs[2].set_ylim(params['ylims'][0], params['ylims'][1])
    if params['xlims']:
        axs[0].set_xlim(params['xlims'][0], params['xlims'][1])
        axs[1].set_xlim(params['xlims'][0], params['xlims'][1])
        axs[2].set_xlim(params['xlims'][0], params['xlims'][1])
    axs[0].title.set_text('left')
    axs[1].title.set_text('right')
    axs[0].set_ylabel(params['ylabel'])
    
    n_left = len(fragment_dfs[0])
    n_right = len(fragment_dfs[1])

    axs[0].text(-4,-2, f"n={n_left}", size=15)
    axs[1].text(-4,1.5, f"n={n_right}", size=15)

    plt.tight_layout() 


def plot_fragments_offset_with_baseline(fragment_dfs,fragment_dfs_baseline, plot_col, params):
    fig, axs = plt.subplots(1,3, figsize=(8,2), sharey=True)
    
    for fragment_df in fragment_dfs[0]: #left
        axs[0].plot(fragment_df.t_fragment, fragment_df[plot_col], color='black', alpha=params['alpha'])
    for fragment_df in fragment_dfs[1]: #right
        axs[1].plot(fragment_df.t_fragment, fragment_df[plot_col], color='black', alpha=params['alpha'])
    for fragment_df in fragment_dfs_baseline:
        axs[2].plot(fragment_df.t_fragment, fragment_df[plot_col], color='black', alpha=params['alpha'])

    if params['ylims']: 
        axs[0].set_ylim(params['ylims'][0], params['ylims'][1])
        axs[1].set_ylim(params['ylims'][0], params['ylims'][1])
        axs[2].set_ylim(params['ylims'][0], params['ylims'][1])
    if params['xlims']:
        axs[0].set_xlim(params['xlims'][0], params['xlims'][1])
        axs[1].set_xlim(params['xlims'][0], params['xlims'][1])
        axs[2].set_xlim(params['xlims'][0], params['xlims'][1])
    axs[0].title.set_text('left lane change')
    axs[1].title.set_text('right lane change')
    axs[2].title.set_text('no lane change')
    axs[0].set_ylabel(params['ylabel'],size=11)
    axs[1].set_xlabel(params['xlabel'],size=12)
    
    
    n_left = len(fragment_dfs[0])
    n_right = len(fragment_dfs[1])
    n_baseline = len(fragment_dfs_baseline)

    axs[0].text(-1.2,2.5, f"n={n_left}", size=13)
    axs[1].text(-1.2,2.5, f"n={n_right}", size=13)
    axs[2].text(-1.2,2.5, f"n={n_baseline}", size=13)
    plt.tight_layout() 


def precision_recall_f1(TP, FN, FP): 
    precision = TP/(TP+FP) # positive predictive value, how many detections are relevant
    recall = TP/(TP+FN) # hit rate
    f1 = 2*TP / (2*TP + FP + FN)
    return precision, recall, f1


def round_sort_cols(df):
    df.accuracy =df.accuracy.astype(float).round(3)
    df.TPR = df.TPR.astype(float).round(3)
    df.FPR = df.FPR.astype(float).round(3)
    return df[['device', 'sections', 'window_size', 'threshold','accuracy', 'TP', 'FP', 'TN',
       'FN', 'TPR', 'FPR']]