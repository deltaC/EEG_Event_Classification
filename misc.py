from datetime import datetime
import pandas as pd
import numpy as np


def get_two_side_events(meas_date: str, timestamps_filepath: str, sample_rate: float) -> dict:
    """
    1. meas_data = raw.info['meas_date'], 
        where raw is the raw edf file object. Ex.:
        raw = mne.io.read_raw_edf('EEG.edf')

    2. timestamps_filepath is the path to csv file.
        Format:     Ear            ---    Time
        Ex.:        Left / Right   ---    2024-10-09 16:02:40.026203
    """

    stamps = pd.read_csv(timestamps_filepath)
    stamps.head()

    left_stamps = [datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S.%f') for stamp in stamps[stamps['Ear'] == 'Left']['Time']]
    right_stamps = [datetime.strptime(stamp, '%Y-%m-%d %H:%M:%S.%f') for stamp in stamps[stamps['Ear'] == 'Right']['Time']]

    initial_stamp = datetime.strptime(datetime.strftime(meas_date, '%Y-%m-%d %H:%M:%S.%f'), '%Y-%m-%d %H:%M:%S.%f')

    events = {
        'left': [round(sample_rate * (left_stamps[i] - initial_stamp).total_seconds()) for i in range(len(left_stamps))],
        'right': [round(sample_rate * (right_stamps[i] - initial_stamp).total_seconds()) for i in range(len(right_stamps))]
    }
    return events


def get_trials_from_two_side_events(data: np.ndarray, sample_rate: float, events: dict, 
                                    n_channels: int, onset = 0.1, offset = 0.1) -> tuple:
    """
    1. The window will be fit on (-onset, offset) in seconds
    2. Events is in the format as in function get_events()
    

    Function returns tuple (trials_l, trials_r)
    """
    win = np.arange(int(-onset * sample_rate), int(offset * sample_rate))

    trials_l = []
    trials_r = []

    for side in events.keys():
        for event in events[side]:
            for i in win:
                col = []
                for j in range(n_channels):
                    col.append(data[event + i][j] * 1e6)
                    
                if side == 'right':
                    trials_r.append(col)
                else:
                    trials_l.append(col)
            
    trials_l = np.array(trials_l)
    trials_r = np.array(trials_r)
    
    n_events_l = len(events['left'])
    n_events_r = len(events['right'])
    trials_l = trials_l.T.reshape((n_channels, len(win), n_events_l))
    trials_r = trials_r.T.reshape((n_channels, len(win), n_events_r))
    
    return (trials_l, trials_r)


def get_trials_from_events(data: np.ndarray, sample_rate: float, events: dict, 
                                    n_channels: int, onset = 0.1, offset = 0.1) -> np.ndarray:
    """
    1. The window will be fit on (-onset, offset) in seconds    

    Function returns np.ndarray   trials
    """
    win = np.arange(int(-onset * sample_rate), int(offset * sample_rate))

    trials = []

    for event in events:
        for i in win:
            col = []
            for j in range(n_channels):
                col.append(data[event + i][j] * 1e6)
                
            trials.append(col)
            
    trials = np.array(trials)
    
    n_events = len(events)
    trials = trials.T.reshape((n_channels, len(win), n_events))
    
    return trials


def get_corr_matrix(trials: np.ndarray) -> np.ndarray:
    """
    The function returns the correlations between trials
    """
    n_channels, _, n_trials = trials.shape
    
    corrs = []
    for i in range(n_trials):
        corr_mat = []
        for j in range(n_trials):
            corr_mat_row = []
            for k in range(n_channels):
                corr_mat_row.append(correlation(trials[k, :, i], trials[k, :, j]))
            corr_mat.append(corr_mat_row)
        corrs.append(corr_mat)

    corrs = np.array(corrs)
    return corrs


def correlation(in1: np.ndarray, in2: np.ndarray) -> float:
    N = len(in1)
    numenator = np.sum((in1 - np.mean(in1)) * (in2 - np.mean(in2)))
    denumenator = np.std(in1) * np.std(in2) * N ** 2
    return numenator / denumenator