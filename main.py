# Классификация двух сигналов посредством анализа ЭЭГ

import mne 
from scipy import signal
import numpy
import pandas
import matplotlib.pyplot as plt


### Loading, filtering and extracting data
raw:mne.io.Raw = mne.io.read_raw_edf('data/Dual_Aural.edf', preload=True)
raw.drop_channels(raw.ch_names[-5:])

mapping:dict[str, str] = dict(zip(raw.ch_names, ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']))
raw.rename_channels(mapping)


### Converting into pandas dataframe
data:numpy.ndarray = raw['data'][0].T
df:pandas.DataFrame = raw.to_data_frame()
df = df.drop(['time'], axis=1)

final_columns:list[str] = list(df.columns)
final_columns.append('mark')


### Events searching
raw_events:tuple[numpy.ndarray, dict[str, int]] = mne.events_from_annotations(raw)

events_from_raw:list[list[int]] = []
for event in raw_events[0][1:-1]:
    events_from_raw.append([event[0], event[-1]])

events:list[list[int]] = []

mark_l:list[int] = list(range(17072, 17072 + int(60.1 * 500), 500))
mark_r:list[int] = list(range(154144, 154144 + int(60.1 * 500), 500))
for stamp in mark_l:
    events.append([stamp, 0])
for stamp in mark_r:
    events.append([stamp, 1])


### Segmentation
sample_rate:float = raw.info['sfreq']
window:numpy.ndarray = numpy.arange(int(0.0 * sample_rate), int(0.0 * sample_rate)) # size of windows = 100

trials:list[list[float]] = []

for event in events:
    for i in window:
        row:list = []
        for j in range(20):
            if j < 19:
                row.append(data[event[0] + i][j] * 1e6)
            else:
                row.append(event[1])
        trials.append(row)

trials:numpy.ndarray = numpy.array(trials)
trials_df:pandas.DataFrame = pandas.DataFrame(trials, columns=final_columns)

dataset_l:numpy.ndarray = trials[:6100]
dataset_l:numpy.ndarray = numpy.delete(dataset_l, numpy.s_[-1], axis=1)

dataset_r:numpy.ndarray = trials[6100:]
dataset_r:numpy.ndarray = numpy.delete(dataset_r, numpy.s_[-1], axis=1)

windows_l:numpy.ndarray = dataset_l.T.reshape(19, 100, 61)
y_l:numpy.ndarray = numpy.zeros((61, 1))

windows_r:numpy.ndarray = dataset_r.T.reshape(19, 100, 61)
y_r:numpy.ndarray = numpy.ones((61, 1))

### Logvar fuction and computing variances
def logvar(x:numpy.ndarray, axis:int=0)->numpy.ndarray:
    return numpy.log(numpy.var(x, axis=axis))

variances:list[list[float]] = []

trials_df_l:pandas.DataFrame = trials_df[trials_df['mark'] == 0]
trials_df_r:pandas.DataFrame = trials_df[trials_df['mark'] == 1]

l_var:list[float] = []
for col in final_columns[:-1]:
    l_var.append(logvar(trials_df_l[col]))
l_var.append(0.0)

variances.append(l_var)

r_var:list[float] = []
for col in final_columns[:-1]:
    r_var.append(logvar(trials_df_r[col]))
r_var.append(1.0)

variances.append(r_var)
variances:pandas.DataFrame = pandas.DataFrame(variances, columns=final_columns)


### PSD computing
f1, Pxx_den1 = signal.welch(trials_df_l['T3'], sample_rate, nperseg=256)
f2, Pxx_den2 = signal.welch(trials_df_r['T3'], sample_rate, nperseg=256)

f1_2, Pxx_den1_2 = signal.welch(trials_df_l['C4'], sample_rate, nperseg=256)
f2_2, Pxx_den2_2 = signal.welch(trials_df_r['C4'], sample_rate, nperseg=256)

figure, axes = plt.subplots(2)

axes[0].semilogy(f1, Pxx_den1, alpha=0.5)
axes[0].semilogy(f2, Pxx_den2, c='red', alpha=0.5)

axes[0].set_xlabel('frequency [Hz]')
axes[0].set_ylabel('PSD $[V^2/Hz]$')

axes[1].semilogy(f1_2, Pxx_den1_2, alpha=0.5)
axes[1].semilogy(f2_2, Pxx_den2_2, c='red', alpha=0.5)

axes[1].set_xlabel('frequency [Hz]')
axes[1].set_ylabel('PSD $[V^2/Hz]$')

plt.savefig('holy_c.jpg')


### CSP and postprocessing

### Classification
