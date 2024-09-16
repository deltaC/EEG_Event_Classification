# Классификация двух сигналов посредством анализа ЭЭГ

import mlp
from mlp import SignalsDataset1, SignalsDataset2, DataLoader
from mlp import MLP1, MLP2, train, eval

import torch
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
window:numpy.ndarray = numpy.arange(int(-0.1 * sample_rate), int(0.3 * sample_rate)) # size of windows = 200

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


### Logvar fuction and computing variances
def logvar(x:numpy.ndarray)->numpy.ndarray:
    return numpy.log(numpy.var(x))

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

# plt.semilogy(f1, Pxx_den1)
# plt.semilogy(f2, Pxx_den2, c='red', alpha=0.5)
# 
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD $[V^2/Hz]$')
# plt.show()

###
##
# Место для отбора каналов
##
###


# Параметры для моделей
device:str = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs:int = 100
batch_size:int = 4
learning_rate:float = 0.001


### Полносвязная модель 1.0
df_train:numpy.ndarray = trials[:int(0.8*trials.shape[0])]
df_test:numpy.ndarray = trials[int(0.8*trials.shape[0]):]

train_dataset:SignalsDataset1 = SignalsDataset1(df_train)
test_dataset:SignalsDataset1 = SignalsDataset1(df_test)

train_loader:DataLoader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader:DataLoader = DataLoader(test_dataset, batch_size, shuffle=True)


model:MLP1 = MLP1().to(device)
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

for epoch in range(num_epochs):
    print(f"epoch {epoch}/{num_epochs}")
    train(model, device, train_loader, optimizer)
eval(model, device, test_loader)


### Полносвязная модель 2.0
dataset_l:numpy.ndarray = trials[:12200]
dataset_l:numpy.ndarray = numpy.delete(dataset_l, numpy.s_[-1], axis=1)

dataset_r:numpy.ndarray = trials[12200:]
dataset_r:numpy.ndarray = numpy.delete(dataset_r, numpy.s_[-1], axis=1)

windows_l:numpy.ndarray = numpy.array(numpy.vsplit(dataset_l, 61))
y_l:numpy.ndarray = numpy.zeros((61, 1))

windows_r:numpy.ndarray = numpy.array(numpy.vsplit(dataset_r, 61))
y_r:numpy.ndarray = numpy.ones((61, 1))

X_l:torch.Tensor = torch.tensor(windows_l, dtype=torch.float32) # 61x200x19
X_r:torch.Tensor = torch.tensor(windows_r, dtype=torch.float32)
y_l:torch.Tensor = torch.tensor(y_l, dtype=torch.float32) # 61x1
y_r:torch.Tensor = torch.tensor(y_r, dtype=torch.float32)

train_dataset:SignalsDataset2 = SignalsDataset2(X_l[:50], X_r[:50], y_l[:50], y_r[:50])
test_dataset:SignalsDataset2 = SignalsDataset2(X_l[50:], X_r[50:], y_l[50:], y_r[50:])

train_loader:DataLoader = DataLoader(train_dataset, batch_size, True)
test_loader:DataLoader = DataLoader(test_dataset, batch_size, True)


model:MLP2 = MLP2().to(device)
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

for epoch in range(num_epochs):
    print(f"epoch {epoch}/{num_epochs}")
    train(model, device, train_loader, optimizer)
eval(model, device, test_loader)


# TODO create several CNN architectures