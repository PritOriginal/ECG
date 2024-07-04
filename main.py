import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os.path
from pathlib import Path

# df = pd.read_csv('ECGLog-2024-7-1-12-51-11.txt', sep='\t')
# print(df)
# print(df['timestamp'])
# plt.plot(df['timestamp'].to_numpy(), df['HeartRate4sAverage'].to_numpy())
# plt.show()
file = "ECGLog-2024-7-1-14-51-1"
if not os.path.exists(file + '.csv'):
    with open(file + '.txt', 'r') as f:
        rows = f.read().split('\n')
        data = pd.DataFrame(columns=rows[0].split())
        rows = rows[1:]
        for row in rows:
            if row:
                data_row = row.split()
                data_row[0] = float(data_row[0][:-1].replace(',', '.'))
                data.loc[len(data)] = data_row
        # print(data)
        data.to_csv(file + '.csv')
else:
    data = pd.read_csv(file + '.csv')

time_start = data.iloc[0,1]
data['timestamp:'] -= time_start

old_time = np.arange(len(data))

all_time = np.array([])
time_0 = data['timestamp:'][0]
time_step = 0

i = 1
for index in range(len(data['timestamp:'])):
    time = data['timestamp:'][index]
    if time != time_0 or index == len(data['timestamp:']) - 1:
        all_time = np.hstack([all_time, np.linspace(time_0, time, i)])
        time_0 = time
        i = 1
    else:
        i += 1

fig, ax = plt.subplots(2, 1)

# data.drop_duplicates(subset=['timestamp:'])

ax[0].plot(all_time, data['HeartRate4sAverage'].to_numpy(dtype=int))
ax[0].plot(all_time, data['HeartRate30sAverage'].to_numpy(dtype=int))
ax[0].set_xlabel('Time')
ax[0].set_ylabel('HeartRate')
ax[0].grid(True)

ax[1].plot(all_time, data['ADC'].to_numpy(dtype=int))
ax[1].set_xlabel('Time')
ax[1].set_ylabel('ADC')
ax[1].grid(True)

plt.show()