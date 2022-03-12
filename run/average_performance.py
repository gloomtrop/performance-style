import os
from utils import paths
import pandas as pd


piece = 'D960'
path = os.path.join(paths.get_root_folder(), 'processed data', piece, 'notes')

file_names = paths.get_files(path)

all_notes = pd.DataFrame()
for performance_name in file_names:
    if performance_name != 'avg.json':
        performance_path = os.path.join(path, performance_name)
        df = pd.read_json(performance_path)
        all_notes = pd.concat([all_notes, df])

averages = all_notes.groupby(by=all_notes.index).mean()
stds = all_notes.groupby(by=all_notes.index).std()

columns = ['time_onset', 'time_offset', 'velocity_onset', 'velocity_offset', 'duration', 'inter_onset_interval', 'offset_time_duration']
data = averages[columns].join(stds[columns], lsuffix='_avg', rsuffix='_std')

file_path = os.path.join(path, 'avg.json')

data.to_json(file_path)