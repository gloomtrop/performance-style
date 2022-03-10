import os
from utils import paths
import pandas as pd

columns = ['time_onset', 'time_offset', 'velocity_onset', 'velocity_offset','duration']
piece = 'D960'
notes_path = os.path.join(paths.get_root_folder(), 'processed data', piece, 'notes')
deviation_path = os.path.join(paths.get_root_folder(), 'processed data', piece, 'deviations')

performance_names = paths.get_files(notes_path)

avg_filepath = os.path.join(notes_path, 'avg.json')
avg = pd.read_json(avg_filepath)
avg_columns = [name + '_avg' for name in columns]
std_columns = [name + '_std' for name in columns]

for performance_name in performance_names:
    if performance_name != 'avg.json':
        performance_path = os.path.join(notes_path, performance_name)
        df = pd.read_json(performance_path)
        deviations = pd.DataFrame()
        for i, column in enumerate(columns):
            deviations[column] = df[column] - avg[avg_columns[i]]
            deviations[column+'_standardized'] = deviations[column] / avg[std_columns[i]]

        # Filter out values that are present in avg, but not in the given performance
        mask = deviations.index.isin(df.index)
        filtered_deviations = deviations[mask]

        deviations_filepath = os.path.join(deviation_path, performance_name)
        filtered_deviations.to_json(deviations_filepath)
