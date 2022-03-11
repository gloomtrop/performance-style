from utils import paths
import os
import pandas as pd

COLUMNS = ['time_onset', 'time_offset', 'velocity_onset', 'velocity_offset', 'duration']
STD_COLUMNS = [name + '_standardized' for name in COLUMNS]


def load_data(piece='D960', split=0.8, filter='std'):
    training_data = pd.DataFrame()
    test_data = pd.DataFrame()
    deviations_path = os.path.join(paths.get_root_folder(), 'processed data', piece, 'deviations')
    deviations_names = paths.get_files(deviations_path)
    for deviation in deviations_names:
        data_path = os.path.join(deviations_path, deviation)
        performer = deviation.split('-')[0]
        data = pd.read_json(data_path)
        length = int(data.shape[0] * split)
        data['performer'] = performer
        training_data = pd.concat([training_data, data[:length]])
        test_data = pd.concat([test_data, data[length:]])

    # Filtering
    if filter == 'std':
        columns = STD_COLUMNS + ['performer']
    elif filter == 'unstd':
        columns = STD_COLUMNS + ['performer']
    else:
        columns = COLUMNS + STD_COLUMNS + ['performer']

    return training_data[columns], test_data[columns]
