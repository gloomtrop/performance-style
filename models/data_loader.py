from utils import paths
import os
import pandas as pd
from utils.preprocessing import NUMBER_COLUMN_NAMES, STD_NUMBER_COLUMN_NAMES


def load_data(piece='D960', filter='unstd'):
    deviations_path = os.path.join(paths.get_root_folder(), 'processed data', piece, 'deviations.json')
    data = pd.read_json(deviations_path)

    # Filtering
    if filter == 'std':
        columns = STD_NUMBER_COLUMN_NAMES + ['performer']
    elif filter == 'unstd':
        columns = NUMBER_COLUMN_NAMES + ['performer']
    else:
        columns = NUMBER_COLUMN_NAMES + STD_NUMBER_COLUMN_NAMES + ['performer']

    return data[columns]


def load_split(piece='D960', split=0.8, filter='unstd'):
    training_data = pd.DataFrame()
    test_data = pd.DataFrame()

    data = load_data(piece, filter)

    for performer in data['performer'].unique():
        performer_mask = data['performer'] == performer
        performer_data = data[performer_mask]
        length = int(performer_data.shape[0] * split)
        training_data = pd.concat([training_data, performer_data[:length]])
        test_data = pd.concat([test_data, performer_data[length:]])

    return training_data, test_data
