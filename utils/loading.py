import os

import pandas as pd

from utils.paths import get_root_folder


def load_notes(piece='D960'):
    notes_path = os.path.join(get_root_folder(), 'data', 'processed', piece, 'notes.json')
    return pd.read_json(notes_path)


def load_data(piece='D960', deviation_from='average'):
    if deviation_from == 'average':
        file_name = 'deviations_from_average.json'
    else:
        file_name = 'deviations_from_score.json'
    path = os.path.join(get_root_folder(), 'data', 'processed', piece, file_name)
    return pd.read_json(path)


def load_split(piece='D960', split=0.8, deviation_from='average'):
    training_data = pd.DataFrame()
    test_data = pd.DataFrame()

    data = load_data(piece, deviation_from)

    for performer in data['performer'].unique():
        performer_mask = data['performer'] == performer
        performer_data = data[performer_mask]
        length = int(performer_data.shape[0] * split)
        training_data = pd.concat([training_data, performer_data[:length]])
        test_data = pd.concat([test_data, performer_data[length:]])

    return training_data, test_data
