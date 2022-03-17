import pandas as pd

from utils.paths import processed_data_path
from utils.preprocessing import NOTES_FILENAME, DEVIATIONS_FROM_SCORE_FILENAME, DEVIATIONS_FROM_AVERAGE_FILENAME


def load_notes(piece='D960'):
    notes_path = processed_data_path(piece, NOTES_FILENAME)
    return pd.read_json(notes_path)


def load_data(piece='D960', deviation_from='average'):
    if deviation_from == 'average':
        file_name = DEVIATIONS_FROM_AVERAGE_FILENAME
    else:
        file_name = DEVIATIONS_FROM_SCORE_FILENAME
    path = processed_data_path(piece, file_name)
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
