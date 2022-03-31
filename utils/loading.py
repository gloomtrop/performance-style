import pandas as pd
import torch

from utils.paths import processed_path, path_from_root, processed_labelled_path
from utils.preprocessing import NOTES_FILENAME, DEVIATIONS_FROM_SCORE_FILENAME, DEVIATIONS_FROM_AVERAGE_FILENAME

META_FILENAME = 'meta.json'
META_PATH = path_from_root('data', 'raw', 'e_competition', META_FILENAME)


def load_labelled_data():
    labels_path = processed_labelled_path('data.json')
    return pd.read_json(labels_path)


def load_model(path):
    saved_dict = torch.load(path)
    model = saved_dict['class'](**saved_dict['input_arguments'])
    model.load_state_dict(saved_dict['state_dict'])
    return model


def load_notes(dataset='e_competition', piece='D960'):
    notes_path = processed_path(dataset, piece, NOTES_FILENAME)
    return pd.read_json(notes_path)


def load_data(dataset='e_competition', piece='D960', deviation_from='average'):
    if deviation_from == 'average':
        file_name = DEVIATIONS_FROM_AVERAGE_FILENAME
    else:
        file_name = DEVIATIONS_FROM_SCORE_FILENAME
    path = processed_path(dataset, piece, file_name)
    return pd.read_json(path).dropna()


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


def load_performers(piece='D960'):
    performers = pd.read_json(META_PATH)
    piece_mask = performers['piece'] == piece
    return performers[piece_mask]


def performers_full_name_list(piece='D960'):
    perf_df = load_performers(piece)
    return list(perf_df['name'].to_numpy())


def performers_last_name_list(piece='D960'):
    perf_full = performers_full_name_list(piece)
    return [name.split()[-1] for name in perf_full]
