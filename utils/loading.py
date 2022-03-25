import pandas as pd
import torch
from torch.utils.data import Dataset, random_split

from utils.paths import processed_data_path, path_from_root
from utils.preprocessing import NOTES_FILENAME, DEVIATIONS_FROM_SCORE_FILENAME, DEVIATIONS_FROM_AVERAGE_FILENAME, \
    standardize_df, normalize_df, PERFORMERS
from utils.testing import chunker

META_FILENAME = 'meta.json'
META_PATH = path_from_root('data', 'raw', 'schubert', META_FILENAME)


class DeviationDataset(Dataset):
    def __init__(self, piece='D960', deviation_type='average', processing=None, sequence_size=50, sequence_offset=10):
        data = load_data(piece, deviation_from=deviation_type)

        if processing == 'standardized':
            data = standardize_df(data)
        elif processing == 'normalized':
            data = normalize_df(data)

        self.x = []
        self.y = []

        output_length = len(PERFORMERS)

        for performer in data['performer'].unique():
            performer_mask = data['performer'] == performer
            performer_data = data[performer_mask].reset_index(drop=True).drop(columns=['performer'])

            performer_id = PERFORMERS.index(performer)
            y_true = torch.zeros(output_length)
            y_true[performer_id] = 1

            for chunk in chunker(performer_data, sequence_size, sequence_offset):
                self.x.append(torch.from_numpy(chunk.to_numpy()).float())
                self.y.append(y_true)
        self.len = len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

    def split(self, ratio=0.9):
        first = int(ratio * self.__len__())
        second = self.__len__() - first
        return random_split(self, [first, second])


def load_model(path):
    saved_dict = torch.load(path)
    model = saved_dict['class'](**saved_dict['input_arguments'])
    model.load_state_dict(saved_dict['state_dict'])
    return model


def load_notes(piece='D960'):
    notes_path = processed_data_path(piece, NOTES_FILENAME)
    return pd.read_json(notes_path)


def load_data(piece='D960', deviation_from='average'):
    if deviation_from == 'average':
        file_name = DEVIATIONS_FROM_AVERAGE_FILENAME
    else:
        file_name = DEVIATIONS_FROM_SCORE_FILENAME
    path = processed_data_path(piece, file_name)
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
