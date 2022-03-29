import torch
from torch.utils.data import Dataset, random_split

from utils.loading import load_data
from utils.preprocessing import standardize_df, normalize_df, PERFORMERS
from utils.testing import chunker


class ECompetitionDataSet(Dataset):
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
