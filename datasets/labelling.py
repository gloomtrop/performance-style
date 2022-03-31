import numpy
import torch
from torch.utils.data import Dataset

from utils.loading import load_labelled_data, load_data


class LabelledDataSet(Dataset):

    def __init__(self, piece='D960', deviation_type='average'):
        data = load_data('labelling', piece, deviation_from=deviation_type)
        labels = load_labelled_data()

        self.x = []
        self.y = []

        x_columns = ['time_onset', 'time_offset', 'velocity_onset',
                     'velocity_offset', 'duration', 'inter_onset_interval',
                     'offset_time_duration']
        y_columns = list(filter(lambda x: 'Question' in x, labels.columns))
        for index, row in labels.iterrows():
            y_np = row[y_columns].to_numpy(dtype=numpy.float32)
            self.y.append(torch.from_numpy(y_np))
            performer_mask = data['performer'].astype(str) == row['performer']
            segment_mask = data['segment'] == row['segment']
            x_df = data[performer_mask & segment_mask]
            x_np = x_df[x_columns].to_numpy(dtype=numpy.float32)
            self.x.append(torch.from_numpy(x_np))

        self.len = len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len
