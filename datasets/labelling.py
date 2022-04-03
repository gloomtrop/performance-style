import numpy
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from utils.loading import load_labelled_data, load_data
from datasets.subset import DataSubSet

RANDOM_STATE = 42
MAX_NUM = 7
MIN_NUM = 1


class LabelledDataSet(Dataset):

    def __init__(self, validation_size=0.1, test_size=0.1, piece='D960', deviation_type='average', scaled=False):
        data = load_data('labelling', piece, deviation_from=deviation_type)
        labels = load_labelled_data()

        test_validation_size = validation_size + test_size
        test_size_of_test_validation = test_size / test_validation_size

        x = []
        y = []

        x_columns = ['time_onset', 'time_offset', 'velocity_onset',
                     'velocity_offset', 'duration', 'inter_onset_interval',
                     'offset_time_duration']
        y_columns = list(filter(lambda x: 'Question' in x, labels.columns))

        for index, row in labels.iterrows():
            y_np = row[y_columns].to_numpy(dtype=numpy.float32)
            # Scaling from -1 to 1
            if scaled:
                y_np = (y_np - MIN_NUM) / (0.5 * (MAX_NUM - MIN_NUM)) - 1
            y.append(torch.from_numpy(y_np))
            performer_mask = data['performer'].astype(str) == row['performer']
            segment_mask = data['segment'] == row['segment']
            x_df = data[performer_mask & segment_mask]
            x_np = x_df[x_columns].to_numpy(dtype=numpy.float32)
            x.append(torch.from_numpy(x_np))
        X_train, X_val_test, y_train, y_val_test = train_test_split(x, y, test_size=test_validation_size,
                                                                    random_state=RANDOM_STATE)
        X_validation, X_test, y_validation, y_test = train_test_split(X_val_test, y_val_test,
                                                                      test_size=test_size_of_test_validation,
                                                                      random_state=RANDOM_STATE)

        self.train = DataSubSet(X_train, y_train)
        self.validation = DataSubSet(X_validation, y_validation)
        self.test = DataSubSet(X_test, y_test)
        self.all = DataSubSet(x, y)
