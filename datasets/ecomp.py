import math
import random

import numpy as np
import pandas as pd
import torch

from datasets.subset import DataSubSet
from utils.loading import load_data
from utils.preprocessing import standardize_df, normalize_df, PERFORMERS
from utils.testing import chunker


def split_time_series(time_steps, sequence_size, sequence_offset, validation_share, test_share, chunk_amounts=5):
    all_window_starts = [i for i in range(0, time_steps - sequence_size, sequence_offset)]
    min_window = sequence_size // sequence_offset

    validation_windows = math.ceil(validation_share * len(all_window_starts))
    validation_windows_per_chunk = validation_windows // chunk_amounts
    test_windows = math.ceil(test_share * len(all_window_starts))
    test_windows_per_chunk = test_windows // chunk_amounts
    windows_per_chunk = validation_windows_per_chunk + test_windows_per_chunk

    side_margin = np.arange(sequence_offset, sequence_offset * min_window, sequence_offset)
    impossible_chunk_starts = set(side_margin)
    impossible_chunk_starts.update(all_window_starts[-1] - side_margin)

    indices = []

    for i in range(chunk_amounts):
        possible_chunk_starts = set(all_window_starts).difference(impossible_chunk_starts)

        chosen_start = random.sample(possible_chunk_starts, 1)[0]
        chosen_i = all_window_starts.index(chosen_start)

        new_impossible_start = (chosen_i - windows_per_chunk + 1) * sequence_offset
        new_impossible_end = (chosen_i + windows_per_chunk) * sequence_offset

        middle = np.arange(new_impossible_start, new_impossible_end, sequence_offset)
        left_margin = new_impossible_start - side_margin - sequence_offset
        right_margin = new_impossible_end + side_margin
        impossible_chunk_starts.update(np.concatenate((middle, left_margin, right_margin), axis=0))

        is_test_chunk_first = bool(random.getrandbits(1))
        if is_test_chunk_first:
            test_end = (chosen_i + test_windows_per_chunk) * sequence_offset
            indices.append((chosen_start, test_end, 'test'))
            indices.append((test_end, new_impossible_end, 'validation'))
        else:
            validation_end = (chosen_i + test_windows_per_chunk) * sequence_offset
            indices.append((chosen_start, validation_end, 'validation'))
            indices.append((validation_end, new_impossible_end, 'test'))

    return indices


class ECompetitionDataSet:
    def __init__(self, validation_share=0.1, test_share=0.1, validation_test_chunks=5,
                 chunk_training=True,
                 output_type='tensor', piece='D960', deviation_type='average', processing=None, sequence_size=50,
                 sequence_offset=10):

        data = load_data('e_competition', piece, deviation_from=deviation_type)

        if processing == 'standardized':
            data = standardize_df(data)
        elif processing == 'normalized':
            data = normalize_df(data)

        train_X = []
        train_y = []
        validation_X = []
        validation_y = []
        test_X = []
        test_y = []

        output_length = len(PERFORMERS)

        for performer in PERFORMERS:
            performer_mask = data['performer'] == performer
            performer_data = data[performer_mask].reset_index(drop=True).drop(columns=['performer'])

            performer_id = PERFORMERS.index(performer)
            if output_type == 'tensor':
                y_true = torch.zeros(output_length)
            else:
                y_true = np.zeros(output_length)
            y_true[performer_id] = 1

            indices = split_time_series(len(performer_data), sequence_size, sequence_offset,
                                        validation_share,
                                        test_share, validation_test_chunks)

            # Sort indices after start index
            indices.sort(key=lambda x: x[0])
            training_starts, training_ends = [0], [len(performer_data)]

            for (start, end, data_type) in indices:
                df = performer_data.iloc[start:end, :]

                # Getting the ids for training data, gaps between sorted test and validation indices
                training_starts.append(end)
                training_ends.insert(-1, start)

                for chunk in chunker(df, sequence_size, sequence_offset):
                    if output_type == 'tensor':
                        X = torch.from_numpy(chunk.to_numpy()).float()
                    elif output_type == 'numpy':
                        X = chunk.to_numpy()
                    else:
                        X = chunk
                    if data_type == 'test':
                        test_X.append(X)
                        test_y.append(y_true)
                    elif data_type == 'validation':
                        validation_X.append(X)
                        validation_y.append(y_true)

            for (start, end) in zip(training_starts, training_ends):
                df = performer_data.iloc[start:end, :]
                if start == end:
                    continue
                if chunk_training:
                    for chunk in chunker(df, sequence_size, sequence_offset):
                        if output_type == 'tensor':
                            X = torch.from_numpy(chunk.to_numpy()).float()
                        else:
                            X = chunk.to_numpy()
                        train_X.append(X)
                        train_y.append(y_true)
                else:
                    df = pd.DataFrame(df)
                    df['performer'] = performer_id
                    train_X.append(df.to_numpy())
        if chunk_training:
            self.train = DataSubSet(train_X, train_y)
        else:
            train_X = np.concatenate(train_X)
            self.train = pd.DataFrame(train_X, columns=data.columns)
        if output_type == 'df':
            self.validation = validation_X, validation_y
            self.test = test_X, test_y
        else:
            self.validation = DataSubSet(validation_X, validation_y)
            self.test = DataSubSet(test_X, test_y)
