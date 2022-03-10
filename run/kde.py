from sklearn.neighbors import KernelDensity
from utils import paths
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

PERFORMERS = [f'p{i}' for i in range(11)]
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


def compute_entropies(sample_distributions, performer_distributions):
    return np.array(
        [sum([scipy.stats.entropy(sample_distributions[i], pdist[i]) for i in range(len(sample_distributions))]) for
         pdist in performer_distributions])


def get_sample_distributions(df, bandwidth, min_value, max_value, n_samples):
    sample_distributions = []
    for column in df.columns:
        if column != 'performer':
            X = df[column].dropna().to_numpy().reshape(-1, 1)
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
            x = np.linspace(min_value[column], max_value[column], n_samples)
            log = kde.score_samples(x.reshape(-1, 1))
            sample_distributions.append(np.exp(log))
    return np.array(sample_distributions)


def get_performer_distributions(df, bandwidth, min_value, max_value, n_samples):
    ds = []
    for performer in PERFORMERS:
        mask = df['performer'] == performer
        data = df[mask]
        performer_ds = []
        for column in data.columns:
            if column != 'performer':
                X = data[column].dropna().to_numpy().reshape(-1, 1)
                kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(X)
                x = np.linspace(min_value[column], max_value[column], n_samples)
                log = kde.score_samples(x.reshape(-1, 1))
                performer_ds.append(np.exp(log))
        ds.append(np.array(performer_ds))
    return np.array(ds)


def classify_performer(entropies):
    index = np.argmin(np.array(entropies))
    return PERFORMERS[index]


bandwidth = 0.1
n_samples = 100

train, test = load_data()

min_value = train.min()
max_value = train.max()

pdist = get_performer_distributions(train, bandwidth, min_value, max_value, n_samples)

sample = test[test['performer'] == PERFORMERS[5]]

sdist = get_sample_distributions(sample, bandwidth, min_value, max_value, n_samples)

entropies = compute_entropies(sdist, pdist)

print(entropies)
print(classify_performer(entropies))