import numpy as np
import scipy
from sklearn.neighbors import KernelDensity


class KDE_classifier:

    def __init__(self, training_data, performers, weights=None, bandwidth=0.1, n_samples=100):
        self.performers = performers
        self.bandwidth = bandwidth
        self.n_samples = n_samples
        self.min_value = training_data.min()
        self.max_value = training_data.max()
        self.performer_distributions = self.get_performer_distributions(training_data)
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.ones(len(training_data.columns) - 1)

    def train(self, training_data):
        self.min_value = training_data.min()
        self.max_value = training_data.max()
        self.performer_distributions = self.get_performer_distributions(training_data)

    def predict(self, X):
        sample_distributions = self.get_sample_distributions(X)
        entropy_matrix = self.compute_entropy_matrix(sample_distributions)
        summed_endtropies = self.sum_entropies(entropy_matrix)
        return self.classify_performer(summed_endtropies)

    def compute_entropy_matrix(self, sample_distributions):
        return np.array(
            [[scipy.stats.entropy(sample_distributions[i], pdist[i]) for i in range(len(sample_distributions))] for
             pdist in self.performer_distributions])

    def sum_entropies(self, entropy_matrix):
        return np.dot(entropy_matrix, self.weights)

    def get_sample_distributions(self, df):
        sample_distributions = []
        for column in df.columns:
            if column != 'performer':
                X = df[column].dropna().to_numpy().reshape(-1, 1)
                kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(X)
                x = np.linspace(self.min_value[column], self.max_value[column], self.n_samples)
                log = kde.score_samples(x.reshape(-1, 1))
                sample_distributions.append(np.exp(log))
        return np.array(sample_distributions)

    def get_performer_distributions(self, df):
        ds = []
        for performer in self.performers:
            mask = df['performer'] == performer
            data = df[mask]
            performer_ds = []
            for column in data.columns:
                if column != 'performer':
                    X = data[column].dropna().to_numpy().reshape(-1, 1)
                    kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(X)
                    x = np.linspace(self.min_value[column], self.max_value[column], self.n_samples)
                    log = kde.score_samples(x.reshape(-1, 1))
                    performer_ds.append(np.exp(log))
            ds.append(np.array(performer_ds))
        return np.array(ds)

    def classify_performer(self, entropies):
        index = np.argmin(np.array(entropies))
        return self.performers[index]
