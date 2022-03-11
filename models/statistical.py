from sklearn.neighbors import KernelDensity
import numpy as np
import scipy


class KDE_classifier:

    def __init__(self, training_data, performers, bandwidth=0.1, n_samples=100):
        self.performers = performers
        self.bandwidth = bandwidth
        self.n_samples = n_samples
        self.min_value = training_data.min()
        self.max_value = training_data.max()
        self.performer_distributions = self.get_performer_distributions(training_data)

    def predict(self, X):
        sample_distributions = self.get_sample_distributions(X)
        entropies = self.compute_entropy_summed(sample_distributions)
        return self.classify_performer(entropies)

    def compute_entropy_matrix(self, sample_distributions):
        return np.array(
            [[scipy.stats.entropy(sample_distributions[i], pdist[i]) for i in range(len(sample_distributions))] for
             pdist in self.performer_distributions])

    def compute_entropy_summed(self, sample_distributions):
        return np.array(
            [sum([scipy.stats.entropy(sample_distributions[i], pdist[i]) for i in range(len(sample_distributions))]) for
             pdist in self.performer_distributions])

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
