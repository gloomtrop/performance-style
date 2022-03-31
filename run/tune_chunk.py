import numpy as np
import optuna

from models.kde import KDE_classifier
from utils.loading import load_split
from utils.preprocessing import NUMBER_COLUMN_NAMES
from utils.testing import test_classifier, accuracy_score

PERFORMERS = [f'p{i}' for i in range(11)]
CHUNK_SIZE = 100
CHUNK_OFFSET = 25

train, test = load_split()


def objective(trial):
    bandwidth = trial.suggest_float('Bandwidth', 0.01, 1, log=True)
    n_samples = trial.suggest_int('N_samples', 10, 100, log=True)
    weights = np.array([trial.suggest_float(feature, 0, 1) for feature in NUMBER_COLUMN_NAMES])
    cl = KDE_classifier(train, PERFORMERS, weights=weights, bandwidth=bandwidth, n_samples=n_samples)
    y_true, y_pred = test_classifier(cl, test, CHUNK_SIZE, CHUNK_OFFSET)
    accuracy = accuracy_score(y_true, y_pred)
    return 1 - accuracy


study = optuna.create_study()
study.optimize(objective, n_trials=1000)

print(study.best_params)
