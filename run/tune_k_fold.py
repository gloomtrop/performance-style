import numpy as np
import optuna

from models.kde import KDE_classifier
from utils.loading import load_data
from utils.preprocessing import NUMBER_COLUMN_NAMES, PERFORMERS
from utils.testing import compute_accuracy
from utils.testing import get_kfold_data, test_k_fold

K = 8
data = load_data()
y_true = PERFORMERS * K

train, test = get_kfold_data(data, K, PERFORMERS)


def objective(trial):
    bandwidth = trial.suggest_float('Bandwidth', 0.01, 1, log=True)
    n_samples = trial.suggest_int('N_samples', 10, 100, log=True)
    weights = np.array([trial.suggest_float(feature, 0, 1) for feature in NUMBER_COLUMN_NAMES])
    y_pred = []
    for fold in range(K):
        clf = KDE_classifier(train[fold], PERFORMERS, weights, bandwidth, n_samples)
        y_pred = y_pred + test_k_fold(clf, test[fold], PERFORMERS)
    accuracy = compute_accuracy(y_true, y_pred)
    return 1 - accuracy


study = optuna.create_study()
study.optimize(objective, n_trials=1000)

print(study.best_params)
