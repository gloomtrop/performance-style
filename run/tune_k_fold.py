import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold

from models.kde import KDE_classifier
from utils.loading import load_data
from utils.preprocessing import NUMBER_COLUMN_NAMES, PERFORMERS
from utils.testing import compute_accuracy


def get_kfold_data(data, k, PERFORMERS, ):
    fold = KFold(n_splits=k, shuffle=True, random_state=2)
    fold_indices = [list(fold.split(data[data['performer'] == p])) for p in PERFORMERS]
    test_data = []
    training_data = []

    for f in range(k):
        fold_training_data = pd.DataFrame()
        fold_test_data = []

        for i, p in enumerate(PERFORMERS):
            performer_data = data[data['performer'] == p]
            train_ids, test_ids = fold_indices[i][f]

            new_training = performer_data.iloc[train_ids]
            new_test = performer_data.iloc[test_ids]

            fold_training_data = pd.concat([fold_training_data, new_training])
            fold_test_data.append(new_test)

        training_data.append(fold_training_data)
        test_data.append(fold_test_data)
    return training_data, test_data


def test_k_fold(clf, test_data):
    y_pred = []
    for i, p in enumerate(PERFORMERS):
        prediction = clf.predict(test_data[i])
        y_pred.append(prediction)
    return y_pred


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
        y_pred = y_pred + test_k_fold(clf, test[fold])
    accuracy = compute_accuracy(y_true, y_pred)
    return 1 - accuracy


study = optuna.create_study()
study.optimize(objective, n_trials=1000)

print(study.best_params)
