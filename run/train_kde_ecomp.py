import numpy as np
import optuna

from datasets.ecomp import ECompetitionDataSet
from models.kde import KDE_classifier
from utils.preprocessing import NUMBER_COLUMN_NAMES
from sklearn.metrics import accuracy_score
from utils.training import get_storage

# Ignore entropy true divide errors
np.seterr(divide='ignore', invalid='ignore')

STUDY_NAME = 'KDE'
N_TRIALS = 1000

dataset = ECompetitionDataSet(chunk_training=False, output_type='df')
performer_ids = [i for i in range(11)]


def objective(trial):
    bandwidth = trial.suggest_float('Bandwidth', 0.001, 10, log=True)
    n_samples = trial.suggest_int('N_samples', 10, 1000, log=True)
    weights = np.array([trial.suggest_float(feature, 0, 1) for feature in NUMBER_COLUMN_NAMES])

    y_true = []
    y_pred = []

    model = KDE_classifier(dataset.train, performer_ids, weights=weights, bandwidth=bandwidth, n_samples=n_samples)

    for X, y in zip(*dataset.validation):
        y_pred.append(model.predict(X))
        y_true.append(y.argmax())

    return accuracy_score(y_true, y_pred)


study = optuna.load_study(study_name=STUDY_NAME, storage=get_storage())
study.optimize(objective, n_trials=N_TRIALS)
