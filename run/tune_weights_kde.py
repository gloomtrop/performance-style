import numpy as np
import optuna
from models.kde import KDE_classifier
from utils.loading import load_split
from utils.testing import test_classifier, compute_accuracy

PERFORMERS = [f'p{i}' for i in range(11)]

bandwidth = 0.1
n_samples = 100

train, test = load_split()
cl = KDE_classifier(train, PERFORMERS, bandwidth=bandwidth, n_samples=n_samples)


def objective(trial):
    # Variables
    x1 = trial.suggest_float('Note Onset', 0, 1)
    x2 = trial.suggest_float('Note Offset', 0, 1)
    x3 = trial.suggest_float('Velocity Onset', 0, 1)
    x4 = trial.suggest_float('Velocity Offset', 0, 1)
    x5 = trial.suggest_float('Duration', 0, 1)
    x6 = trial.suggest_float('Inter Onset Interval', 0, 1)
    x7 = trial.suggest_float('Onset Time Duration', 0, 1)

    cl.weights = np.array([x1, x2, x3, x4, x5, x6, x7])
    y_true, y_pred = test_classifier(cl, test, 100, 25)
    accuracy = compute_accuracy(y_true, y_pred)
    return 1 - accuracy


study = optuna.create_study()
study.optimize(objective, n_trials=1000)

print(study.best_params)
