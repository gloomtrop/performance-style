import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from models.kde import KDE_classifier
from utils.loading import load_data
from utils.preprocessing import PERFORMERS
from utils.testing import compute_accuracy

K = 8
WEIGHTS = np.array([0.6194548522595035, 0.15466335764281725, 0.8085611780645111, 0.8700511998978692, 0.188636589856367,
                    0.18516518830581052, 0.7342581057982771])
BANDWIDTH = 0.8232245982435871
N_SAMPLES = 24


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


data = load_data()
y_true = PERFORMERS * K

train, test = get_kfold_data(data, K, PERFORMERS)

y_pred = []
for fold in range(K):
    clf = KDE_classifier(train[fold], PERFORMERS, WEIGHTS, BANDWIDTH, N_SAMPLES)
    y_pred = y_pred + test_k_fold(clf, test[fold])

accuracy = compute_accuracy(y_true, y_pred)
print(accuracy)

cm = confusion_matrix(y_true, y_pred, normalize='true', labels=PERFORMERS)
cm_df = pd.DataFrame(cm, index=PERFORMERS, columns=PERFORMERS)
plt.figure(figsize=(10, 7))
sn.heatmap(cm_df, annot=True)
plt.show()
