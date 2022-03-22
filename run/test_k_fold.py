import numpy as np

from models.kde import KDE_classifier
from utils.loading import load_data, performers_last_name_list
from utils.plotting import plot_confusion_matrix
from utils.preprocessing import PERFORMERS
from utils.testing import compute_accuracy
from utils.testing import get_kfold_data, test_k_fold

K = 8
WEIGHTS = np.array([0.6194548522595035, 0.15466335764281725, 0.8085611780645111, 0.8700511998978692, 0.188636589856367,
                    0.18516518830581052, 0.7342581057982771])
BANDWIDTH = 0.8232245982435871
N_SAMPLES = 24
PERFORMER_NAMES = performers_last_name_list()

data = load_data()
y_true = PERFORMERS * K

train, test = get_kfold_data(data, K, PERFORMERS)

y_pred = []
for fold in range(K):
    clf = KDE_classifier(train[fold], PERFORMERS, WEIGHTS, BANDWIDTH, N_SAMPLES)
    y_pred = y_pred + test_k_fold(clf, test[fold], PERFORMERS)

print(compute_accuracy(y_true, y_pred))
plot_confusion_matrix(y_true, y_pred, PERFORMERS, PERFORMER_NAMES)
