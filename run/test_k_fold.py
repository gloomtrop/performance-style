import numpy as np
import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from models.kde import KDE_classifier
from utils.loading import load_data, performers_last_name_list
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

cm = confusion_matrix(y_true, y_pred, normalize='true', labels=PERFORMERS)
cm_df = pd.DataFrame(cm, index=PERFORMER_NAMES, columns=PERFORMER_NAMES)
plt.figure(figsize=(10, 7))
sn.heatmap(cm_df, annot=True)
plt.show()
