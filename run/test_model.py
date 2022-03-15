import numpy as np

from models.data_loader import load_split
from models.statistical import KDE_classifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from utils.testing import test_classifier, compute_accuracy

PERFORMERS = [f'p{i}' for i in range(11)]

bandwidth = 0.1
n_samples = 100

train, test = load_split()
cl = KDE_classifier(train, PERFORMERS, bandwidth, n_samples)

y_true, y_pred = test_classifier(cl, test)

accuracy = compute_accuracy(y_true, y_pred)
print(accuracy)

cm = confusion_matrix(y_true, y_pred, normalize='true', labels=PERFORMERS)
cm_df = pd.DataFrame(cm, index=PERFORMERS, columns=PERFORMERS)
plt.figure(figsize=(10, 7))
sn.heatmap(cm_df, annot=True)
plt.show()
