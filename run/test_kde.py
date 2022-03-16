import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix

from models.kde import KDE_classifier
from utils.loading import load_split
from utils.testing import test_classifier, compute_accuracy

PERFORMERS = [f'p{i}' for i in range(11)]
SAMPLE_SIZE = 100
SAMPLE_OFFSET = 25
bandwidth = 0.1
n_samples = 100
weights = np.array([0.0001, 0.0004, 0.2565, 0.1705, 0.2924, 0.9177, 0.4494])
weights_2 = np.array([0.0005, 0.0006, 0.1743, 0.1182, 0.2179, 0.7752, 0.7750])
weights_3 = np.array([0.0008, 0.0001, 0.2954, 0.2238, 0.5822, 0.8793, 0.6488])

train, test = load_split()
cl = KDE_classifier(train, PERFORMERS, weights=weights_3, bandwidth=bandwidth, n_samples=n_samples)

y_true, y_pred = test_classifier(cl, test, SAMPLE_SIZE, SAMPLE_OFFSET)

accuracy = compute_accuracy(y_true, y_pred)
print(accuracy)

cm = confusion_matrix(y_true, y_pred, normalize='true', labels=PERFORMERS)
cm_df = pd.DataFrame(cm, index=PERFORMERS, columns=PERFORMERS)
plt.figure(figsize=(10, 7))
sn.heatmap(cm_df, annot=True)
plt.show()
