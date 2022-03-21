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
bandwidth = 0.6932785319057954
n_samples = 56
weights = np.array(
    [0.9803701032803395, 0.06889461912880954, 0.07764404791412743, 0.3875642462285485, 0.08259937926241562,
     0.6964800043141498, 0.7700041725975063])

train, test = load_split()
cl = KDE_classifier(train, PERFORMERS, weights=weights, bandwidth=bandwidth, n_samples=n_samples)

y_true, y_pred = test_classifier(cl, test, SAMPLE_SIZE, SAMPLE_OFFSET)

accuracy = compute_accuracy(y_true, y_pred)
print(accuracy)

cm = confusion_matrix(y_true, y_pred, normalize='true', labels=PERFORMERS)
cm_df = pd.DataFrame(cm, index=PERFORMERS, columns=PERFORMERS)
plt.figure(figsize=(10, 7))
sn.heatmap(cm_df, annot=True)
plt.show()
