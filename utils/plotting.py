import pandas as pd
import seaborn as sn
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

cmap = 'Blues'


def plot_confusion_matrix(y_true, y_pred, classification_labels, display_labels, normalize='true'):
    cm = confusion_matrix(y_true, y_pred, normalize=normalize, labels=classification_labels)
    cm_df = pd.DataFrame(cm, index=display_labels, columns=display_labels)
    plt.figure(figsize=(10, 7))
    sn.heatmap(cm_df, annot=True, cmap=cmap)
    plt.show()
