import numpy as np

from models.kde import KDE_classifier
from utils.loading import load_split, performers_last_name_list
from utils.plotting import plot_confusion_matrix
from utils.testing import test_classifier, compute_accuracy

PERFORMERS = [f'p{i}' for i in range(11)]
PERFORMER_NAMES = performers_last_name_list()
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

print(compute_accuracy(y_true, y_pred))
plot_confusion_matrix(y_true, y_pred, PERFORMERS, PERFORMER_NAMES)
