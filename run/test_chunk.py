import numpy as np

from models.kde import KDE_classifier
from utils.loading import load_split, performers_last_name_list
from utils.plotting import plot_confusion_matrix
from utils.preprocessing import normalize_df, standardize_df
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

norm_train, min_values, max_values = normalize_df(train)
norm_test, _, _ = normalize_df(test, min_values, max_values)

standardized_train, means, stds = standardize_df(train)
standardized_test, _, _ = normalize_df(test, means, stds)

cl = KDE_classifier(train, PERFORMERS, bandwidth=bandwidth, n_samples=n_samples)
cln = KDE_classifier(norm_train, PERFORMERS, bandwidth=bandwidth, n_samples=n_samples)
cls = KDE_classifier(standardized_train, PERFORMERS, bandwidth=bandwidth, n_samples=n_samples)

y_true, y_pred = test_classifier(cl, test, SAMPLE_SIZE, SAMPLE_OFFSET)
y_true_n, y_pred_n = test_classifier(cln, norm_test, SAMPLE_SIZE, SAMPLE_OFFSET)
y_true_s, y_pred_s = test_classifier(cls, standardized_test, SAMPLE_SIZE, SAMPLE_OFFSET)

print(f'Raw values: {compute_accuracy(y_true, y_pred)}')
print(f'Normalized values: {compute_accuracy(y_true_n, y_pred_n)}')
print(f'Standardized values: {compute_accuracy(y_true_s, y_pred_s)}')

plot_confusion_matrix(y_true, y_pred, PERFORMERS, PERFORMER_NAMES)
plot_confusion_matrix(y_true_n, y_pred_n, PERFORMERS, PERFORMER_NAMES)
plot_confusion_matrix(y_true_s, y_pred_s, PERFORMERS, PERFORMER_NAMES)
