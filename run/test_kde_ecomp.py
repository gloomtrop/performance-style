import numpy as np

from datasets.ecomp import ECompetitionDataSet
from models.kde import KDE_classifier
from utils.loading import performers_last_name_list
from utils.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score

# Ignore entropy true divide errors
np.seterr(divide='ignore', invalid='ignore')
bandwidth = 1.2607231727912682
n_samples = 747
weights = [0.5465273761281865, 0.9204239858359122, 0.6225109195455564, 0.6813596379610143, 0.09445053300273365,
           0.3725065684696263, 0.4750341463754865]
dataset = ECompetitionDataSet(chunk_training=False, output_type='df')
performer_ids = [i for i in range(11)]

y_true = []
y_pred = []

model = KDE_classifier(dataset.train, performer_ids, weights=weights, bandwidth=bandwidth, n_samples=n_samples)

for X, y in zip(*dataset.test):
    y_pred.append(model.predict(X))
    y_true.append(y.argmax())

print(f'Accuracy: {accuracy_score(y_true, y_pred)}')
plot_confusion_matrix(y_true, y_pred, performer_ids, performers_last_name_list())
