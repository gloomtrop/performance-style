from models.statistical import KDE_classifier
from models.data_loader import load_data

PERFORMERS = [f'p{i}' for i in range(11)]

bandwidth = 0.1
n_samples = 100

train, test = load_data()

classifier = KDE_classifier(train, PERFORMERS, bandwidth, n_samples)

sample = test[test['performer'] == PERFORMERS[5]]

print(classifier.predict(sample))