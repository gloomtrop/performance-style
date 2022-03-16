from tpot import TPOTClassifier

from utils.loading import load_split
from utils.preprocessing import transform_data

train, test = load_split()
X_train, y_train = transform_data(train.drop(columns=['time_onset', 'time_offset']), 20)
X_test, y_test = transform_data(test.drop(columns=['time_onset', 'time_offset']), 20)

tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_classifier_.py')
