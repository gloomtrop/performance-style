import pandas as pd
from sklearn.model_selection import KFold


def chunker(seq, size, offset):
    positions = list(range(0, len(seq) - 1, offset))
    filtered = filter(lambda pos: pos + size < seq.shape[0], positions)
    return [seq.iloc[pos:pos + size] for pos in filtered]


def test_classifier(classifier, test_data, chunk_size=50, chunk_offset=25):
    y_true = []
    y_pred = []
    for performer in test_data['performer'].unique():
        performer_mask = test_data['performer'] == performer
        performer_test = test_data[performer_mask].reset_index(drop=True)

        for chunk in chunker(performer_test, chunk_size, chunk_offset):
            prediction = classifier.predict(chunk)
            y_true.append(performer)
            y_pred.append(prediction)

    return y_true, y_pred


def get_kfold_data(data, k, performers):
    fold = KFold(n_splits=k, shuffle=True, random_state=2)
    fold_indices = [list(fold.split(data[data['performer'] == p])) for p in performers]
    test_data = []
    training_data = []

    for f in range(k):
        fold_training_data = pd.DataFrame()
        fold_test_data = []

        for i, p in enumerate(performers):
            performer_data = data[data['performer'] == p]
            train_ids, test_ids = fold_indices[i][f]

            new_training = performer_data.iloc[train_ids]
            new_test = performer_data.iloc[test_ids]

            fold_training_data = pd.concat([fold_training_data, new_training])
            fold_test_data.append(new_test)

        training_data.append(fold_training_data)
        test_data.append(fold_test_data)
    return training_data, test_data


def test_k_fold(clf, test_data, performers):
    y_pred = []
    for i, p in enumerate(performers):
        prediction = clf.predict(test_data[i])
        y_pred.append(prediction)
    return y_pred
