def compute_accuracy(y_true, y_pred):
    equal = [t == p for t, p in zip(y_true, y_pred)]
    return sum(equal) / len(equal)


def chunker(seq, size, offset):
    positions = list(range(0, len(seq) - 1, offset))
    filtered = filter(lambda pos: pos + size < seq.shape[0], positions)
    return (seq.iloc[pos:pos + size] for pos in filtered)


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
