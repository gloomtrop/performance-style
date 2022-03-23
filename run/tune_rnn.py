import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

import optuna
from models.rnn import RNN
from utils.loading import DeviationDataset
from utils.testing import compute_accuracy

INPUT_SIZE = 7

NUM_CLASSES = 11
SEQUENCE_LENGTH = 50
SEQUENCE_OFFSET = 10
BATCH_SIZE = 25
EPOCHS = 30

criterion = CrossEntropyLoss()

dataset = DeviationDataset()
train, val = dataset.split()
trainloader = DataLoader(train, batch_size=BATCH_SIZE,
                         shuffle=True)
valloader = DataLoader(val, batch_size=BATCH_SIZE,
                       shuffle=False)


def objective(trial):
    lr = trial.suggest_float('lr', 0.001, 0.1, log=True)
    hidden_size = trial.suggest_int('hidden size', 10, 100, log=True)
    num_layers = trial.suggest_int('num layers', 1, 4)

    model = RNN(INPUT_SIZE, hidden_size, num_layers, NUM_CLASSES)

    optimizer = Adam(params=model.parameters(), lr=lr)

    accuracy = 0
    for epoch in range(EPOCHS):
        model.train()
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        y_true = []
        y_pred = []
        for i, data in enumerate(valloader):
            inputs, label = data
            outputs = model(inputs)
            y_true += torch.argmax(label, dim=1)
            y_pred += torch.argmax(outputs, dim=1)

        accuracy = float(compute_accuracy(y_true, y_pred))

        trial.report(accuracy, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return accuracy


db_name = "sqlite:///db.sqlite3"

storage = optuna.storages.RDBStorage(
    url=db_name,
)

study = optuna.load_study(study_name='RNN', storage=storage)
study.optimize(objective, n_trials=100)
