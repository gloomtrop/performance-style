import optuna
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import trange

from models.rnn import RNN
from utils.loading import DeviationDataset
from utils.testing import compute_accuracy
from utils.training import get_storage, handle_pruning

INPUT_SIZE = 7

NUM_CLASSES = 11
SEQUENCE_LENGTH = 50
SEQUENCE_OFFSET = 10
BATCH_SIZE = 25
EPOCHS = 100

criterion = CrossEntropyLoss()

dataset = DeviationDataset()
train, val = dataset.split()
trainloader = DataLoader(train, batch_size=BATCH_SIZE,
                         shuffle=True)
valloader = DataLoader(val, batch_size=BATCH_SIZE,
                       shuffle=False)


def objective(trial):
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    hidden_size = trial.suggest_int('hidden size', 70, 120, log=True)
    num_layers = trial.suggest_int('num layers', 1, 4)

    model = RNN(INPUT_SIZE, hidden_size, num_layers, NUM_CLASSES)

    optimizer = Adam(params=model.parameters(), lr=lr)

    accuracy = 0
    for epoch in trange(EPOCHS):
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
        handle_pruning(trial, accuracy, epoch)

    return accuracy


study = optuna.load_study(study_name='RNN', storage=get_storage())
study.optimize(objective, n_trials=100)
