import optuna
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import RAdam
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange

from datasets.ecomp import ECompetitionDataSet
from models.transformer import TransformerEncoder 
from utils.testing import compute_accuracy
from utils.training import get_storage, handle_pruning, save_best_end_model
from utils.training import save_best_intermediary_model, load_best_intermediary_score, \
    load_best_end_score

# Optuna
STUDY_NAME = 'TransformerEncoder'
N_TRIALS = 100

# Model
MODEL_TYPE = TransformerEncoder
INPUT_SIZE = 7
NUM_CLASSES = 11
SEQUENCE_LENGTH = 50
SEQUENCE_OFFSET = 10

# Training
BATCH_SIZE = 25
EPOCHS = 100
CRITERION = CrossEntropyLoss()
OPTIMIZER_TYPE = RAdam

# Load previous best scores
best_intermediary_score = load_best_intermediary_score(STUDY_NAME)
best_end_score = load_best_end_score(STUDY_NAME)

# Dataset and loaders
dataset = ECompetitionDataSet()
train_loader = DataLoader(dataset.train, batch_size=BATCH_SIZE,
                          shuffle=True)
validation_loader = DataLoader(dataset.validation, batch_size=BATCH_SIZE,
                               shuffle=False)


def objective(trial):
    global best_end_score
    global best_intermediary_score
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    hidden_size = trial.suggest_int('hidden size', 20, 50, step = 2)
    num_layers = trial.suggest_int('num layers', 1, 4)

    model = MODEL_TYPE(num_layers, SEQUENCE_LENGTH, input_dim = SEQUENCE_LENGTH, num_heads = 1
                        , dim_feedforward = hidden_size 
                        , output_size = NUM_CLASSES ,dropout=0.0)
    optimizer = OPTIMIZER_TYPE(params=model.parameters(), lr=lr)

    accuracy = 0
    for epoch in trange(EPOCHS):
        model.train()
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = CRITERION(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        y_true = []
        y_pred = []
        for i, data in enumerate(validation_loader):
            inputs, label = data
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim = 1)
            y_true += torch.argmax(label, dim=1)
            y_pred += torch.argmax(outputs, dim=1)

        accuracy = float(compute_accuracy(y_true, y_pred))

        if accuracy > best_intermediary_score:
            best_intermediary_score = accuracy
            save_best_intermediary_model(STUDY_NAME, model, accuracy)

        handle_pruning(trial, accuracy, epoch)

    if accuracy > best_end_score:
        best_end_score = accuracy
        save_best_end_model(STUDY_NAME, model, accuracy)

    return accuracy


study = optuna.load_study(study_name=STUDY_NAME, storage=get_storage())
study.optimize(objective, n_trials=N_TRIALS)
