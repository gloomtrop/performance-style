import optuna
import torch
from torch.nn import L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import trange

from datasets.labelling import LabelledDataSet
from models.rnn import GRU
from utils.training import get_storage, handle_pruning, save_best_end_model
from utils.training import save_best_intermediary_model, load_best_intermediary_score, \
    load_best_end_score

# Optuna
STUDY_NAME = 'GRU_Labelled_L1'
N_TRIALS = 100

# Model
MODEL_TYPE = GRU
INPUT_SIZE = 7
NUM_DIMS = 25

# Training
EPOCHS = 100
CRITERION = L1Loss()
OPTIMIZER_TYPE = Adam

# Load previous best scores
best_intermediary_score = load_best_intermediary_score(STUDY_NAME, maximize=False)
best_end_score = load_best_end_score(STUDY_NAME, maximize=False)

# Dataset and loaders
dataset = LabelledDataSet()
train_loader = DataLoader(dataset.train, shuffle=True)
validation_loader = DataLoader(dataset.validation, shuffle=False)


def objective(trial):
    global best_end_score
    global best_intermediary_score
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    hidden_size = trial.suggest_int('hidden size', 10, 150)
    num_layers = trial.suggest_int('num layers', 1, 3)

    model = MODEL_TYPE(INPUT_SIZE, hidden_size, num_layers, NUM_DIMS)
    optimizer = OPTIMIZER_TYPE(params=model.parameters(), lr=lr)

    avg_loss = 0
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
        total_loss = 0
        for i, data in enumerate(validation_loader):
            inputs, labels = data
            outputs = model(inputs)
            total_loss += CRITERION(outputs, labels)
        avg_loss = total_loss / len(validation_loader)

        if avg_loss < best_intermediary_score:
            best_intermediary_score = avg_loss
            save_best_intermediary_model(STUDY_NAME, model, avg_loss)

        handle_pruning(trial, avg_loss, epoch)

    if avg_loss < best_end_score:
        best_end_score = avg_loss
        save_best_end_model(STUDY_NAME, model, avg_loss)

    return avg_loss


study = optuna.load_study(study_name=STUDY_NAME, storage=get_storage())
study.optimize(objective, n_trials=N_TRIALS)
