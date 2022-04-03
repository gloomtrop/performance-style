import optuna
import numpy as np

from torch.nn import L1Loss, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import trange

from datasets.labelling import LabelledDataSet
from models.rnn import GRU
from utils.training import get_storage, handle_pruning, save_best_end_model
from utils.training import save_best_intermediary_model, load_best_intermediary_score, \
    load_best_end_score
from sklearn.metrics import r2_score

# Optuna
STUDY_NAME = 'GRU_Labelled'
N_TRIALS = 1000

# Model
MODEL_TYPE = GRU
INPUT_SIZE = 7
NUM_DIMS = 25

# Training
EPOCHS = 100
OPTIMIZER_TYPE = Adam
LOSS_FUNCTIONS = {
    'mse': MSELoss(),
    'mae': L1Loss()
}

# Data
SCALE_DATA = True

# Load previous best scores
best_intermediary_score = load_best_intermediary_score(STUDY_NAME)
best_end_score = load_best_end_score(STUDY_NAME)

# Dataset and loaders
dataset = LabelledDataSet(scaled=SCALE_DATA)
train_loader = DataLoader(dataset.train, shuffle=True)
validation_loader = DataLoader(dataset.validation, shuffle=False)

y_true_validation = [outputs.detach().numpy().squeeze() for _, outputs in validation_loader]


def objective(trial):
    global best_end_score
    global best_intermediary_score
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    hidden_size = trial.suggest_int('hidden size', 50, 150)
    num_layers = trial.suggest_int('num layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0, 1)
    loss_function = trial.suggest_categorical('loss_function', LOSS_FUNCTIONS.keys())
    criterion = LOSS_FUNCTIONS[loss_function]

    model = MODEL_TYPE(INPUT_SIZE, hidden_size, num_layers, NUM_DIMS, dropout)
    optimizer = OPTIMIZER_TYPE(params=model.parameters(), lr=lr)

    r2 = 0
    for epoch in trange(EPOCHS):
        model.train()
        for i, data in enumerate(train_loader):
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

        y_pred_validation = [model(inputs).detach().numpy().squeeze() for inputs, _ in validation_loader]
        r2 = r2_score(y_true_validation, y_pred_validation, multioutput='variance_weighted')

        if r2 > best_intermediary_score:
            best_intermediary_score = r2
            save_best_intermediary_model(STUDY_NAME, model, r2)

        handle_pruning(trial, r2, epoch)

    if r2 > best_end_score:
        best_end_score = r2
        save_best_end_model(STUDY_NAME, model, r2)

    return r2


study = optuna.load_study(study_name=STUDY_NAME, storage=get_storage())
study.optimize(objective, n_trials=N_TRIALS)
