import os

import torch
from dotenv import load_dotenv
from optuna.exceptions import TrialPruned
from optuna_dashboard import run_server

from utils.paths import path_from_root


def get_storage():
    load_dotenv()
    storage = os.getenv('database_url')
    if not storage:
        raise ValueError('You have not set database_url as an environment variable')
    else:
        return storage


def start_dashboard(host: str = 'localhost', port: int = 8080):
    run_server(get_storage(), host=host, port=port)


def handle_pruning(trial, objective, epoch):
    trial.report(objective, epoch)

    if trial.should_prune():
        raise TrialPruned()


def intermediary_file_name(study):
    return f'{study}_int.pkl'


def end_file_name(study):
    return f'{study}_end.pkl'


def load_model_score(file_name):
    best_save_path = path_from_root('models', 'saved', file_name)
    saved_dict = torch.load(best_save_path)
    return saved_dict['score']


def load_best_intermediary_score(study_name):
    try:
        return load_model_score(intermediary_file_name(study_name))
    except FileNotFoundError:
        return 0


def load_best_end_score(study_name):
    try:
        return load_model_score(end_file_name(study_name))
    except FileNotFoundError:
        return 0


def save_best_intermediary_model(study, model, score):
    save_path = path_from_root('models', 'saved', intermediary_file_name(study))
    model.save(save_path, score=score)


def save_best_end_model(study, model, score):
    save_path = path_from_root('models', 'saved', end_file_name(study))
    model.save(save_path, score=score)
