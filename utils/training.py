import os

from dotenv import load_dotenv
from optuna.exceptions import TrialPruned
from optuna_dashboard import run_server


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
