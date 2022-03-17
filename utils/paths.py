import os
from pathlib import Path


def get_current_directory_path():
    directory_path, filename = os.path.split(__file__)
    return os.path.join(directory_path, "")


def get_root_folder():
    return os.path.join(Path(__file__).parent.parent, '')


def get_files(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def path_from_root(*args):
    return os.path.join(get_root_folder(), *args)


def raw_midi_path(piece, *args):
    return path_from_root('data', 'raw', 'schubert', piece, *args)


def processed_data_path(piece, *args):
    return path_from_root('data', 'processed', 'schubert', piece, *args)


def match_path(piece, *args):
    return path_from_root('data', 'processed', piece, 'match', *args)


def alignment_path(*args):
    return path_from_root('external', 'AlignmentTool', *args)
