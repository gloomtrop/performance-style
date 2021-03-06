import numpy as np
import pandas as pd

from utils.paths import match_path, get_files
from utils.testing import chunker

MATCH_COLUMN_NAMES = ['id', 'time_onset', 'time_offset', 'pitch', 'velocity_onset', 'velocity_offset', 'channel',
                      'match_status', 'time_score', 'note_id', 'error_index', 'skip_index']

NUMBER_COLUMN_NAMES = ['time_onset', 'time_offset', 'velocity_onset', 'velocity_offset', 'duration',
                       'inter_onset_interval',
                       'offset_time_duration']

AVERAGE_NUMBER_COLUMN_NAMES = [name + '_avg' for name in NUMBER_COLUMN_NAMES]
STD_NUMBER_COLUMN_NAMES = [name + '_std' for name in NUMBER_COLUMN_NAMES]

NOTES_FILENAME = 'notes.json'
AVERAGE_FILENAME = 'average.json'
DEVIATIONS_FROM_AVERAGE_FILENAME = 'deviations_from_average.json'
DEVIATIONS_FROM_SCORE_FILENAME = 'deviations_from_score.json'

PERFORMERS = [f'p{i}' for i in range(11)]


def get_performer(match_filename, dataset):
    if 'score' in match_filename:
        return 'score'
    if dataset == 'e_competition':
        performance = match_filename.split('_')[0]
        return performance.split('-')[0]
    elif dataset == 'labelling':
        return match_filename.split('_')[3]


def get_notes_df(file_path: str) -> pd.DataFrame:
    lines = []
    with open(file_path) as f:
        lines = f.readlines()
    notes = []
    for line in lines:
        if '//' not in line:
            notes.append(line.split('\t'))

    notes_df = pd.DataFrame(notes, columns=MATCH_COLUMN_NAMES)

    # Fix data types for dataframe
    notes_df = notes_df.astype({
        'id': int,
        'time_onset': float,
        'time_offset': float,
        'pitch': str,
        'velocity_onset': int,
        'velocity_offset': int,
        'channel': int,
        'match_status': int,
        'time_score': int,
        'note_id': str,
        'error_index': int,
        'skip_index': str
    })

    # Add duration feature
    notes_df['duration'] = notes_df['time_offset'] - notes_df['time_onset']

    # Add Inter Onset Interval
    notes_df['inter_onset_interval'] = notes_df['time_onset'].diff()

    # Add Offset Time Duration
    notes_df['offset_time_duration'] = notes_df['time_onset'].shift() - notes_df['time_offset']

    # Remove notes that are not found in the score
    unmatched = notes_df['note_id'] == '*'
    matched_notes = notes_df[~unmatched].reset_index(drop=True)

    # Remove duplicates, several notes have been matched with the same note in the score
    removed_duplicates = matched_notes[~matched_notes['note_id'].duplicated(keep='first')].reset_index(drop=True)
    return removed_duplicates


def get_notes_df_from_all_match_files(dataset: str, piece: str) -> pd.DataFrame:
    match_folder_path = match_path(dataset, piece)
    performances = get_files(match_folder_path)
    all_notes = pd.DataFrame()
    for match_filename in performances:
        performer = get_performer(match_filename, dataset)
        match_filepath = match_path(dataset, piece, match_filename)
        new_notes = get_notes_df(match_filepath)
        new_notes['performer'] = performer

        if dataset == 'labelling':
            new_notes['segment'] = match_filename.split('_')[4]
            new_notes['uid'] = performer + '-' + new_notes['segment'] + '-' + new_notes['note_id']
        else:
            new_notes['uid'] = performer + '-' + new_notes['note_id']
        all_notes = pd.concat([all_notes, new_notes.set_index(['uid'])])
    return all_notes


def compute_average_performance(notes: pd.DataFrame) -> pd.DataFrame:
    averages = notes.groupby(by=notes['note_id']).mean()
    stds = notes.groupby(by=notes['note_id']).std()
    return averages[NUMBER_COLUMN_NAMES].join(stds[NUMBER_COLUMN_NAMES], lsuffix='', rsuffix='_std')


def compute_deviations(notes: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    deviations = pd.DataFrame()
    for performer in notes['performer'].unique():
        new_deviations = pd.DataFrame()
        performer_notes = notes[notes['performer'] == performer].set_index('note_id')

        for column in NUMBER_COLUMN_NAMES:
            new_deviations[column] = performer_notes[column] - reference[column]

        # Filter out values that are present in reference, but not in the given performance
        mask = new_deviations.index.isin(performer_notes.index)
        filtered_deviations = pd.DataFrame(new_deviations[mask])

        filtered_deviations['performer'] = performer
        if 'segment' in notes.columns:
            filtered_deviations = filtered_deviations.join(performer_notes['segment'])

        df = filtered_deviations.reset_index()

        deviations = pd.concat([deviations, df])
    return deviations.reset_index(drop=True)


def transform_data(data, chunk_size=50, chunk_offset=25):
    X = []
    y = []
    for performer in data['performer'].unique():
        performer_mask = data['performer'] == performer
        performer_test = data[performer_mask].reset_index(drop=True).drop(columns=['performer'])
        performer_id = PERFORMERS.index(performer)
        for chunk in chunker(performer_test, chunk_size, chunk_offset):
            y.append(performer_id)
            X.append(chunk.to_numpy().transpose().flatten())

    return np.array(X), np.array(y)


def standardize_df(df, means=None, stds=None):
    df_values = df.drop(columns=['performer'])
    if means is None or stds is None:
        means = df_values.mean()
        stds = df_values.std()

    standardized = (df_values - means) / stds
    standardized['performer'] = df['performer']
    return standardized, means, stds


def normalize_df(df, min_values=None, max_values=None):
    df_values = df.drop(columns=['performer'])
    if min_values is None or max_values is None:
        min_values = df_values.min()
        max_values = df_values.max()

    norm = (df_values - min_values) / (max_values - min_values)
    norm['performer'] = df['performer']
    return norm, min_values, max_values
