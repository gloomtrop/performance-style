import pandas as pd

COLUMN_NAMES = ['id', 'time_onset', 'time_offset', 'pitch', 'velocity_onset', 'velocity_offset', 'channel',
                'match_status', 'time_score', 'note_id', 'error_index', 'skip_index']


def get_notes_df(file_path: str) -> pd.DataFrame:
    lines = []
    with open(file_path) as f:
        lines = f.readlines()
    notes = []
    for line in lines:
        if '//' not in line:
            notes.append(line.split('\t'))

    notes_df = pd.DataFrame(notes, columns=COLUMN_NAMES)

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

    # Remove notes that are not found in the score
    unmatched = notes_df['note_id'] == '*'
    matched_notes = notes_df[~unmatched].reset_index(drop=True)

    # Remove duplicates, several notes have been matched with the same note in the score
    removed_duplicates = matched_notes[~matched_notes['note_id'].duplicated(keep='first')].reset_index(drop=True)

    return removed_duplicates
