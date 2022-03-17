import os

import pandas as pd

from utils.paths import get_root_folder
from utils.preprocessing import get_notes_df_from_all_match_files, compute_average_performance, \
    compute_deviations

PIECE = 'D960'
SAVE = True

# Paths
PATH = os.path.join(get_root_folder(), 'data', 'processed', PIECE)

SCORE_FILEPATH = os.path.join(PATH, 'match', 'score_match.txt')
NOTES_FILEPATH = os.path.join(PATH, 'notes.json')
AVERAGE_FILEPATH = os.path.join(PATH, 'avg.json')
DEVIATIONS_FROM_AVERAGE_FILEPATH = os.path.join(PATH, 'deviations_from_average.json')
DEVIATIONS_FROM_SCORE_FILEPATH = os.path.join(PATH, 'deviations_from_score.json')

# Processing
notes = get_notes_df_from_all_match_files(PIECE)
print('Notes are extracted from the match files')

score_mask = notes['performer'] == 'score'
score_notes = pd.DataFrame(notes[score_mask])
performer_notes = pd.DataFrame(notes[~score_mask])

average = compute_average_performance(performer_notes)
print('Average is computed')

average_deviations = compute_deviations(performer_notes, average)
score_deviations = compute_deviations(performer_notes, score_notes.set_index('note_id'))
print('Deviations from average and score are computed')

# Saving
if SAVE:
    notes.to_json(NOTES_FILEPATH)
    average.to_json(AVERAGE_FILEPATH)
    average_deviations.to_json(DEVIATIONS_FROM_AVERAGE_FILEPATH)
    score_deviations.to_json(DEVIATIONS_FROM_SCORE_FILEPATH)
    print('Files are saved')
