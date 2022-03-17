import pandas as pd

from utils.paths import processed_data_path
from utils.preprocessing import NOTES_FILENAME, AVERAGE_FILENAME, DEVIATIONS_FROM_AVERAGE_FILENAME, \
    DEVIATIONS_FROM_SCORE_FILENAME
from utils.preprocessing import get_notes_df_from_all_match_files, compute_average_performance, \
    compute_deviations

PIECE = 'D960'
SAVE = True

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
    notes.to_json(processed_data_path(PIECE, NOTES_FILENAME))
    average.to_json(processed_data_path(PIECE, AVERAGE_FILENAME))
    average_deviations.to_json(processed_data_path(PIECE, DEVIATIONS_FROM_AVERAGE_FILENAME))
    score_deviations.to_json(processed_data_path(PIECE, DEVIATIONS_FROM_SCORE_FILENAME))
    print('Files are saved')
