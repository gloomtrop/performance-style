import utils.preprocessing as pp
import utils.paths as pa
import os


PIECE = 'D960'
SAVE = True

# Paths
PATH = os.path.join(pa.get_root_folder(), 'processed data', PIECE)
NOTES_FILEPATH = os.path.join(PATH, 'notes.json')
AVERAGE_FILEPATH = os.path.join(PATH, 'avg.json')
DEVIATIONS_FILEPATH = os.path.join(PATH, 'deviations.json')


# Processing
notes = pp.get_notes_df_from_all_match_files(PIECE)

average = pp.compute_average_performance(notes)

deviations = pp.compute_deviations(notes, average)


# Saving
if SAVE:
    notes.to_json(NOTES_FILEPATH)
    average.to_json(AVERAGE_FILEPATH)
    deviations.to_json(DEVIATIONS_FILEPATH)