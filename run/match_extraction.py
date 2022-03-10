import os
from utils import paths, match

piece = 'D960'

match_path = os.path.join(paths.get_root_folder(), 'processed data', piece, 'match')
performances = paths.get_files(match_path)


for match_filename in performances:
    performance = match_filename.split('_')[0]
    notes_filename = performance + '.json'

    match_filepath = os.path.join(match_path, match_filename)
    notes_filepath = os.path.join(paths.get_root_folder(), 'processed data', piece, 'notes', notes_filename)

    notes = match.get_notes_df(match_filepath)
    notes.to_json(notes_filepath)