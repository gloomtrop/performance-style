import os
from utils import paths, match

performance = 'p3-0'

match_filename = performance + '_match.txt'
notes_filename = performance + '.json'

match_filepath = os.path.join(paths.get_root_folder(), 'processed data', 'D960', 'match', match_filename)
notes_filepath = os.path.join(paths.get_root_folder(), 'processed data', 'D960', 'notes', notes_filename)

notes = match.get_notes_df(match_filepath)
notes.to_json(notes_filepath)