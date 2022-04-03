from utils.midi import get_midi_df
from utils.paths import raw_path

PIECE = 'D960'
FILENAME = 'p3-0.MID'
DATASET = 'e_competition'

file_path = raw_path(DATASET, PIECE, FILENAME)

df = get_midi_df(file_path)
print(df)
