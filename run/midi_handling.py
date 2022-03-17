from utils.midi import get_midi_df
from utils.paths import raw_midi_path

PIECE = 'D960'
FILENAME = 'p3-0.MID'

file_path = raw_midi_path(PIECE, FILENAME)

df = get_midi_df(file_path)
print(df)
