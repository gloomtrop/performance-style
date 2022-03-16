import os

from utils import paths, midi

d960_folder = os.path.join(paths.get_root_folder(), 'data', 'raw', 'schubert', 'D960')
file_name = 'p3-0.MID'
file_path = os.path.join(d960_folder, file_name)

df = midi.get_midi_df(file_path)
print(df)
