import os
import shutil
import subprocess

from utils.paths import get_files, raw_path, alignment_path, match_path
from tqdm import tqdm

DATASET = 'labelling'
PIECE = 'D960'

midi_files = get_files(raw_path(DATASET, PIECE))

# Changing working directory to alignment directory
os.chdir(alignment_path())
print('Changed directory to alignment directory')

for midi_file in tqdm(midi_files):
    midi_name = midi_file.split('.')[0]

    # Copying midi file to alignment directory
    midi_src = raw_path(DATASET, PIECE, midi_file)
    midi_dst = alignment_path(midi_file)
    shutil.copyfile(midi_src, midi_dst)

    # Aligning
    process = subprocess.Popen(['./MusicXMLToMIDIAlign.sh', PIECE, midi_name], stdout=subprocess.DEVNULL)
    process.wait()

    # Copying match file to processed data
    match_file = midi_name + '_match.txt'
    match_src = alignment_path(match_file)
    match_dst = match_path(DATASET, PIECE, match_file)
    shutil.copyfile(match_src, match_dst)

    # Identify extra created file
    spr_file = midi_name + '_spr.txt'
    spr_path = alignment_path(spr_file)

    # Delete used files
    os.remove(midi_dst)
    os.remove(match_src)
    os.remove(spr_path)
