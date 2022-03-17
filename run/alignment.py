import os
import shutil
import subprocess

from utils.paths import get_files, raw_midi_path, alignment_path, match_path
from utils.progress import ProgressBar

PIECE = 'D960'

midi_files = get_files(raw_midi_path(PIECE))
segmented_midi_files = list(filter(lambda x: 'all' not in x, midi_files))

# Changing working directory to alignment directory
os.chdir(alignment_path())
print('Changed directory to alignment directory')

progress = ProgressBar(len(segmented_midi_files), text=f'Aligning midi performances of {PIECE}:')

for i, midi_file in enumerate(segmented_midi_files[:1]):
    progress.show(i)
    midi_name = midi_file.split('.')[0]

    # Copying midi file to alignment directory
    midi_src = raw_midi_path(PIECE, midi_file)
    midi_dst = alignment_path(midi_file)
    shutil.copyfile(midi_src, midi_dst)

    # Aligning
    process = subprocess.Popen(['./MusicXMLToMIDIAlign.sh', 'D960', midi_name], stdout=subprocess.DEVNULL)
    process.wait()

    # Copying match file to processed data
    match_file = midi_name + '_match.txt'
    match_src = alignment_path(match_file)
    match_dst = match_path(PIECE, match_file)
    shutil.copyfile(match_src, match_dst)

    # Identify extra created file
    spr_file = midi_name + '_spr.txt'
    spr_path = alignment_path(spr_file)

    # Delete used files
    os.remove(midi_dst)
    os.remove(match_src)
    os.remove(spr_path)
