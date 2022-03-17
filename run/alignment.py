import os
import shutil
import subprocess

from utils.paths import get_root_folder, get_files
from utils.progress import ProgressBar

PIECE = 'D960'
MIDI_PATH = os.path.join(get_root_folder(), 'data', 'raw', 'schubert', PIECE)
PROCESSED_MATCH_PATH = os.path.join(get_root_folder(), 'data', 'processed', PIECE, 'match')
ALIGNMENT_DIRECTORY_PATH = os.path.join(get_root_folder(), 'external', 'AlignmentTool')

midi_files = get_files(MIDI_PATH)
segmented_midi_files = list(filter(lambda x: 'all' not in x, midi_files))

# Changing working directory to alignment directory
os.chdir(ALIGNMENT_DIRECTORY_PATH)
print('Changed directory to alignment directory')

progress = ProgressBar(len(segmented_midi_files), text=f'Aligning midi performances of {PIECE}:')

for i, midi_file in enumerate(segmented_midi_files):
    progress.show(i)
    midi_name = midi_file.split('.')[0]

    # Copying midi file to alignment directory
    midi_src = os.path.join(MIDI_PATH, midi_file)
    midi_dst = os.path.join(ALIGNMENT_DIRECTORY_PATH, midi_file)
    shutil.copyfile(midi_src, midi_dst)

    # Aligning
    process = subprocess.Popen(['./MusicXMLToMIDIAlign.sh', 'D960', midi_name], stdout=subprocess.DEVNULL)
    process.wait()

    # Copying match file to processed data
    match_file = midi_name + '_match.txt'
    match_src = os.path.join(ALIGNMENT_DIRECTORY_PATH, match_file)
    match_dst = os.path.join(PROCESSED_MATCH_PATH, match_file)
    shutil.copyfile(match_src, match_dst)

    # Identify extra created file
    spr_file = midi_name + '_spr.txt'
    spr_path = os.path.join(ALIGNMENT_DIRECTORY_PATH, spr_file)

    # Delete used files
    os.remove(midi_dst)
    os.remove(match_src)
    os.remove(spr_path)
