import mido
import pandas as pd

MIDI_TYPES = ['note_on', 'note_off']

def get_midi_df(file_path):
    notes = []
    column_names = ['type', 'channel', 'note', 'velocity', 'time']
    midi = mido.MidiFile(file_path)
    for msg in midi:
        if msg.type in MIDI_TYPES:
            notes.append([msg.type, msg.channel, msg.note, msg.velocity, msg.time])

    return pd.DataFrame(notes, columns=column_names)