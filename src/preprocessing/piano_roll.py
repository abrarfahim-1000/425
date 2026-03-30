import numpy as np
import pretty_midi

def midi_to_piano_roll(midi: pretty_midi.PrettyMIDI, fs: int = 16) -> np.ndarray:
    """
    Converts a PrettyMIDI object to a binarized piano roll of shape (128, T).

    Args:
        midi (pretty_midi.PrettyMIDI): Parsed MIDI object.
        fs (int): Sampling frequency (steps per bar/second). Default is 16.
    """
    return (midi.get_piano_roll(fs=fs) > 0).astype(np.float32)

def segment_piano_roll(roll: np.ndarray, window: int = 64) -> list[np.ndarray]:
    """
    Segments a piano roll into fixed-length windows.

    Args:
        roll (np.ndarray): Piano roll of shape (128, T).
        window (int): Segment length in steps.

    Returns:
        list[np.ndarray]: List of segments of shape (128, window).
    """
    T = roll.shape[1]
    return [roll[:, i:i + window] for i in range(0, T - window + 1, window)]

def piano_roll_to_pretty_midi(roll: np.ndarray, fs: int = 16, program: int = 0) -> pretty_midi.PrettyMIDI:
    """
    Converts a piano roll back into a PrettyMIDI object.
    
    Args:
        roll (np.ndarray): Piano roll of shape (128, T).
        fs (int): Sampling frequency.
        program (int): MIDI program number (default 0: Piano).
        
    Returns:
        pretty_midi.PrettyMIDI: Reconstructed MIDI object.
    """
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)
    pm.instruments.append(instrument)
    
    # roll is (128, T)
    # Thresholding for binarization if not already
    roll_bin = (roll > 0.5)
    
    for pitch in range(128):
        # Find continuous sequences of 1s for this pitch
        notes = roll_bin[pitch, :]
        if not np.any(notes):
            continue
            
        # Add 0 at start and end to catch edges
        padded_notes = np.pad(notes, (1, 1), 'constant')
        diff = np.diff(padded_notes.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for s, e in zip(starts, ends):
            # Time in seconds
            start_time = s / fs
            end_time = e / fs
            note = pretty_midi.Note(
                velocity=100,
                pitch=pitch,
                start=start_time,
                end=end_time
            )
            instrument.notes.append(note)
            
    return pm
