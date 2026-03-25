import pretty_midi
import os
from typing import Optional

def load_midi(filepath: str) -> Optional[pretty_midi.PrettyMIDI]:
    """
    Loads a MIDI file and returns a pretty_midi.PrettyMIDI object.
    
    Args:
        filepath (str): Path to the MIDI file.
        
    Returns:
        Optional[pretty_midi.PrettyMIDI]: The parsed MIDI object or None if failed.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(filepath)
        return midi_data
    except Exception as e:
        print(f"Error loading MIDI file {filepath}: {e}")
        return None
