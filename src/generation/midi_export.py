from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# Add src to path when run as a script.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import FS
from src.preprocessing.piano_roll import piano_roll_to_pretty_midi


def export_piano_roll_to_midi(
    roll: np.ndarray,
    output_path: str | Path,
    fs: int = FS,
) -> None:
    """
    Converts a piano-roll array to a MIDI file and saves it to disk.

    Args:
        roll: Piano roll of shape (128, T).
        output_path: Destination file path (.mid).
        fs: Sampling frequency used during piano-roll conversion.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pm = piano_roll_to_pretty_midi(roll, fs=fs)
    pm.write(str(output_path))
