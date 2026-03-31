from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pretty_midi
from tqdm import tqdm

# Add src to path when run as a script.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import FS, MAESTRO_CSV, MAESTRO_DIR, PROCESSED_DATA_DIR, SEQ_LEN
from src.preprocessing.piano_roll import midi_to_piano_roll, segment_piano_roll

def load_midi(filepath: str) -> pretty_midi.PrettyMIDI | None:
    """
    Loads a MIDI file and returns a pretty_midi.PrettyMIDI object.
    
    Args:
        filepath (str): Path to the MIDI file.
        
    Returns:
        Optional[pretty_midi.PrettyMIDI]: The parsed MIDI object or None if failed.
    """
    try:
        return pretty_midi.PrettyMIDI(filepath)
    except Exception as e:
        print(f"Error loading MIDI file {filepath}: {e}")
        return None


def process_maestro_dataset(
    maestro_csv: Path = MAESTRO_CSV,
    maestro_root: Path = MAESTRO_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    fs: int = FS,
    seq_len: int = SEQ_LEN,
) -> None:
    """
    Processes MAESTRO splits into piano-roll segments and saves .npy files.

    Output files follow existing convention:
      maestro_train.npy, maestro_validation.npy, maestro_test.npy
    """

    print(f"Loading CSV from {maestro_csv}")
    df = pd.read_csv(maestro_csv)

    splits = df["split"].unique()
    print(f"Splits found: {splits}")

    for split in splits:
        split_df = df[df["split"] == split]
        print(f"Processing {split} split ({len(split_df)} files)...")

        all_segments = []
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            midi_path = maestro_root / row["midi_filename"]
            midi = load_midi(str(midi_path))
            if midi is None:
                continue

            roll = midi_to_piano_roll(midi, fs=fs)
            segments = segment_piano_roll(roll, window=seq_len)

            # Store as (seq_len, 128) uint8 to match training scripts.
            segments = [s.T.astype(np.uint8) for s in segments]
            all_segments.extend(segments)

        if not all_segments:
            print(f"No segments found for {split} split.")
            continue

        all_segments_np = np.stack(all_segments)
        output_path = output_dir / f"maestro_{split}.npy"
        print(f"Saving {len(all_segments)} segments to {output_path} (shape: {all_segments_np.shape})")
        np.save(output_path, all_segments_np)


if __name__ == "__main__":
    process_maestro_dataset()
