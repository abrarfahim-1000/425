from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pretty_midi
from tqdm import tqdm

# Add src to path when run as a script.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    FS,
    GROOVE_DIR,
    MAESTRO_CSV,
    MAESTRO_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SEQ_LEN,
    SEED,
    TRAIN_VAL_TEST_SPLIT,
)
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


def _process_generic_midi_dataset(
    dataset_name: str,
    dataset_root: Path,
    output_dir: Path = PROCESSED_DATA_DIR,
    fs: int = FS,
    seq_len: int = SEQ_LEN,
    split_ratio: tuple[float, float, float] = TRAIN_VAL_TEST_SPLIT,
    seed: int = SEED,
) -> None:
    """
    Processes recursively discovered MIDI files and writes split .npy files as:
      {dataset_name}_train.npy, {dataset_name}_validation.npy, {dataset_name}_test.npy
    """

    midi_files = sorted(
        list(dataset_root.rglob("*.mid"))
        + list(dataset_root.rglob("*.midi"))
    )
    if not midi_files:
        print(f"No MIDI files found under {dataset_root}")
        return

    print(f"Processing {len(midi_files)} MIDI files from {dataset_root} as '{dataset_name}'...")
    all_segments = []
    for midi_path in tqdm(midi_files):
        midi = load_midi(str(midi_path))
        if midi is None:
            continue

        roll = midi_to_piano_roll(midi, fs=fs)
        segments = segment_piano_roll(roll, window=seq_len)
        segments = [s.T.astype(np.uint8) for s in segments]
        all_segments.extend(segments)

    if not all_segments:
        print(f"No valid segments found for dataset '{dataset_name}'.")
        return

    all_segments_np = np.stack(all_segments)
    total = len(all_segments_np)

    train_ratio, val_ratio, test_ratio = split_ratio
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratio must sum to 1.0, got {split_ratio}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(total)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    split_indices = {
        "train": indices[:train_end],
        "validation": indices[train_end:val_end],
        "test": indices[val_end:],
    }

    for split_name, idx in split_indices.items():
        split_data = all_segments_np[idx]
        out_path = output_dir / f"{dataset_name}_{split_name}.npy"
        np.save(out_path, split_data)
        print(f"Saved {len(split_data)} segments to {out_path} (shape: {split_data.shape})")


def process_groove_dataset(
    groove_root: Path = GROOVE_DIR,
    output_dir: Path = PROCESSED_DATA_DIR,
    fs: int = FS,
    seq_len: int = SEQ_LEN,
) -> None:
    _process_generic_midi_dataset(
        dataset_name="groove",
        dataset_root=groove_root,
        output_dir=output_dir,
        fs=fs,
        seq_len=seq_len,
    )


def process_lakh_dataset(
    lakh_root: Path = RAW_DATA_DIR / "lakh",
    output_dir: Path = PROCESSED_DATA_DIR,
    fs: int = FS,
    seq_len: int = SEQ_LEN,
) -> None:
    _process_generic_midi_dataset(
        dataset_name="lakh",
        dataset_root=lakh_root,
        output_dir=output_dir,
        fs=fs,
        seq_len=seq_len,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="maestro",
        choices=["maestro", "groove", "lakh", "all"],
        help="Dataset to preprocess.",
    )
    args = parser.parse_args()

    if args.dataset in {"maestro", "all"}:
        process_maestro_dataset()
    if args.dataset in {"groove", "all"}:
        process_groove_dataset()
    if args.dataset in {"lakh", "all"}:
        process_lakh_dataset()
