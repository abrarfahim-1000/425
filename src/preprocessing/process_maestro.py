import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.preprocessing.midi_parser import load_midi
from src.preprocessing.piano_roll import midi_to_piano_roll, segment_piano_roll
from src.config import MAESTRO_DIR, MAESTRO_CSV, PROCESSED_DATA_DIR, FS, SEQ_LEN

def process_maestro():
    print(f"Loading CSV from {MAESTRO_CSV}")
    df = pd.read_csv(MAESTRO_CSV)
    
    splits = df['split'].unique()
    print(f"Splits found: {splits}")
    
    for split in splits:
        split_df = df[df['split'] == split]
        print(f"Processing {split} split ({len(split_df)} files)...")
        
        all_segments = []
        # Limit for testing/speed if needed, but let's try full for now
        # split_df = split_df.head(10) 
        
        for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
            midi_path = MAESTRO_DIR / row['midi_filename']
            midi = load_midi(str(midi_path))
            if midi is None:
                continue
                
            roll = midi_to_piano_roll(midi, fs=FS)
            segments = segment_piano_roll(roll, window=SEQ_LEN)
            
            # segments is list of (128, 64)
            # Transpose to (64, 128) for (seq_len, features)
            segments = [s.T.astype(np.uint8) for s in segments]
            all_segments.extend(segments)
            
        if all_segments:
            all_segments_np = np.stack(all_segments)
            output_path = PROCESSED_DATA_DIR / f"maestro_{split}.npy"
            print(f"Saving {len(all_segments)} segments to {output_path} (shape: {all_segments_np.shape})")
            np.save(output_path, all_segments_np)
        else:
            print(f"No segments found for {split} split.")

if __name__ == "__main__":
    process_maestro()
