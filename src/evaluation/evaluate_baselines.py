import os
from pathlib import Path
import sys
import numpy as np
import pretty_midi
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import GENERATED_MIDI_DIR, MAESTRO_DIR, MAESTRO_CSV
from src.evaluation.metrics import compute_all_metrics

def evaluate():
    # Load a reference MIDI from training
    df = pd.read_csv(MAESTRO_CSV)
    train_df = df[df['split'] == 'train']
    ref_file = MAESTRO_DIR / train_df.iloc[0]['midi_filename']
    print(f"Using reference file: {ref_file}")
    
    results = []
    
    # Evaluate baselines and task 1 samples
    for filename in os.listdir(GENERATED_MIDI_DIR):
        if filename.endswith(".mid"):
            midi_path = GENERATED_MIDI_DIR / filename
            print(f"Evaluating {filename}...")
            metrics = compute_all_metrics(str(midi_path), str(ref_file))
            metrics["filename"] = filename
            results.append(metrics)
            
    if results:
        results_df = pd.DataFrame(results)
        print("\nEvaluation Results:")
        print(results_df.to_string())
        
        # Save to CSV
        output_csv = GENERATED_MIDI_DIR / "evaluation_results.csv"
        results_df.to_csv(output_csv, index=False)
        print(f"\nSaved results to {output_csv}")

if __name__ == "__main__":
    evaluate()
