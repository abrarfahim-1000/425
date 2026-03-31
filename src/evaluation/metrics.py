import pretty_midi
from pathlib import Path
import sys

import pandas as pd

# Add src to path when run as a script.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import GENERATED_MIDI_DIR, MAESTRO_CSV, MAESTRO_DIR
from src.evaluation.pitch_histogram import get_pitch_histogram, pitch_histogram_similarity
from src.evaluation.rhythm_score import rhythm_diversity, repetition_ratio

def compute_all_metrics(gen_midi_path, ref_midi_path=None):
    gen_pm = pretty_midi.PrettyMIDI(gen_midi_path)
    
    metrics = {
        "rhythm_diversity": rhythm_diversity(gen_pm),
        "repetition_ratio": repetition_ratio(gen_pm)
    }
    
    if ref_midi_path:
        ref_pm = pretty_midi.PrettyMIDI(ref_midi_path)
        metrics["pitch_histogram_similarity"] = pitch_histogram_similarity(gen_pm, ref_pm)
        
    return metrics


def evaluate_generated_midis(output_csv: Path | None = None, ref_file: str | Path | None = None):
    """
    Evaluates all .mid files in outputs/generated_midis and writes a CSV report.

    Args:
        output_csv: Destination CSV path. Defaults to outputs/generated_midis/evaluation_results.csv.
        ref_file:   Reference MIDI for pitch histogram similarity. If None, falls back
                    to the first MAESTRO training file when the CSV is available.
                    Pass an explicit path for Task 2 / Task 3 evaluations that use
                    non-MAESTRO reference material.
    """

    if ref_file is None:
        if MAESTRO_CSV.exists():
            df = pd.read_csv(MAESTRO_CSV)
            train_df = df[df["split"] == "train"]
            ref_file = MAESTRO_DIR / train_df.iloc[0]["midi_filename"]
            print(f"Using reference file: {ref_file}")
        else:
            print("[metrics] No ref_file provided and MAESTRO CSV not found. Skipping pitch similarity.")

    results = []
    for midi_path in sorted(GENERATED_MIDI_DIR.glob("*.mid")):
        filename = midi_path.name
        print(f"Evaluating {filename}...")
        metrics = compute_all_metrics(str(midi_path), str(ref_file) if ref_file else None)
        metrics["filename"] = filename
        results.append(metrics)

    if not results:
        print("No MIDI files found for evaluation.")
        return None

    results_df = pd.DataFrame(results)
    print("\nEvaluation Results:")
    print(results_df.to_string())

    if output_csv is None:
        output_csv = GENERATED_MIDI_DIR / "evaluation_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")
    return results_df


if __name__ == "__main__":
    evaluate_generated_midis()
