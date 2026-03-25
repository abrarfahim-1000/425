import numpy as np
import pretty_midi
from typing import List

def get_pitch_histogram(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    """Returns a 12-dim pitch histogram (C, C#, ..., B) normalized to sum to 1."""
    histogram = np.zeros(12)
    for instrument in pm.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                histogram[note.pitch % 12] += 1
    if np.sum(histogram) > 0:
        histogram /= np.sum(histogram)
    return histogram

def pitch_histogram_similarity(gen_midi: pretty_midi.PrettyMIDI, ref_midi: pretty_midi.PrettyMIDI) -> float:
    """
    Computes similarity between two pitch histograms using 1 - Manhattan distance.
    H(p,q) = sum_i |p_i - q_i|
    Similarity = 1 - 0.5 * sum_i |p_i - q_i| (normalized to [0, 1])
    """
    p = get_pitch_histogram(gen_midi)
    q = get_pitch_histogram(ref_midi)
    
    distance = np.sum(np.abs(p - q))
    similarity = 1 - 0.5 * distance
    return max(0.0, similarity)

def rhythm_diversity(pm: pretty_midi.PrettyMIDI) -> float:
    """
    D = #unique_durations / #total_notes
    """
    durations = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            durations.append(round(note.end - note.start, 3))
            
    if not durations:
        return 0.0
        
    unique_durations = len(set(durations))
    diversity = unique_durations / len(durations)
    return diversity

def repetition_ratio(pm: pretty_midi.PrettyMIDI, n=4) -> float:
    """
    R = #repeated_patterns / #total_patterns
    Simplistic: count repeated sequences of pitches.
    """
    pitches = []
    for instrument in pm.instruments:
        for note in instrument.notes:
            pitches.append(note.pitch)
            
    if len(pitches) < n * 2:
        return 0.0
        
    patterns = []
    for i in range(len(pitches) - n + 1):
        patterns.append(tuple(pitches[i:i+n]))
        
    if not patterns:
        return 0.0
        
    unique_patterns = len(set(patterns))
    ratio = (len(patterns) - unique_patterns) / len(patterns)
    return ratio

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
