from __future__ import annotations

import numpy as np
import pretty_midi


def get_pitch_histogram(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    """Returns a 12-dim pitch class histogram (C … B) normalized to sum to 1."""
    histogram = np.zeros(12)
    for instrument in pm.instruments:
        if not instrument.is_drum:
            for note in instrument.notes:
                histogram[note.pitch % 12] += 1
    total = histogram.sum()
    if total > 0:
        histogram /= total
    return histogram


def pitch_histogram_similarity(
    gen_midi: pretty_midi.PrettyMIDI,
    ref_midi: pretty_midi.PrettyMIDI,
) -> float:
    """
    Similarity between two pitch histograms using 1 - normalised Manhattan distance.

    H(p, q) = sum_i |p_i - q_i|
    Similarity = 1 - 0.5 * H(p, q)  → range [0, 1]
    """
    p = get_pitch_histogram(gen_midi)
    q = get_pitch_histogram(ref_midi)
    distance = np.sum(np.abs(p - q))
    return max(0.0, 1 - 0.5 * distance)
