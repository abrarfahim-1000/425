from __future__ import annotations

import pretty_midi


def rhythm_diversity(pm: pretty_midi.PrettyMIDI) -> float:
    """
    D = #unique_durations / #total_notes

    Higher values indicate more rhythmic variety.
    """
    durations = [
        round(note.end - note.start, 3)
        for inst in pm.instruments if not inst.is_drum
        for note in inst.notes
    ]
    if not durations:
        return 0.0
    return len(set(durations)) / len(durations)


def repetition_ratio(pm: pretty_midi.PrettyMIDI, n: int = 4) -> float:
    """
    R = #repeated_patterns / #total_patterns

    Counts repeated n-gram pitch patterns. Higher values mean more repetition.
    """
    pitches = [
        note.pitch
        for inst in pm.instruments if not inst.is_drum
        for note in inst.notes
    ]
    if len(pitches) < n * 2:
        return 0.0
    patterns = [tuple(pitches[i:i + n]) for i in range(len(pitches) - n + 1)]
    if not patterns:
        return 0.0
    return (len(patterns) - len(set(patterns))) / len(patterns)
