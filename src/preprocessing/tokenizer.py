from __future__ import annotations

import numpy as np

# Event-token vocabulary.
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2

NOTE_ON_BASE = 3
NUM_PITCHES = 128

NOTE_OFF_BASE = NOTE_ON_BASE + NUM_PITCHES

VELOCITY_BASE = NOTE_OFF_BASE + NUM_PITCHES
NUM_VELOCITY_BINS = 32

DURATION_BASE = VELOCITY_BASE + NUM_VELOCITY_BINS
MAX_DURATION_STEPS = 64

TIME_SHIFT_BASE = DURATION_BASE + MAX_DURATION_STEPS
MAX_TIME_SHIFT_STEPS = 32

VOCAB_SIZE = TIME_SHIFT_BASE + MAX_TIME_SHIFT_STEPS


def note_on_token(pitch: int) -> int:
    return NOTE_ON_BASE + pitch


def note_off_token(pitch: int) -> int:
    return NOTE_OFF_BASE + pitch


def velocity_token(velocity: int) -> int:
    velocity = int(np.clip(velocity, 1, 127))
    bin_idx = int(np.clip((velocity - 1) * NUM_VELOCITY_BINS / 127, 0, NUM_VELOCITY_BINS - 1))
    return VELOCITY_BASE + bin_idx


def duration_token(steps: int) -> int:
    steps = int(np.clip(steps, 1, MAX_DURATION_STEPS))
    return DURATION_BASE + (steps - 1)


def time_shift_token(steps: int) -> int:
    steps = int(np.clip(steps, 1, MAX_TIME_SHIFT_STEPS))
    return TIME_SHIFT_BASE + (steps - 1)


def token_to_pitch_from_note_on(token: int) -> int:
    return token - NOTE_ON_BASE


def token_to_pitch_from_note_off(token: int) -> int:
    return token - NOTE_OFF_BASE


def token_to_duration_steps(token: int) -> int:
    return (token - DURATION_BASE) + 1


def token_to_time_shift_steps(token: int) -> int:
    return (token - TIME_SHIFT_BASE) + 1


def token_is_note_on(token: int) -> bool:
    return NOTE_ON_BASE <= token < NOTE_ON_BASE + NUM_PITCHES


def token_is_note_off(token: int) -> bool:
    return NOTE_OFF_BASE <= token < NOTE_OFF_BASE + NUM_PITCHES


def token_is_velocity(token: int) -> bool:
    return VELOCITY_BASE <= token < VELOCITY_BASE + NUM_VELOCITY_BINS


def token_is_duration(token: int) -> bool:
    return DURATION_BASE <= token < DURATION_BASE + MAX_DURATION_STEPS


def token_is_time_shift(token: int) -> bool:
    return TIME_SHIFT_BASE <= token < TIME_SHIFT_BASE + MAX_TIME_SHIFT_STEPS


def _extract_notes_from_roll(seq: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Extracts note tuples (start_step, end_step, pitch, velocity) from (T, 128) roll.
    """

    if seq.ndim != 2 or seq.shape[1] != 128:
        raise ValueError(f"Expected seq shape (T, 128), got {seq.shape}")

    active = seq > 0
    notes = []
    t_len = seq.shape[0]

    for pitch in range(128):
        track = active[:, pitch]
        if not np.any(track):
            continue

        padded = np.pad(track.astype(np.int32), (1, 1), mode="constant")
        diff = np.diff(padded)
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        for start, end in zip(starts, ends):
            start = int(start)
            end = int(min(end, t_len))
            if end <= start:
                continue
            # Input rolls are binary in this project; keep a default expressive velocity token.
            notes.append((start, end, pitch, 100))

    notes.sort(key=lambda x: (x[0], x[2]))
    return notes


def piano_roll_sequence_to_event_tokens(seq: np.ndarray, max_seq_len: int = 256) -> np.ndarray:
    """
    Converts one piano-roll sequence to fixed-length event tokens.

    Event format per note:
      [time_shift]*, note_on, velocity, duration, note_off
    """

    notes = _extract_notes_from_roll(seq)
    tokens: list[int] = [BOS_TOKEN_ID]
    current_time = 0

    for start, end, pitch, vel in notes:
        delta = start - current_time
        while delta > 0:
            shift = min(delta, MAX_TIME_SHIFT_STEPS)
            tokens.append(time_shift_token(shift))
            current_time += shift
            delta -= shift

        dur_steps = max(1, end - start)
        while dur_steps > 0:
            # Long notes are decomposed into max-size duration chunks.
            dur_chunk = min(dur_steps, MAX_DURATION_STEPS)
            tokens.extend(
                [
                    note_on_token(pitch),
                    velocity_token(vel),
                    duration_token(dur_chunk),
                    note_off_token(pitch),
                ]
            )
            dur_steps -= dur_chunk

    tokens.append(EOS_TOKEN_ID)

    if len(tokens) > max_seq_len:
        tokens = tokens[: max_seq_len - 1] + [EOS_TOKEN_ID]

    arr = np.full((max_seq_len,), PAD_TOKEN_ID, dtype=np.int64)
    arr[: len(tokens)] = np.asarray(tokens, dtype=np.int64)
    return arr


def piano_roll_batch_to_event_tokens(batch: np.ndarray, max_seq_len: int = 256) -> np.ndarray:
    """
    Converts a batch (N, T, 128) into fixed-length event token IDs (N, max_seq_len).
    """

    if batch.ndim != 3 or batch.shape[2] != 128:
        raise ValueError(f"Expected batch shape (N, T, 128), got {batch.shape}")

    return np.stack(
        [piano_roll_sequence_to_event_tokens(seq, max_seq_len=max_seq_len) for seq in batch],
        axis=0,
    )


def tokens_to_piano_roll(tokens: np.ndarray, num_pitches: int = 128) -> np.ndarray:
    """
    Decodes event token IDs into a binary piano-roll of shape (128, T).
    """

    if tokens.ndim != 1:
        raise ValueError(f"Expected token shape (T,), got {tokens.shape}")

    events = [int(t) for t in tokens if int(t) != PAD_TOKEN_ID]
    if not events:
        return np.zeros((num_pitches, 1), dtype=np.float32)

    current_time = 0
    decoded_notes: list[tuple[int, int, int]] = []
    i = 0

    while i < len(events):
        tok = events[i]
        if tok == EOS_TOKEN_ID:
            break
        if tok in (BOS_TOKEN_ID, PAD_TOKEN_ID):
            i += 1
            continue

        if token_is_time_shift(tok):
            current_time += token_to_time_shift_steps(tok)
            i += 1
            continue

        if token_is_note_on(tok):
            pitch = token_to_pitch_from_note_on(tok)
            # Expected pattern: note_on, velocity, duration, note_off.
            vel_tok = events[i + 1] if i + 1 < len(events) else None
            dur_tok = events[i + 2] if i + 2 < len(events) else None

            if vel_tok is not None and token_is_velocity(vel_tok) and dur_tok is not None and token_is_duration(dur_tok):
                dur = token_to_duration_steps(dur_tok)
                start = current_time
                end = current_time + dur
                decoded_notes.append((start, end, pitch))

                # Skip optional note_off if present.
                if i + 3 < len(events) and token_is_note_off(events[i + 3]):
                    i += 4
                else:
                    i += 3
                continue

        i += 1

    if not decoded_notes:
        return np.zeros((num_pitches, max(1, current_time + 1)), dtype=np.float32)

    total_steps = max(end for _, end, _ in decoded_notes)
    roll = np.zeros((num_pitches, max(1, total_steps)), dtype=np.float32)
    for start, end, pitch in decoded_notes:
        roll[pitch, start:end] = 1.0
    return roll
