import numpy as np
import os
from pathlib import Path
import sys
import pretty_midi

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import GENERATED_MIDI_DIR, PROCESSED_DATA_DIR, FS, SEQ_LEN
from src.preprocessing.piano_roll import piano_roll_to_pretty_midi

def generate_random_music(num_steps=64, num_samples=5):
    print("Generating Random Baseline samples...")
    for i in range(num_samples):
        # Random binary matrix (128, num_steps)
        roll = (np.random.rand(128, num_steps) > 0.98).astype(np.float32)
        pm = piano_roll_to_pretty_midi(roll, fs=FS)
        output_path = GENERATED_MIDI_DIR / f"baseline_random_{i}.mid"
        pm.write(str(output_path))
        print(f"Saved {output_path}")

def train_markov_chain():
    # Load training data
    train_data_path = PROCESSED_DATA_DIR / "maestro_train.npy"
    if not train_data_path.exists():
        print(f"Training data not found at {train_data_path}")
        return None
        
    train_data = np.load(train_data_path) # (N, 64, 128)
    
    transitions = {}
    
    print("Training Markov Chain...")
    # Take a subset if too large
    subset = train_data[:2000]
    
    for seq in subset:
        for t in range(len(seq) - 1):
            curr = tuple(seq[t])
            nxt = tuple(seq[t+1])
            if curr not in transitions:
                transitions[curr] = {}
            transitions[curr][nxt] = transitions[curr].get(nxt, 0) + 1
            
    # Normalize
    for curr in list(transitions.keys()):
        total = sum(transitions[curr].values())
        for nxt in transitions[curr]:
            transitions[curr][nxt] /= total
            
    return transitions

def generate_markov_music(transitions, num_steps=64, num_samples=5):
    if transitions is None:
        return
        
    print("Generating Markov Baseline samples...")
    all_keys = list(transitions.keys())
    
    for i in range(num_samples):
        curr = all_keys[np.random.randint(len(all_keys))]
        roll_seq = [np.array(curr)]
        
        for _ in range(num_steps - 1):
            if curr in transitions:
                options = list(transitions[curr].keys())
                probs = list(transitions[curr].values())
                # numpy choice expects options to be 1D array-like, but keys are tuples
                idx = np.random.choice(len(options), p=probs)
                curr = options[idx]
            else:
                curr = all_keys[np.random.randint(len(all_keys))]
            roll_seq.append(np.array(curr))
            
        roll = np.stack(roll_seq).T # (128, 64)
        pm = piano_roll_to_pretty_midi(roll, fs=FS)
        output_path = GENERATED_MIDI_DIR / f"baseline_markov_{i}.mid"
        pm.write(str(output_path))
        print(f"Saved {output_path}")

if __name__ == "__main__":
    generate_random_music()
    transitions = train_markov_chain()
    generate_markov_music(transitions)
