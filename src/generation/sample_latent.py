import torch
import numpy as np
import os
from pathlib import Path
import sys
import pretty_midi

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import CHECKPOINT_DIR, GENERATED_MIDI_DIR, DEVICE, AE_CONFIG, FS
from src.models.autoencoder import MusicAutoencoder
from src.preprocessing.piano_roll import piano_roll_to_pretty_midi

def sample_ae(num_samples=5):
    print(f"Using device: {DEVICE}")
    
    # Load model
    model = MusicAutoencoder(
        input_size=128, 
        hidden_size=AE_CONFIG['hidden_size'], 
        latent_dim=AE_CONFIG['latent_dim'],
        seq_len=AE_CONFIG['seq_len']
    ).to(DEVICE)
    
    checkpoint_path = CHECKPOINT_DIR / "best_ae.pt"
    if not checkpoint_path.exists():
        print(f"Checkpoint not found at {checkpoint_path}")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    print(f"Generating {num_samples} samples from latent space...")
    
    with torch.no_grad():
        for i in range(num_samples):
            # Sample random z ~ N(0, 1) or uniform
            # Standard AE doesn't guarantee N(0,1), but let's try
            z = torch.randn(1, AE_CONFIG['latent_dim']).to(DEVICE)
            
            x_hat = model.decoder(z) # (1, seq_len, 128)
            
            # Convert to numpy and transpose to (128, seq_len)
            roll = x_hat.squeeze(0).cpu().numpy().T
            
            pm = piano_roll_to_pretty_midi(roll, fs=FS)
            output_path = GENERATED_MIDI_DIR / f"task1_sample_{i}.mid"
            pm.write(str(output_path))
            print(f"Saved {output_path}")

if __name__ == "__main__":
    sample_ae()
