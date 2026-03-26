import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import PROCESSED_DATA_DIR, CHECKPOINT_DIR, PLOTS_DIR, DEVICE, AE_CONFIG, SEQ_LEN
from src.models.autoencoder import MusicAutoencoder

def train_ae():
    print(f"Using device: {DEVICE}")
    
    # Load data
    train_path = PROCESSED_DATA_DIR / "maestro_train.npy"
    val_path = PROCESSED_DATA_DIR / "maestro_validation.npy"
    
    if not train_path.exists():
        print(f"Training data not found at {train_path}")
        return
    if not val_path.exists():
        print(f"Validation data not found at {val_path}")
        return
        
    print("Loading data...")
    x_train = np.load(train_path)
    x_val = np.load(val_path)

    if x_train.size == 0 or x_val.size == 0:
        print(
            "Processed data is empty. Re-run preprocessing and verify "
            f"{train_path} and {val_path} contain segments."
        )
        return
    
    # Create datasets
    train_ds = TensorDataset(torch.from_numpy(x_train))
    val_ds = TensorDataset(torch.from_numpy(x_val))
    
    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=AE_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=AE_CONFIG['batch_size'])

    if len(train_loader) == 0 or len(val_loader) == 0:
        print(
            "No training/validation batches available. Check processed split sizes "
            "and batch size configuration."
        )
        return
    
    # Model
    model = MusicAutoencoder(
        input_size=128, 
        hidden_size=AE_CONFIG['hidden_size'], 
        latent_dim=AE_CONFIG['latent_dim'],
        seq_len=AE_CONFIG['seq_len']
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=AE_CONFIG['lr'])
    criterion = nn.MSELoss()
    
    # Training Loop
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    print(f"Starting training for {AE_CONFIG['epochs']} epochs...")
    
    for epoch in range(AE_CONFIG['epochs']):
        model.train()
        total_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{AE_CONFIG['epochs']} [Train]"):
            x = batch[0].to(DEVICE, dtype=torch.float32)
            
            optimizer.zero_grad()
            x_hat = model(x)
            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{AE_CONFIG['epochs']} [Val]"):
                x = batch[0].to(DEVICE, dtype=torch.float32)
                x_hat = model(x)
                loss = criterion(x_hat, x)
                total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_ae.pt")
            print("Saved best model.")
            
        # Optional: Save checkpoint every N epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"ae_epoch_{epoch+1}.pt")
            
    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Task 1: LSTM Autoencoder Training Loss')
    plt.savefig(PLOTS_DIR / "task1_loss.png")
    plt.close()
    print(f"Loss curve saved to {PLOTS_DIR / 'task1_loss.png'}")

if __name__ == "__main__":
    train_ae()
