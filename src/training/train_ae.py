import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.config import PROCESSED_DATA_DIR, CHECKPOINT_DIR, PLOTS_DIR, DEVICE, AE_CONFIG, SEQ_LEN
from src.models.autoencoder import MusicAutoencoder


def train_ae(
    epochs: int | None = None,
    train_max_batches: int | None = None,
    val_max_batches: int | None = None,
):
    epochs = epochs if epochs is not None else AE_CONFIG['epochs']
    print(f"Using device: {DEVICE}")

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

    train_ds = TensorDataset(torch.from_numpy(x_train))
    val_ds = TensorDataset(torch.from_numpy(x_val))

    train_loader = DataLoader(train_ds, batch_size=AE_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=AE_CONFIG['batch_size'])

    if len(train_loader) == 0 or len(val_loader) == 0:
        print(
            "No training/validation batches available. Check processed split sizes "
            "and batch size configuration."
        )
        return

    model = MusicAutoencoder(
        input_size=128,
        hidden_size=AE_CONFIG['hidden_size'],
        latent_dim=AE_CONFIG['latent_dim'],
        seq_len=AE_CONFIG['seq_len']
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=AE_CONFIG['lr'])
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        train_batches_used = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            if train_max_batches is not None and batch_idx >= train_max_batches:
                break

            x = batch[0].to(DEVICE, dtype=torch.float32)
            optimizer.zero_grad()

            # Algorithm 1 (Task 1): z = f_phi(X), X_hat = g_theta(z)
            z = model.encode(x)
            x_hat = model.decode(z)

            loss = criterion(x_hat, x)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_batches_used += 1

        avg_train_loss = total_train_loss / max(1, train_batches_used)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0
        val_batches_used = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")):
                if val_max_batches is not None and batch_idx >= val_max_batches:
                    break

                x = batch[0].to(DEVICE, dtype=torch.float32)
                z = model.encode(x)
                x_hat = model.decode(z)
                loss = criterion(x_hat, x)
                total_val_loss += loss.item()
                val_batches_used += 1

        avg_val_loss = total_val_loss / max(1, val_batches_used)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "best_ae.pt")
            print("Saved best model.")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), CHECKPOINT_DIR / f"ae_epoch_{epoch+1}.pt")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=None, help="Override AE_CONFIG epochs (useful for smoke tests)")
    parser.add_argument("--train_max_batches", type=int, default=None, help="Smoke test: cap number of train batches")
    parser.add_argument("--val_max_batches", type=int, default=None, help="Smoke test: cap number of val batches")
    args = parser.parse_args()

    train_ae(
        epochs=args.epochs,
        train_max_batches=args.train_max_batches,
        val_max_batches=args.val_max_batches,
    )
