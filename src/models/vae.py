import torch
import torch.nn as nn


class VAEEncoder(nn.Module):
    """
    LSTM encoder that outputs parameters of a Gaussian latent distribution.

    Input:  (batch, seq_len, input_size)
    Output: mu, log_var each with shape (batch, latent_dim)
    """

    def __init__(self, input_size: int = 128, hidden_size: int = 256, latent_dim: int = 128):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.mu_layer = nn.Linear(hidden_size, latent_dim)
        self.log_var_layer = nn.Linear(hidden_size, latent_dim)

    def forward(self, x: torch.Tensor):
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]  # (batch, hidden_size)
        mu = self.mu_layer(h_last)
        log_var = self.log_var_layer(h_last)
        return mu, log_var


class VAEDecoder(nn.Module):
    """
    LSTM decoder that reconstructs a sequence from a latent vector.

    Input:  z with shape (batch, latent_dim)
    Output: x_hat with shape (batch, seq_len, output_size)
    """

    def __init__(
        self,
        latent_dim: int = 128,
        hidden_size: int = 256,
        output_size: int = 128,
        seq_len: int = 64,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(latent_dim, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z: torch.Tensor):
        # Repeat z across time steps (simple conditional decoding).
        z_input = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, latent_dim)
        lstm_out, _ = self.lstm(z_input)  # (batch, seq_len, hidden_size)
        out = self.output_layer(lstm_out)  # (batch, seq_len, output_size)
        return self.sigmoid(out)


class MusicVAE(nn.Module):
    """
    VAE for piano-roll sequences (Task 2).

    Forward returns:
      x_hat: (batch, seq_len, input_size)
      mu:     (batch, latent_dim)
      log_var:(batch, latent_dim)
    """

    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 256,
        latent_dim: int = 128,
        seq_len: int = 64,
    ):
        super().__init__()
        self.encoder = VAEEncoder(input_size=input_size, hidden_size=hidden_size, latent_dim=latent_dim)
        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            hidden_size=hidden_size,
            output_size=input_size,
            seq_len=seq_len,
        )

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * log_var)
        return mu + std * eps

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, z: torch.Tensor):
        return self.decoder(z)

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

