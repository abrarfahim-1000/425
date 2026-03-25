import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, latent_dim=128):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.latent_layer = nn.Linear(hidden_size, latent_dim)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_size)
        # Use the last layer's hidden state
        z = self.latent_layer(h_n[-1])
        return z

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_size=256, output_size=128, seq_len=64):
        super(LSTMDecoder, self).__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_size)
        self.lstm = nn.LSTM(latent_dim, hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        # z: (batch, latent_dim)
        # Reconstruct sequence step by step or all at once?
        # A simple way: repeat z as input to LSTM at each time step
        batch_size = z.size(0)
        
        # Initial hidden/cell state from z?
        # h_0 = self.latent_to_hidden(z).unsqueeze(0) # (1, batch, hidden_size)
        # c_0 = torch.zeros_like(h_0)
        
        # Or just use z as input at each step
        z_input = z.unsqueeze(1).repeat(1, self.seq_len, 1) # (batch, seq_len, latent_dim)
        
        lstm_out, _ = self.lstm(z_input)
        # lstm_out: (batch, seq_len, hidden_size)
        
        out = self.output_layer(lstm_out)
        return self.sigmoid(out)

class MusicAutoencoder(nn.Module):
    def __init__(self, input_size=128, hidden_size=256, latent_dim=128, seq_len=64):
        super(MusicAutoencoder, self).__init__()
        self.encoder = LSTMEncoder(input_size, hidden_size, latent_dim)
        self.decoder = LSTMDecoder(latent_dim, hidden_size, input_size, seq_len)
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
