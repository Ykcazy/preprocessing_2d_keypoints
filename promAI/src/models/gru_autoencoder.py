# src/models/gru_autoencoder.py
import torch.nn as nn
import torch


class GRUAutoencoder(nn.Module):
    """
    GRU encoder-decoder autoencoder that reconstructs the full input sequence.
    Input: (batch, seq_len, input_dim)
    Output: (batch, seq_len, input_dim)
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.0):
        super(GRUAutoencoder, self).__init__()
        self.encoder = nn.GRU(input_size=input_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        # decoder expects hidden vectors of size hidden_dim and produces hidden_dim outputs per timestep
        self.decoder = nn.GRU(input_size=hidden_dim,
                              hidden_size=hidden_dim,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0)
        # map decoder hidden_dim -> input_dim at each timestep
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        enc_out, hidden = self.encoder(x)  # enc_out: (batch, seq_len, hidden_dim); hidden: (num_layers, batch, hidden_dim)

        # Use the last layer hidden state as decoder input
        # hidden_last: (batch, hidden_dim) from last layer
        hidden_last = hidden[-1]  # (batch, hidden_dim)

        # Repeat hidden_last across seq_len to form decoder inputs
        # decoder_input shape: (batch, seq_len, hidden_dim)
        decoder_input = hidden_last.unsqueeze(1).repeat(1, x.size(1), 1)

        dec_out, _ = self.decoder(decoder_input)  # (batch, seq_len, hidden_dim)

        # Map to input dimension per timestep
        out = self.output_layer(dec_out)  # (batch, seq_len, input_dim)
        return out
