# src/models/gru_classifier.py

import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2, num_layers=1, dropout=0.2):
        """
        GRU-based classifier for motion sequences.

        Args:
            input_dim (int): Number of features per timestep.
            hidden_dim (int): Hidden size of the GRU.
            num_classes (int): Output classes (default=2).
            num_layers (int): Number of stacked GRU layers.
            dropout (float): Dropout applied between GRU layers (if num_layers > 1).
        """
        super(GRUClassifier, self).__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        _, hidden = self.gru(x)  # hidden shape: (num_layers, batch_size, hidden_dim)
        hidden = hidden[-1]      # take last layer’s hidden state (batch_size, hidden_dim)
        logits = self.fc(hidden) # (batch_size, num_classes)
        return logits
