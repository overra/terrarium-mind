"""Neural hemisphere core with a small GRU encoder."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class HemisphereCore(nn.Module):
    """Encodes observations plus emotion latent into a recurrent hidden state."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, obs: torch.Tensor, emotion_latent: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Update hidden state given obs and emotion latent."""
        x = torch.cat([obs, emotion_latent], dim=-1)
        x = self.activation(self.input_proj(x))
        h_next = self.gru_cell(x, hidden)
        return h_next


def init_hidden(batch_size: int, hidden_dim: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(batch_size, hidden_dim, device=device)
