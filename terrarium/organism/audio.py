"""Proto-audio encoder turning binaural loudness into hemisphere context."""

from __future__ import annotations

import torch
from torch import nn


class AudioEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim * 2),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        audio: [B, 2] -> returns (left_ctx, right_ctx) each [B, out_dim]
        """
        h = self.net(audio)
        left, right = h.chunk(2, dim=-1)
        return left, right
