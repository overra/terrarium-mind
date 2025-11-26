"""Bridge MLP connecting hemisphere hidden states."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class Bridge(nn.Module):
    """Limited-bandwidth connector between the two hemispheres."""

    def __init__(self, hidden_dim: int, bridge_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, bridge_dim),
            nn.ReLU(),
            nn.Linear(bridge_dim, 2 * hidden_dim),
        )

    def forward(self, h_left: torch.Tensor, h_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = torch.cat([h_left, h_right], dim=-1)
        m = self.mlp(z)
        m_left, m_right = torch.chunk(m, 2, dim=-1)
        return h_left + m_left, h_right + m_right
