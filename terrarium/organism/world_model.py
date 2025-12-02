"""Auxiliary predictive head for internal transition modeling."""

from __future__ import annotations

import torch
from torch import nn


class PredictiveHead(nn.Module):
    def __init__(self, emotion_dim: int, core_summary_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        input_dim = emotion_dim + core_summary_dim + action_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head_emotion = nn.Linear(hidden_dim, emotion_dim)
        self.head_core = nn.Linear(hidden_dim, core_summary_dim)

    def forward(
        self, current_emotion: torch.Tensor, current_core_summary: torch.Tensor, action_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([current_emotion, current_core_summary, action_onehot], dim=-1)
        h = self.mlp(x)
        pred_emotion = torch.clamp(self.head_emotion(h), -2.0, 2.0)
        pred_core = self.head_core(h)
        return pred_emotion, pred_core
