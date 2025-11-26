"""Epsilon-greedy policy over a Q-network."""

from __future__ import annotations

import random
from typing import Sequence

import torch


class EpsilonGreedyPolicy:
    """Chooses actions based on Q-values with epsilon exploration."""

    def __init__(self, action_space: Sequence[str], rng: random.Random | None = None) -> None:
        self.action_space = list(action_space)
        self.rng = rng or random.Random()

    def select(self, q_values: torch.Tensor, epsilon: float) -> str:
        """Sample action using epsilon-greedy; q_values shape [num_actions]."""
        if self.rng.random() < epsilon:
            idx = self.rng.randrange(len(self.action_space))
        else:
            idx = int(torch.argmax(q_values).item())
        return self.action_space[idx]
