"""Shared Q-learning utilities for trainers and agent clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .organism import Organism


@dataclass
class TransitionBatch:
    states: torch.Tensor
    next_states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    weights: torch.Tensor


class QTrainer:
    """Reusable Q-learning helper (DQN-style)."""

    def __init__(self, gamma: float, target_clip: float = 50.0):
        self.gamma = gamma
        self.target_clip = target_clip

    def compute_td_loss(self, organism: Organism, batch: TransitionBatch) -> Tuple[torch.Tensor, torch.Tensor]:
        q_values = organism.q_network(batch.states)  # type: ignore[arg-type]
        q_sa = q_values.gather(1, batch.actions.unsqueeze(-1)).squeeze(-1)
        q_sa = torch.clamp(q_sa, -50.0, 50.0)
        with torch.no_grad():
            target_q = organism.target_network(batch.next_states)  # type: ignore[arg-type]
            max_next = torch.max(target_q, dim=1).values
            targets = batch.rewards + self.gamma * (1 - batch.dones) * max_next
            targets = torch.clamp(targets, -self.target_clip, self.target_clip)
        td_error = targets - q_sa
        loss = ((td_error.pow(2)) * batch.weights.squeeze()).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite q_loss")
        return loss, td_error.detach().abs()

    def apply_gradients(
        self, optimizer: torch.optim.Optimizer, loss: torch.Tensor, clip_params: Iterable[nn.Parameter], max_norm: float = 1.0
    ) -> None:
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(clip_params, max_norm)
        optimizer.step()
