"""Simple attachment core tracking preferences for peer slots."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class AttachmentCore(nn.Module):
    """Maintains a tiny associative memory over peer slot embeddings."""

    def __init__(self, slot_dim: int, max_entities: int = 4, ema_decay: float = 0.9):
        super().__init__()
        self.proj = nn.Linear(slot_dim, 1)
        self.register_buffer("values", torch.zeros(max_entities))
        self.ema_decay = ema_decay
        self.max_entities = max_entities

    def update_from_slots(self, peer_slots: torch.Tensor, reward: float) -> None:
        """Update attachment values given peer slots and reward.

        peer_slots: [B, N, slot_dim]
        """
        if peer_slots.numel() == 0:
            return
        scores = torch.tanh(self.proj(peer_slots)).squeeze(-1)  # [B, N]
        mean_scores = scores.mean(dim=0).detach()
        reward_term = torch.clamp(torch.tensor(reward, device=peer_slots.device), -1.0, 1.0)
        new_vals = torch.clamp(mean_scores + 0.1 * reward_term, -1.0, 1.0)
        if self.values.device != peer_slots.device:
            self.values = self.values.to(peer_slots.device)
        self.values[: min(self.max_entities, new_vals.numel())] = (
            self.ema_decay * self.values[: min(self.max_entities, new_vals.numel())] + (1 - self.ema_decay) * new_vals[: self.max_entities]
        )

    def get_attachment_values(self, peer_slots: torch.Tensor) -> torch.Tensor:
        """Return attachment scores per peer slot (broadcasted values)."""
        if peer_slots.numel() == 0:
            return torch.zeros(peer_slots.shape[:2], device=peer_slots.device)
        vals = self.values.to(peer_slots.device)
        return torch.tanh(vals[: peer_slots.shape[1]]).unsqueeze(0).expand(peer_slots.shape[0], -1)
