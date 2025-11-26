"""Slot-based hemisphere core for object-centric encoding."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
from torch import nn


class SlotUpdater(nn.Module):
    """Updates a slot embedding given an input feature vector."""

    def __init__(self, slot_dim: int, input_dim: int):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, slot_dim)
        self.mlp = nn.Sequential(nn.Linear(slot_dim, slot_dim), nn.ReLU(), nn.Linear(slot_dim, slot_dim))

    def forward(self, slot: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        h = self.gru(x, slot)
        return h + self.mlp(h)


class HemisphereSlotCore(nn.Module):
    """Maintains slots for self, objects, peers, and reflections."""

    def __init__(self, slot_dim: int, obj_slots: int, peer_slots: int, refl_slots: int, input_dim_per_entity: int):
        super().__init__()
        self.slot_dim = slot_dim
        self.obj_slots = obj_slots
        self.peer_slots = peer_slots
        self.refl_slots = refl_slots
        self.self_updater = SlotUpdater(slot_dim, input_dim_per_entity)
        self.obj_updater = SlotUpdater(slot_dim, input_dim_per_entity)
        self.peer_updater = SlotUpdater(slot_dim, input_dim_per_entity)
        self.refl_updater = SlotUpdater(slot_dim, input_dim_per_entity)

    def forward(
        self,
        self_feat: torch.Tensor,
        obj_feats: torch.Tensor,
        peer_feats: torch.Tensor,
        refl_feats: torch.Tensor,
        hidden: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hidden shape: [batch, n_slots, slot_dim]; slot order: [self, objs..., peers..., refls...]
        """
        batch = hidden.shape[0]
        slots = hidden.clone()
        idx = 0
        # self slot
        self_slot = slots[:, idx, :]
        self_slot = self.self_updater(self_slot, self_feat)
        slots[:, idx, :] = self_slot
        idx += 1
        # objects
        for i in range(self.obj_slots):
            slot = slots[:, idx, :]
            slot = self.obj_updater(slot, obj_feats[:, i, :])
            slots[:, idx, :] = slot
            idx += 1
        # peers
        for i in range(self.peer_slots):
            slot = slots[:, idx, :]
            slot = self.peer_updater(slot, peer_feats[:, i, :])
            slots[:, idx, :] = slot
            idx += 1
        # reflections
        for i in range(self.refl_slots):
            slot = slots[:, idx, :]
            slot = self.refl_updater(slot, refl_feats[:, i, :])
            slots[:, idx, :] = slot
            idx += 1

        summary = slots.mean(dim=1)
        return slots, summary

    def init_hidden(self, batch_size: int, device: torch.device, total_slots: int) -> torch.Tensor:
        return torch.zeros(batch_size, total_slots, self.slot_dim, device=device)
