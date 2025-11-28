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

    def __init__(self, slot_dim: int, obj_slots: int, peer_slots: int, refl_slots: int, input_dim_per_entity: int, emotion_dim: int, vision_dim: int, audio_dim: int):
        super().__init__()
        self.slot_dim = slot_dim
        self.obj_slots = obj_slots
        self.peer_slots = peer_slots
        self.refl_slots = refl_slots
        self.self_updater = SlotUpdater(slot_dim, input_dim_per_entity)
        self.obj_updater = SlotUpdater(slot_dim, input_dim_per_entity)
        self.peer_updater = SlotUpdater(slot_dim, input_dim_per_entity)
        self.refl_updater = SlotUpdater(slot_dim, input_dim_per_entity)
        self.emotion_mlp = nn.Linear(emotion_dim, slot_dim * 2)
        self.vision_mlp = nn.Linear(vision_dim, slot_dim * 2)
        self.audio_mlp = nn.Linear(audio_dim, slot_dim * 2)

    def forward(
        self,
        self_feat: torch.Tensor,
        obj_feats: torch.Tensor,
        peer_feats: torch.Tensor,
        refl_feats: torch.Tensor,
        hidden: torch.Tensor,
        emotion: torch.Tensor,
        vision: torch.Tensor,
        audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hidden shape: [batch, n_slots, slot_dim]; slot order: [self, objs..., peers..., refls...]
        """
        # FiLM-style modulation
        gamma_e, beta_e = self.emotion_mlp(emotion).chunk(2, dim=-1)
        gamma_v, beta_v = self.vision_mlp(vision).chunk(2, dim=-1)
        gamma_a, beta_a = self.audio_mlp(audio).chunk(2, dim=-1)
        mod_hidden = gamma_e.unsqueeze(1) * hidden + beta_e.unsqueeze(1)
        mod_hidden = gamma_v.unsqueeze(1) * mod_hidden + beta_v.unsqueeze(1)
        mod_hidden = gamma_a.unsqueeze(1) * mod_hidden + beta_a.unsqueeze(1)

        slots_out = []
        idx = 0
        self_slot = self.self_updater(mod_hidden[:, idx, :], self_feat)
        slots_out.append(self_slot.unsqueeze(1))
        idx += 1
        for i in range(self.obj_slots):
            slot = self.obj_updater(mod_hidden[:, idx, :], obj_feats[:, i, :])
            slots_out.append(slot.unsqueeze(1))
            idx += 1
        for i in range(self.peer_slots):
            slot = self.peer_updater(mod_hidden[:, idx, :], peer_feats[:, i, :])
            slots_out.append(slot.unsqueeze(1))
            idx += 1
        for i in range(self.refl_slots):
            slot = self.refl_updater(mod_hidden[:, idx, :], refl_feats[:, i, :])
            slots_out.append(slot.unsqueeze(1))
            idx += 1

        slots = torch.cat(slots_out, dim=1)
        summary = slots.mean(dim=1)
        return slots, summary

    def init_hidden(self, batch_size: int, device: torch.device, total_slots: int) -> torch.Tensor:
        return torch.zeros(batch_size, total_slots, self.slot_dim, device=device)
