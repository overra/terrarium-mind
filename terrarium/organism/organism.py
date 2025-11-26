"""Organism with neural hemisphere cores, bridge, and Q-policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
from copy import deepcopy
import random
import math
import torch.nn as nn

from terrarium.backend import TorchBackend

from .cores import Bridge
from .emotion import EmotionEngine, EmotionState
from .expression import ExpressionHead
from .slot_core import HemisphereSlotCore
from .policy import EpsilonGreedyPolicy
from .q_network import QNetwork
from typing import Iterable


@dataclass
class EncodedState:
    """Representation of the organism state for RL."""

    brain_state_tensor: torch.Tensor
    brain_state: List[float]
    hidden_left_in: List[float]
    hidden_right_in: List[float]
    hidden_left: List[float]
    hidden_right: List[float]
    emotion: EmotionState
    drives: Dict[str, float]
    core_affect: Dict[str, float]
    expression: Dict[str, Any]


class Organism:
    """Top-level container for Stage 1 organism components."""

    def __init__(
        self,
        action_space: Sequence[str],
        backend: TorchBackend | None = None,
        hidden_dim: int = 32,
        bridge_dim: int = 16,
        grid_size: int = 8,
        max_steps: int = 60,
        task_ids: Sequence[str] = ("goto_mirror", "touch_object"),
        policy_rng: random.Random | None = None,
        max_objects: int = 5,
        max_peers: int = 1,
        max_reflections: int = 2,
    ) -> None:
        self.action_space = list(action_space)
        self.action_to_idx = {a: i for i, a in enumerate(self.action_space)}
        self.backend = backend or TorchBackend()
        self.device = self.backend.device
        self.hidden_dim = hidden_dim
        self.bridge_dim = bridge_dim
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.task_ids = list(task_ids)
        self.max_objects = max_objects
        self.max_peers = max_peers
        self.max_reflections = max_reflections
        self.slot_input_dim = 6  # unified per-entity feature dim

        self.emotion_engine = EmotionEngine()
        self.expression_head = ExpressionHead()
        self.policy_head = EpsilonGreedyPolicy(self.action_space, rng=policy_rng)

        self.left_core: HemisphereSlotCore | None = None
        self.right_core: HemisphereSlotCore | None = None
        self.bridge: Bridge | None = None
        self.q_network: QNetwork | None = None
        self.target_network: QNetwork | None = None

        self.hidden_left: torch.Tensor | None = None
        self.hidden_right: torch.Tensor | None = None
        self.obs_dim: int | None = None
        self.emotion_dim: int | None = None

    def reset(self) -> None:
        """Reset internal states."""
        self.emotion_engine.reset()
        self.hidden_left = None
        self.hidden_right = None

    def _ensure_modules(self, obs_dim: int, emotion_dim: int) -> None:
        if self.left_core is None:
            slot_dim = self.hidden_dim
            input_dim_per_entity = self.slot_input_dim
            total_slots = 1 + self.max_objects + self.max_peers + self.max_reflections
            self.left_core = HemisphereSlotCore(
                slot_dim=slot_dim,
                obj_slots=self.max_objects,
                peer_slots=self.max_peers,
                refl_slots=self.max_reflections,
                input_dim_per_entity=input_dim_per_entity,
            ).to(self.device)
            self.right_core = HemisphereSlotCore(
                slot_dim=slot_dim,
                obj_slots=self.max_objects,
                peer_slots=self.max_peers,
                refl_slots=self.max_reflections,
                input_dim_per_entity=input_dim_per_entity,
            ).to(self.device)
            self.bridge = Bridge(self.hidden_dim, self.bridge_dim).to(self.device)
            slot_count = 1 + self.max_objects + self.max_peers + self.max_reflections
            concat_dim = self.hidden_dim * 2 * slot_count + emotion_dim
            self.brain_proj = nn.Linear(concat_dim, self.hidden_dim * 2).to(self.device)
            self.q_network = QNetwork(self.hidden_dim * 2, self.hidden_dim, len(self.action_space)).to(self.device)
            self.target_network = deepcopy(self.q_network).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
        self.obs_dim = obs_dim
        self.emotion_dim = emotion_dim

    def encode_observation(
        self,
        observation: Dict[str, Any],
        reward: float,
        novelty: float,
        prediction_error: float,
        info: Dict[str, Any],
    ) -> EncodedState:
        """Update world cores and emotion, returning the current brain state."""
        emotion_state = self.emotion_engine.update(
            reward=reward,
            novelty=novelty,
            prediction_error=prediction_error,
            mirror_contact=bool(info.get("mirror_contact", False)),
        )
        emotion_tensor = self.backend.tensor(emotion_state.latent, dtype=self.backend.float_dtype).unsqueeze(0)
        slices = self._split_observation(observation)
        self._ensure_modules(slices["self"].shape[-1], emotion_tensor.shape[-1])

        total_slots = 1 + self.max_objects + self.max_peers + self.max_reflections
        if self.hidden_left is None:
            self.hidden_left = self.left_core.init_hidden(1, self.device, total_slots)
            self.hidden_right = self.right_core.init_hidden(1, self.device, total_slots)
        h_left_in = self.hidden_left
        h_right_in = self.hidden_right

        # Hemisphere-specific slices (even/odd split)
        objs_left, objs_right = slices["objects"][:, ::2, :], slices["objects"][:, 1::2, :]
        peers_left, peers_right = slices["peers"][:, ::2, :], slices["peers"][:, 1::2, :]
        refl_left, refl_right = slices["reflections"][:, ::2, :], slices["reflections"][:, 1::2, :]

        # Pad halves to fixed counts
        def pad_half(tensor, target_slots):
            if tensor.shape[1] >= target_slots:
                return tensor[:, :target_slots, :]
            pad_slots = target_slots - tensor.shape[1]
            pad = torch.zeros(tensor.shape[0], pad_slots, tensor.shape[2], device=self.device)
            return torch.cat([tensor, pad], dim=1)

        objs_left = pad_half(objs_left, self.max_objects)
        objs_right = pad_half(objs_right, self.max_objects)
        peers_left = pad_half(peers_left, self.max_peers)
        peers_right = pad_half(peers_right, self.max_peers)
        refl_left = pad_half(refl_left, self.max_reflections)
        refl_right = pad_half(refl_right, self.max_reflections)

        h_left_slots, summary_left = self.left_core(slices["self"], objs_left, peers_left, refl_left, h_left_in)
        h_right_slots, summary_right = self.right_core(slices["self"], objs_right, peers_right, refl_right, h_right_in)
        # Apply bridge to summaries then broadcast
        mod_left, mod_right = self.bridge(summary_left, summary_right)
        h_left_slots = h_left_slots + mod_left.unsqueeze(1)
        h_right_slots = h_right_slots + mod_right.unsqueeze(1)

        self.hidden_left = h_left_slots.detach()
        self.hidden_right = h_right_slots.detach()

        concat = torch.cat([h_left_slots.flatten(1), h_right_slots.flatten(1), emotion_tensor], dim=-1)
        brain_state_tensor = self.brain_proj(concat)
        brain_state = brain_state_tensor.detach().squeeze(0).cpu().tolist()
        h_left_in_list = h_left_in.detach().flatten(1).squeeze(0).cpu().tolist()
        h_right_in_list = h_right_in.detach().flatten(1).squeeze(0).cpu().tolist()
        h_left_list = h_left_slots.detach().flatten(1).squeeze(0).cpu().tolist()
        h_right_list = h_right_slots.detach().flatten(1).squeeze(0).cpu().tolist()

        gaze_target = self._pick_gaze_target(observation)
        facing = observation.get("agent_pose", {}).get("facing", "up")
        expression = self.expression_head.generate(emotion_state.latent, facing, drives=self.emotion_engine.drives_dict(), gaze_target=gaze_target)

        return EncodedState(
            brain_state_tensor=brain_state_tensor,
            brain_state=brain_state,
            hidden_left_in=h_left_in_list,
            hidden_right_in=h_right_in_list,
            hidden_left=h_left_list,
            hidden_right=h_right_list,
            emotion=emotion_state,
            drives=self.emotion_engine.drives_dict(),
            core_affect=self.emotion_engine.core_affect_dict(),
            expression=expression,
        )

    def encode_batch_stateless(
        self,
        observations: list[Dict[str, Any]],
        emotion_latents: list[Sequence[float]],
    ) -> torch.Tensor:
        """Encode a batch of observations with zeroed hidden state (used in replay training)."""
        if not observations:
            return torch.empty(0, self.hidden_dim * 2 + len(emotion_latents[0]), device=self.device)

        obs_tensors = [self._obs_to_tensor(obs) for obs in observations]
        emotion_tensors = [
            self.backend.tensor(latent, dtype=self.backend.float_dtype).unsqueeze(0) for latent in emotion_latents
        ]

        obs_dim = obs_tensors[0].shape[-1]
        emotion_dim = emotion_tensors[0].shape[-1]
        self._ensure_modules(obs_dim, emotion_dim)

        h_left_batch = []
        h_right_batch = []
        zero_hidden = torch.zeros(1, self.hidden_dim, device=self.device)
        for obs_tensor, emo_tensor in zip(obs_tensors, emotion_tensors):
            obs_left = torch.cat([obs_tensor, torch.zeros((1, 1), device=self.device)], dim=-1)
            obs_right = torch.cat([obs_tensor, torch.ones((1, 1), device=self.device)], dim=-1)
            h_left = self.left_core(obs_left, emo_tensor, zero_hidden)  # type: ignore[arg-type]
            h_right = self.right_core(obs_right, emo_tensor, zero_hidden)  # type: ignore[arg-type]
            h_left, h_right = self.bridge(h_left, h_right)  # type: ignore[arg-type]
            h_left_batch.append(h_left)
            h_right_batch.append(h_right)

        h_left_cat = torch.cat(h_left_batch, dim=0)
        h_right_cat = torch.cat(h_right_batch, dim=0)
        emotion_cat = torch.cat(emotion_tensors, dim=0)
        brain_state_tensor = torch.cat([h_left_cat, h_right_cat, emotion_cat], dim=-1)
        return brain_state_tensor

    def parameters_for_learning(self) -> Iterable[torch.nn.Parameter]:
        """Return parameters of cores, bridge, and Q-network."""
        params: list[torch.nn.Parameter] = []
        if self.left_core is not None:
            params += list(self.left_core.parameters())
        if self.right_core is not None:
            params += list(self.right_core.parameters())
        if self.bridge is not None:
            params += list(self.bridge.parameters())
        if hasattr(self, "brain_proj"):
            params += list(self.brain_proj.parameters())
        if self.q_network is not None:
            params += list(self.q_network.parameters())
        return params

    def encode_replay_state(
        self,
        observation: Dict[str, Any],
        emotion_latent: Sequence[float],
        hidden_left: Sequence[float],
        hidden_right: Sequence[float],
    ) -> torch.Tensor:
        """Encode a stored transition state using provided hidden states."""
        emotion_tensor = self.backend.tensor(emotion_latent, dtype=self.backend.float_dtype).unsqueeze(0)
        obs_tensor = self._obs_to_tensor(observation)
        self._ensure_modules(obs_tensor.shape[-1], emotion_tensor.shape[-1])
        h_left = torch.tensor(hidden_left, dtype=self.backend.float_dtype, device=self.device).unsqueeze(0)
        h_right = torch.tensor(hidden_right, dtype=self.backend.float_dtype, device=self.device).unsqueeze(0)

        obs_left = torch.cat([obs_tensor, torch.zeros((1, 1), device=self.device)], dim=-1)
        obs_right = torch.cat([obs_tensor, torch.ones((1, 1), device=self.device)], dim=-1)

        h_left = self.left_core(obs_left, emotion_tensor, h_left)  # type: ignore[arg-type]
        h_right = self.right_core(obs_right, emotion_tensor, h_right)  # type: ignore[arg-type]
        h_left, h_right = self.bridge(h_left, h_right)  # type: ignore[arg-type]
        brain_state_tensor = torch.cat([h_left, h_right, emotion_tensor], dim=-1)
        return brain_state_tensor

    def select_action(self, brain_state_tensor: torch.Tensor, epsilon: float) -> tuple[str, torch.Tensor]:
        """Compute Q-values and sample an action."""
        if self.q_network is None:
            raise RuntimeError("Q-network not initialized.")
        q_values = self.q_network(brain_state_tensor)
        action = self.policy_head.select(q_values.squeeze(0), epsilon)
        return action, q_values.squeeze(0).detach()

    def update_target(self) -> None:
        """Sync target network with online network."""
        if self.q_network is None or self.target_network is None:
            return
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _split_observation(self, observation: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Produce per-entity tensors for slot cores (batch=1), padded to slot_input_dim."""
        feat_dim = self.slot_input_dim

        def pad_feat(values: List[float]) -> List[float]:
            vals = values[:feat_dim] + [0.0] * max(0, feat_dim - len(values))
            return vals[:feat_dim]

        if "self" not in observation:
            pose = observation.get("agent_pose", {})
            pos = [float(pose.get("x", 0)) / max(1, self.grid_size), float(pose.get("y", 0)) / max(1, self.grid_size)]
            orientation = 0.0
            vel = [0.0, 0.0]
            self_feat = torch.tensor([pad_feat([pos[0], pos[1], orientation, vel[0], vel[1]])], device=self.device)
            obj_feats = torch.zeros(1, self.max_objects, feat_dim, device=self.device)
            peer_feats = torch.zeros(1, self.max_peers, feat_dim, device=self.device)
            refl_feats = torch.zeros(1, self.max_reflections, feat_dim, device=self.device)
            return {"self": self_feat, "objects": obj_feats, "peers": peer_feats, "reflections": refl_feats}

        self_info = observation.get("self", {})
        pos = self_info.get("pos", [0.0, 0.0])
        orientation = float(self_info.get("orientation", 0.0))
        vel = self_info.get("velocity", [0.0, 0.0])
        self_feat = torch.tensor(
            [pad_feat([pos[0], pos[1], math.sin(orientation), math.cos(orientation), vel[0], vel[1]])],
            device=self.device,
        )

        def build_tensor(items: List[Dict[str, float]], fields: List[str], pad_len: int) -> torch.Tensor:
            rows: List[List[float]] = []
            for item in items[:pad_len]:
                rows.append(pad_feat([float(item.get(f, 0.0)) for f in fields]))
            while len(rows) < pad_len:
                rows.append(pad_feat([0.0 for _ in fields]))
            return torch.tensor([rows], dtype=self.backend.float_dtype, device=self.device)

        obj_feats = build_tensor(
            observation.get("objects", []), ["rel_x", "rel_y", "size", "visible", "type_id"], self.max_objects
        )
        peer_feats = build_tensor(
            observation.get("peers", []), ["rel_x", "rel_y", "orientation", "expression"], self.max_peers
        )
        refl_feats = build_tensor(
            observation.get("mirror_reflections", []), ["rel_x", "rel_y", "orientation"], self.max_reflections
        )

        return {"self": self_feat, "objects": obj_feats, "peers": peer_feats, "reflections": refl_feats}

    def _pick_gaze_target(self, observation: Dict[str, Any]) -> str | None:
        peers = observation.get("peers", [])
        reflections = observation.get("mirror_reflections", [])
        if peers and any(p.get("rel_x", 0.0) != 0.0 or p.get("rel_y", 0.0) != 0.0 for p in peers):
            return "peer"
        if reflections and any(r.get("rel_x", 0.0) != 0.0 or r.get("rel_y", 0.0) != 0.0 for r in reflections):
            return "mirror"
        return None
