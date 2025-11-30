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
from .vision import VisionEncoder
from .audio import AudioEncoder
from .policy import EpsilonGreedyPolicy
from .q_network import QNetwork
from .attachment import AttachmentCore
from .world_model import PredictiveHead
from typing import Iterable


@dataclass
class EncodedState:
    """Representation of the organism state for RL."""

    brain_state_tensor: torch.Tensor
    brain_state: List[float]
    core_summary: List[float]
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
        hidden_dim: int = 64,
        bridge_dim: int = 16,
        grid_size: int = 8,
        max_steps: int = 60,
        task_ids: Sequence[str] = ("goto_mirror", "touch_object"),
        policy_rng: random.Random | None = None,
        max_objects: int = 5,
        max_peers: int = 1,
        max_reflections: int = 2,
        retina_channels: int = 7,
        vision_dim: int = 32,
    ) -> None:
        self.action_space = list(action_space)
        if "sleep" not in self.action_space:
            self.action_space.append("sleep")
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
        self.slot_input_dim = 11  # pos/orient/vel/time + body config
        self.is_sleeping: bool = False
        self.retina_channels = retina_channels
        self.vision_encoder = VisionEncoder(in_channels=retina_channels, hidden_dim=hidden_dim, out_dim=vision_dim).to(self.device)
        self.audio_encoder = AudioEncoder(out_dim=vision_dim).to(self.device)
        self.attachment_core = AttachmentCore(slot_dim=hidden_dim, max_entities=max_peers).to(self.device)
        self.predictive_head = PredictiveHead(
            emotion_dim=8, core_summary_dim=hidden_dim * 2, action_dim=len(self.action_space), hidden_dim=hidden_dim * 4
        ).to(self.device)

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
        self.is_sleeping = False

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
                emotion_dim=emotion_dim,
                vision_dim=self.vision_encoder.proj.out_features,
                audio_dim=self.audio_encoder.out_dim,
            ).to(self.device)
            self.right_core = HemisphereSlotCore(
                slot_dim=slot_dim,
                obj_slots=self.max_objects,
                peer_slots=self.max_peers,
                refl_slots=self.max_reflections,
                input_dim_per_entity=input_dim_per_entity,
                emotion_dim=emotion_dim,
                vision_dim=self.vision_encoder.proj.out_features,
                audio_dim=self.audio_encoder.out_dim,
            ).to(self.device)
            self.bridge = Bridge(self.hidden_dim, self.bridge_dim).to(self.device)
            slot_count = 1 + self.max_objects + self.max_peers + self.max_reflections
            concat_dim = self.hidden_dim * 2 * slot_count + emotion_dim + 4  # +4 for target feat
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
        intero_signals: Dict[str, float] | None = None,
    ) -> EncodedState:
        """Update world cores and emotion, returning the current brain state."""
        emotion_state = self.emotion_engine.update(
            reward=reward,
            novelty=novelty,
            prediction_error=prediction_error,
            mirror_contact=bool(info.get("mirror_contact", False)),
            intero_signals=intero_signals,
        )
        emotion_tensor = torch.tanh(self.backend.tensor(emotion_state.latent, dtype=self.backend.float_dtype)).unsqueeze(0)
        slices = self._split_observation(observation)
        # Vision
        retina = observation.get("retina")
        if retina is not None:
            retina_tensor = torch.tensor(retina, dtype=self.backend.float_dtype, device=self.device)
            if retina_tensor.ndim == 3:
                retina_tensor = retina_tensor.unsqueeze(0)  # [1, C, H, W]
            vision_left_vec, vision_right_vec = self.vision_encoder(retina_tensor)
        else:
            vision_left_vec = torch.zeros(1, self.vision_encoder.proj.out_features, device=self.device)
            vision_right_vec = torch.zeros(1, self.vision_encoder.proj.out_features, device=self.device)
        audio_obs = observation.get("audio", {})
        audio_tensor = torch.tensor(
            [[audio_obs.get("left", 0.0), audio_obs.get("right", 0.0)]],
            dtype=self.backend.float_dtype,
            device=self.device,
        )
        audio_left_vec, audio_right_vec = self.audio_encoder(audio_tensor)
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

        h_left_slots, summary_left = self.left_core(slices["self"], objs_left, peers_left, refl_left, h_left_in, emotion_tensor, vision_left_vec, audio_left_vec)
        h_right_slots, summary_right = self.right_core(slices["self"], objs_right, peers_right, refl_right, h_right_in, emotion_tensor, vision_right_vec, audio_right_vec)
        # Apply bridge to summaries then broadcast
        mod_left, mod_right = self.bridge(summary_left, summary_right)
        h_left_slots = h_left_slots + mod_left.unsqueeze(1)
        h_right_slots = h_right_slots + mod_right.unsqueeze(1)

        # Prevent hidden activations from exploding to non-finite values.
        h_left_slots = torch.clamp(h_left_slots, -10.0, 10.0)
        h_right_slots = torch.clamp(h_right_slots, -10.0, 10.0)
        summary_left = torch.clamp(summary_left, -10.0, 10.0)
        summary_right = torch.clamp(summary_right, -10.0, 10.0)

        # Attachment: peer slots span [1 + max_objects : 1 + max_objects + max_peers)
        peer_start = 1 + self.max_objects
        peer_end = peer_start + self.max_peers
        peer_slots = torch.cat(
            [h_left_slots[:, peer_start:peer_end, :], h_right_slots[:, peer_start:peer_end, :]],
            dim=0,
        )
        with torch.no_grad():
            self.attachment_core.update_from_slots(peer_slots.detach(), reward)
        attachment_scores = self.attachment_core.get_attachment_values(peer_slots)
        if attachment_scores.numel() > 0:
            mean_attach = float(attachment_scores.mean().item())
            self.emotion_engine.state.drives.social_hunger = max(
                0.0, self.emotion_engine.state.drives.social_hunger - 0.1 * mean_attach
            )

        self.hidden_left = h_left_slots.detach()
        self.hidden_right = h_right_slots.detach()

        target_slice = slices.get("target", torch.zeros(1, 4, device=self.device, dtype=self.backend.float_dtype))
        concat = torch.cat([h_left_slots.flatten(1), h_right_slots.flatten(1), emotion_tensor, target_slice], dim=-1)
        concat = torch.clamp(concat, -10.0, 10.0)
        brain_state_tensor = self.brain_proj(concat)
        if not torch.isfinite(brain_state_tensor).all():
            wandb.log({"debug/nonfinite_brain_state": 1})
            brain_state_tensor = torch.zeros_like(brain_state_tensor)
            h_left_slots = torch.zeros_like(h_left_slots)
            h_right_slots = torch.zeros_like(h_right_slots)
        brain_state = brain_state_tensor.detach().squeeze(0).cpu().tolist()
        core_summary_tensor = torch.cat([summary_left, summary_right], dim=-1)
        core_summary = core_summary_tensor.detach().squeeze(0).cpu().tolist()
        h_left_in_list = h_left_in.detach().flatten(1).squeeze(0).cpu().tolist()
        h_right_in_list = h_right_in.detach().flatten(1).squeeze(0).cpu().tolist()
        h_left_list = h_left_slots.detach().flatten(1).squeeze(0).cpu().tolist()
        h_right_list = h_right_slots.detach().flatten(1).squeeze(0).cpu().tolist()

        gaze_target = self._pick_gaze_target(observation)
        orientation = observation.get("self", {}).get("head_orientation", observation.get("self", {}).get("orientation", 0.0))
        expression = self.expression_head.generate(
            emotion_state.latent, orientation, drives=self.emotion_engine.drives_dict(), gaze_target=gaze_target
        )

        return EncodedState(
            brain_state_tensor=brain_state_tensor,
            brain_state=brain_state,
            core_summary=core_summary,
            hidden_left_in=h_left_in_list,
            hidden_right_in=h_right_in_list,
            hidden_left=h_left_list,
            hidden_right=h_right_list,
            emotion=emotion_state,
            drives=self.emotion_engine.drives_dict(),
            core_affect=self.emotion_engine.core_affect_dict(),
            expression=expression,
        )

    def parameters_for_learning(self) -> Iterable[torch.nn.Parameter]:
        """Return parameters of vision, cores, bridge, and Q-network."""
        params: list[torch.nn.Parameter] = []
        if self.vision_encoder is not None:
            params += list(self.vision_encoder.parameters())
        if self.audio_encoder is not None:
            params += list(self.audio_encoder.parameters())
        if self.predictive_head is not None:
            params += list(self.predictive_head.parameters())
        if self.left_core is not None:
            params += list(self.left_core.parameters())
        if self.right_core is not None:
            params += list(self.right_core.parameters())
        if self.bridge is not None:
            params += list(self.bridge.parameters())
        if hasattr(self, "brain_proj"):
            params += list(self.brain_proj.parameters())
        if self.attachment_core is not None:
            params += list(self.attachment_core.parameters())
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
        """Recompute brain state for replay using stored hidden inputs and observation."""
        emotion_tensor = torch.tanh(self.backend.tensor(emotion_latent, dtype=self.backend.float_dtype)).unsqueeze(0)
        slices = self._split_observation(observation)
        self._ensure_modules(slices["self"].shape[-1], emotion_tensor.shape[-1])

        total_slots = 1 + self.max_objects + self.max_peers + self.max_reflections
        h_left_in = (
            torch.tensor(hidden_left, dtype=self.backend.float_dtype, device=self.device)
            .view(1, total_slots, self.hidden_dim)
        )
        h_right_in = (
            torch.tensor(hidden_right, dtype=self.backend.float_dtype, device=self.device)
            .view(1, total_slots, self.hidden_dim)
        )

        objs_left, objs_right = slices["objects"][:, ::2, :], slices["objects"][:, 1::2, :]
        peers_left, peers_right = slices["peers"][:, ::2, :], slices["peers"][:, 1::2, :]
        refl_left, refl_right = slices["reflections"][:, ::2, :], slices["reflections"][:, 1::2, :]

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

        # Vision
        retina = observation.get("retina")
        if retina is not None:
            retina_tensor = torch.tensor(retina, dtype=self.backend.float_dtype, device=self.device)
            if retina_tensor.ndim == 3:
                retina_tensor = retina_tensor.unsqueeze(0)
            vision_left_vec, vision_right_vec = self.vision_encoder(retina_tensor)
        else:
            vision_left_vec = torch.zeros(1, self.vision_encoder.proj.out_features, device=self.device)
            vision_right_vec = torch.zeros(1, self.vision_encoder.proj.out_features, device=self.device)

        audio_obs = observation.get("audio", {})
        audio_tensor = torch.tensor(
            [[audio_obs.get("left", 0.0), audio_obs.get("right", 0.0)]],
            dtype=self.backend.float_dtype,
            device=self.device,
        )
        audio_left_vec, audio_right_vec = self.audio_encoder(audio_tensor)

        h_left_slots, summary_left = self.left_core(
            slices["self"], objs_left, peers_left, refl_left, h_left_in, emotion_tensor, vision_left_vec, audio_left_vec
        )
        h_right_slots, summary_right = self.right_core(
            slices["self"], objs_right, peers_right, refl_right, h_right_in, emotion_tensor, vision_right_vec, audio_right_vec
        )
        mod_left, mod_right = self.bridge(summary_left, summary_right)
        h_left_slots = h_left_slots + mod_left.unsqueeze(1)
        h_right_slots = h_right_slots + mod_right.unsqueeze(1)

        target_slice = slices.get("target", torch.zeros(1, 4, device=self.device, dtype=self.backend.float_dtype))
        concat = torch.cat([h_left_slots.flatten(1), h_right_slots.flatten(1), emotion_tensor, target_slice], dim=-1)
        brain_state_tensor = self.brain_proj(concat)
        if not torch.isfinite(brain_state_tensor).all():
            brain_state_tensor = torch.zeros_like(brain_state_tensor)
        return brain_state_tensor

    def select_action(self, brain_state_tensor: torch.Tensor, epsilon: float) -> tuple[str, torch.Tensor]:
        """Compute Q-values and sample an action."""
        if self.q_network is None:
            raise RuntimeError("Q-network not initialized.")
        q_values = self.q_network(brain_state_tensor)
        action = self.policy_head.select(q_values.squeeze(0), epsilon)
        if self.is_sleeping and action != "sleep":
            self.is_sleeping = False
        elif action == "sleep":
            self.is_sleeping = True
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
            ax = float(pose.get("x", 0))
            ay = float(pose.get("y", 0))
            pos = [ax / max(1, self.grid_size), ay / max(1, self.grid_size)]
            facing = pose.get("facing", "up")
            angle_map = {"up": math.pi / 2, "down": -math.pi / 2, "left": math.pi, "right": 0.0, "stay": 0.0}
            ang = angle_map.get(facing, 0.0)
            step_norm = float(observation.get("step_count", 0)) / max(1, self.max_steps)
            task_id = observation.get("task_id", "goto_mirror")
            task_flag = 1.0 if task_id == "goto_mirror" else 0.0
            self_feat = torch.tensor(
                [pad_feat([pos[0], pos[1], math.sin(ang), math.cos(ang), step_norm, task_flag])], device=self.device
            )

            # Objects: use allocentric positions converted to ego-relative.
            objects_obs = observation.get("objects", [])
            obj_rows: List[List[float]] = []
            for obj in objects_obs[: self.max_objects]:
                ox, oy = obj.get("position", (0.0, 0.0))
                rel_x, rel_y = ox - ax, oy - ay
                obj_rows.append(pad_feat([rel_x, rel_y, 0.5, 1.0, 1.0]))
            while len(obj_rows) < self.max_objects:
                obj_rows.append(pad_feat([0.0, 0.0, 0.0, 0.0, 0.0]))
            obj_feats = torch.tensor([obj_rows], dtype=self.backend.float_dtype, device=self.device)

            # Peers absent in gridworld.
            peer_feats = torch.zeros(1, self.max_peers, feat_dim, device=self.device)

            # Mirror reflection: use mirror position if available.
            refl_rows: List[List[float]] = []
            mirror_pos = observation.get("mirror", {}).get("position")
            if mirror_pos:
                mx, my = mirror_pos
                refl_rows.append(pad_feat([mx - ax, my - ay, 0.0]))
            while len(refl_rows) < self.max_reflections:
                refl_rows.append(pad_feat([0.0, 0.0, 0.0]))
            refl_feats = torch.tensor([refl_rows], dtype=self.backend.float_dtype, device=self.device)

            return {"self": self_feat, "objects": obj_feats, "peers": peer_feats, "reflections": refl_feats}

        self_info = observation.get("self", {})
        pos = self_info.get("pos", [0.0, 0.0])
        orientation = float(self_info.get("head_orientation", self_info.get("orientation", 0.0)))
        vel = self_info.get("velocity", [0.0, 0.0])
        body = self_info.get("body", {})
        body_vec = [
            body.get("move_scale", 1.0),
            body.get("turn_scale", 1.0),
            body.get("noise_scale", 0.0),
        ]
        time_info = observation.get("time", {})
        step_norm = float(time_info.get("episode_step_norm", 0.0))
        world_time_norm = float(time_info.get("world_time_norm", 0.0))
        self_feat = torch.tensor(
            [
                pad_feat(
                    [
                        pos[0],
                        pos[1],
                        math.sin(orientation),
                        math.cos(orientation),
                        vel[0],
                        vel[1],
                        step_norm,
                        world_time_norm,
                        body_vec[0],
                        body_vec[1],
                        body_vec[2],
                    ]
                )
            ],
            device=self.device,
        )

        def build_tensor(items: List[Dict[str, float]], fields: List[str], pad_len: int) -> torch.Tensor:
            rows: List[List[float]] = []
            for item in items[:pad_len]:
                rows.append(pad_feat([float(item.get(f, 0.0)) for f in fields]))
            while len(rows) < pad_len:
                rows.append(pad_feat([0.0 for _ in fields]))
            return torch.tensor([rows], dtype=self.backend.float_dtype, device=self.device)

        objects_raw: List[Dict[str, float]] = list(observation.get("objects", []))
        screens_raw: List[Dict[str, float]] = list(observation.get("screens", []))
        objects_obs: List[Dict[str, float]] = []
        # Prioritise screens first up to max_objects, then fill remaining slots with objects.
        for scr in screens_raw:
            if len(objects_obs) >= self.max_objects:
                break
            objects_obs.append(
                {
                    "rel_x": scr.get("rel_x", 0.0),
                    "rel_y": scr.get("rel_y", 0.0),
                    "size": scr.get("size", 0.0),
                    "visible": scr.get("visible", 1.0),
                    "type_id": scr.get("content_id", 0.0) + scr.get("brightness", 0.0),
                }
            )
        for obj in objects_raw:
            if len(objects_obs) >= self.max_objects:
                break
            objects_obs.append(obj)

        obj_feats = build_tensor(objects_obs, ["rel_x", "rel_y", "size", "visible", "type_id"], self.max_objects)
        peer_feats = build_tensor(
            observation.get("peers", []), ["rel_x", "rel_y", "orientation", "expression"], self.max_peers
        )
        refl_feats = build_tensor(
            observation.get("mirror_reflections", []), ["rel_x", "rel_y", "orientation"], self.max_reflections
        )
        target_obs = observation.get("target") or {}
        target_feat = torch.tensor(
            [[float(target_obs.get("rel_x", 0.0)), float(target_obs.get("rel_y", 0.0)), float(target_obs.get("abs_x", 0.0)), float(target_obs.get("abs_y", 0.0))]],
            dtype=self.backend.float_dtype,
            device=self.device,
        )

        return {"self": self_feat, "objects": obj_feats, "peers": peer_feats, "reflections": refl_feats, "target": target_feat}

    def _pick_gaze_target(self, observation: Dict[str, Any]) -> str | None:
        peers = observation.get("peers", [])
        reflections = observation.get("mirror_reflections", [])
        if peers and any(p.get("rel_x", 0.0) != 0.0 or p.get("rel_y", 0.0) != 0.0 for p in peers):
            return "peer"
        if reflections and any(r.get("rel_x", 0.0) != 0.0 or r.get("rel_y", 0.0) != 0.0 for r in reflections):
            return "mirror"
        return None
