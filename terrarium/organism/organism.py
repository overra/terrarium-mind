"""Organism with neural hemisphere cores, bridge, and Q-policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
from copy import deepcopy

from terrarium.backend import TorchBackend

from .cores import Bridge
from .emotion import EmotionEngine, EmotionState
from .expression import ExpressionHead
from .hemisphere_core import HemisphereCore
from .policy import EpsilonGreedyPolicy
from .q_network import QNetwork
from typing import Iterable


@dataclass
class EncodedState:
    """Representation of the organism state for RL."""

    brain_state_tensor: torch.Tensor
    brain_state: List[float]
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

        self.emotion_engine = EmotionEngine()
        self.expression_head = ExpressionHead()
        self.policy_head = EpsilonGreedyPolicy(self.action_space)

        self.left_core: HemisphereCore | None = None
        self.right_core: HemisphereCore | None = None
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
            input_dim = obs_dim + 1 + emotion_dim  # obs + side indicator + emotion latent
            self.left_core = HemisphereCore(input_dim, self.hidden_dim).to(self.device)
            self.right_core = HemisphereCore(input_dim, self.hidden_dim).to(self.device)
            self.bridge = Bridge(self.hidden_dim, self.bridge_dim).to(self.device)
            brain_dim = self.hidden_dim * 2 + emotion_dim
            self.q_network = QNetwork(brain_dim, self.hidden_dim, len(self.action_space)).to(self.device)
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
        obs_tensor = self._obs_to_tensor(observation)
        self._ensure_modules(obs_tensor.shape[-1], emotion_tensor.shape[-1])

        if self.hidden_left is None:
            self.hidden_left = torch.zeros(1, self.hidden_dim, device=self.device)
            self.hidden_right = torch.zeros(1, self.hidden_dim, device=self.device)

        obs_left = torch.cat([obs_tensor, torch.zeros((1, 1), device=self.device)], dim=-1)
        obs_right = torch.cat([obs_tensor, torch.ones((1, 1), device=self.device)], dim=-1)

        h_left = self.left_core(obs_left, emotion_tensor, self.hidden_left)
        h_right = self.right_core(obs_right, emotion_tensor, self.hidden_right)
        h_left, h_right = self.bridge(h_left, h_right)  # type: ignore[arg-type]

        self.hidden_left = h_left.detach()
        self.hidden_right = h_right.detach()

        brain_state_tensor = torch.cat([h_left, h_right, emotion_tensor], dim=-1).detach()
        brain_state = brain_state_tensor.squeeze(0).cpu().tolist()

        facing = observation.get("agent_pose", {}).get("facing", "up")
        expression = self.expression_head.generate(emotion_state.latent, facing)

        return EncodedState(
            brain_state_tensor=brain_state_tensor,
            brain_state=brain_state,
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
        if self.q_network is not None:
            params += list(self.q_network.parameters())
        return params

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

    def _obs_to_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        """Flatten ego patch and auxiliary features into a tensor."""
        patch = observation.get("ego_patch", [])
        token_map = {".": 0.0, "#": -1.0, "A": 0.5, "M": 1.0, "O": 0.8, "S": 0.9}
        flat_patch: List[float] = []
        for row in patch:
            for t in row:
                flat_patch.append(token_map.get(t, 0.0))

        pose = observation.get("agent_pose", {})
        x_norm = float(pose.get("x", 0)) / max(1, self.grid_size - 1)
        y_norm = float(pose.get("y", 0)) / max(1, self.grid_size - 1)
        facing = pose.get("facing", "up")
        facing_onehot = [1.0 if facing == a else 0.0 for a in ("up", "down", "left", "right", "stay")]

        step_count = observation.get("step_count", 0)
        step_norm = float(step_count) / max(1, self.max_steps)

        task_id = observation.get("task_id", "goto_mirror")
        task_onehot = [1.0 if task_id == t else 0.0 for t in self.task_ids]

        obs_vec = flat_patch + [x_norm, y_norm] + facing_onehot + [step_norm] + task_onehot
        return torch.tensor(obs_vec, dtype=self.backend.float_dtype, device=self.device).unsqueeze(0)
