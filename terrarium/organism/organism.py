"""Organism scaffold with split cores, emotion, and expression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

from terrarium.backend import TorchBackend

from .cores import Bridge, WorldCore
from .emotion import EmotionEngine, EmotionState
from .expression import ExpressionHead
from .policy import PolicyHead


@dataclass
class OrganismOutputs:
    """Bundle of outputs for a single decision step."""

    action: str
    emotion: EmotionState
    drives: Dict[str, float]
    core_affect: Dict[str, float]
    expression: Dict[str, Any]
    core_readouts: Dict[str, Any]
    emotion_latent_tensor: Any


class Organism:
    """Top-level container for Stage 0 organism components."""

    def __init__(self, action_space: Sequence[str], backend: TorchBackend | None = None) -> None:
        self.action_space = action_space
        self.backend = backend or TorchBackend()
        self.left_core = WorldCore("left")
        self.right_core = WorldCore("right")
        self.bridge = Bridge()
        self.emotion_engine = EmotionEngine()
        self.policy_head = PolicyHead()
        self.expression_head = ExpressionHead()

    def reset(self) -> None:
        """Reset internal states."""
        self.emotion_engine.reset()

    def step(
        self,
        observation: Dict[str, Any],
        reward: float,
        novelty: float,
        prediction_error: float,
        info: Dict[str, Any],
    ) -> OrganismOutputs:
        """Process observation and return chosen action plus internal signals."""
        left_state = self.left_core.process(observation)
        right_state = self.right_core.process(observation)
        bridge_state = self.bridge.exchange(left_state, right_state)
        emotion_state = self.emotion_engine.update(
            reward=reward,
            novelty=novelty,
            prediction_error=prediction_error,
            mirror_contact=bool(info.get("mirror_contact", False)),
        )
        latent_tensor = self.backend.tensor(emotion_state.latent, dtype=self.backend.float_dtype)
        action = self.policy_head.select_action(observation, emotion_state.latent, self.action_space)
        facing = observation.get("agent_pose", {}).get("facing", "up")
        expression = self.expression_head.generate(emotion_state.latent, facing)

        core_readouts = {
            "left": left_state.summary,
            "right": right_state.summary,
            "bridge": bridge_state,
        }
        return OrganismOutputs(
            action=action,
            emotion=emotion_state,
            drives=self.emotion_engine.drives_dict(),
            core_affect=self.emotion_engine.core_affect_dict(),
            expression=expression,
            core_readouts=core_readouts,
            emotion_latent_tensor=latent_tensor,
        )
