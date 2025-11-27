"""Plasticity controller stub for prioritizing experience."""

from __future__ import annotations

from typing import Sequence


class PlasticityController:
    """Assigns priorities to transitions and could gate learning later."""

    def __init__(
        self,
        novelty_weight: float = 0.4,
        error_weight: float = 0.4,
        reward_weight: float = 0.2,
        arousal_weight: float = 0.1,
    ):
        self.novelty_weight = novelty_weight
        self.error_weight = error_weight
        self.reward_weight = reward_weight
        self.arousal_weight = arousal_weight

    def compute_priority(
        self,
        reward: float,
        novelty: float,
        prediction_error: float,
        emotion_latent: Sequence[float],
        core_affect: dict | None = None,
    ) -> float:
        """Combine basic interoceptive signals into a scalar priority."""
        valence = emotion_latent[0] if emotion_latent else 0.0
        arousal = core_affect["arousal"] if core_affect else (emotion_latent[1] if len(emotion_latent) > 1 else 0.0)
        base = self.reward_weight * reward + self.novelty_weight * novelty + self.error_weight * prediction_error
        # Prioritize surprising positive moments or safety-relevant errors, modulated by arousal.
        shaped = base + 0.1 * valence + 0.05 * abs(prediction_error) + self.arousal_weight * arousal
        return max(0.05, shaped)

    def should_update_online(self) -> bool:
        """Placeholder for gating online learning."""
        return False
