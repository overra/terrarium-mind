"""Plasticity controller stub for prioritizing experience."""

from __future__ import annotations

from typing import Sequence


class PlasticityController:
    """Assigns priorities to transitions and could gate learning later."""

    def __init__(self, novelty_weight: float = 0.6, error_weight: float = 0.3, reward_weight: float = 0.2):
        self.novelty_weight = novelty_weight
        self.error_weight = error_weight
        self.reward_weight = reward_weight

    def compute_priority(
        self,
        reward: float,
        novelty: float,
        prediction_error: float,
        emotion_latent: Sequence[float],
    ) -> float:
        """Combine basic interoceptive signals into a scalar priority."""
        valence = emotion_latent[0] if emotion_latent else 0.0
        base = self.reward_weight * reward + self.novelty_weight * novelty + self.error_weight * prediction_error
        # Prioritize surprising positive moments or safety-relevant errors.
        shaped = base + 0.1 * valence + 0.05 * abs(prediction_error)
        return max(0.0, shaped)

    def should_update_online(self) -> bool:
        """Placeholder for gating online learning."""
        return False
