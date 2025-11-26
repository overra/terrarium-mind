"""Interoception and emotion state tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass
class Drives:
    """Basic drive signals that will influence behavior."""

    social_hunger: float = 0.5
    curiosity_drive: float = 0.5
    safety_drive: float = 0.5
    rest_drive: float = 0.5


@dataclass
class CoreAffect:
    """Compact affective state."""

    valence: float = 0.0
    arousal: float = 0.0


@dataclass
class EmotionState:
    """Current emotional snapshot."""

    drives: Drives
    core_affect: CoreAffect
    mood: float
    latent: List[float]


class EmotionEngine:
    """Tracks interoceptive signals and produces a latent emotion vector."""

    def __init__(self, drive_decay: float = 0.01, mood_tau: float = 0.95) -> None:
        self.drive_decay = drive_decay
        self.mood_tau = mood_tau
        self.state = EmotionState(drives=Drives(), core_affect=CoreAffect(), mood=0.0, latent=[0.0] * 4)

    def reset(self) -> EmotionState:
        """Reset internal state to defaults."""
        self.state = EmotionState(drives=Drives(), core_affect=CoreAffect(), mood=0.0, latent=[0.0] * 4)
        return self.state

    def update(
        self,
        reward: float,
        novelty: float,
        prediction_error: float,
        mirror_contact: bool = False,
    ) -> EmotionState:
        """Update drives and affect based on interoceptive cues."""
        drives = self.state.drives
        affect = self.state.core_affect

        # Passive drift toward mild depletion.
        drives.social_hunger = _clamp(drives.social_hunger + self.drive_decay)
        drives.curiosity_drive = _clamp(drives.curiosity_drive - self.drive_decay * 0.5)
        drives.safety_drive = _clamp(drives.safety_drive - self.drive_decay * 0.25)
        drives.rest_drive = _clamp(drives.rest_drive + self.drive_decay * 0.2)

        # Social contact reduces social hunger.
        if mirror_contact:
            drives.social_hunger = _clamp(drives.social_hunger - 0.1)

        # Novel stimuli nudge curiosity.
        drives.curiosity_drive = _clamp(drives.curiosity_drive + 0.5 * novelty)

        # Prediction error reduces safety sense.
        drives.safety_drive = _clamp(drives.safety_drive - 0.3 * prediction_error)

        # Reward improves valence; error reduces it.
        affect.valence = _clamp(affect.valence * 0.9 + reward - 0.5 * prediction_error)
        affect.arousal = _clamp(affect.arousal * 0.8 + 0.6 * novelty + 0.4 * abs(reward))

        # Slow-moving mood that integrates valence.
        self.state.mood = _clamp(self.mood_tau * self.state.mood + (1 - self.mood_tau) * affect.valence)

        # Compose the latent vector (keep it small and interpretable).
        latent = [
            float(affect.valence),
            float(affect.arousal),
            float(drives.curiosity_drive),
            float(drives.safety_drive),
        ]

        self.state.latent = latent
        return self.state

    def to_dict(self) -> Dict[str, float]:
        """Expose a compact dict for logging or serialization."""
        return {
            "valence": self.state.core_affect.valence,
            "arousal": self.state.core_affect.arousal,
            "mood": self.state.mood,
            "social_hunger": self.state.drives.social_hunger,
            "curiosity_drive": self.state.drives.curiosity_drive,
            "safety_drive": self.state.drives.safety_drive,
            "rest_drive": self.state.drives.rest_drive,
        }

    def drives_dict(self) -> Dict[str, float]:
        """Return drives only."""
        return {
            "social_hunger": self.state.drives.social_hunger,
            "curiosity_drive": self.state.drives.curiosity_drive,
            "safety_drive": self.state.drives.safety_drive,
            "rest_drive": self.state.drives.rest_drive,
        }

    def core_affect_dict(self) -> Dict[str, float]:
        """Return valence/arousal only."""
        return {
            "valence": self.state.core_affect.valence,
            "arousal": self.state.core_affect.arousal,
            "mood": self.state.mood,
        }
