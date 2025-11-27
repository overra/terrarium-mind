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
    novelty_drive: float = 0.5
    self_reflection_drive: float = 0.5
    sleep_drive: float = 0.2


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
        intero_signals: Dict[str, float] | None = None,
    ) -> EmotionState:
        """Update drives and affect based on interoceptive cues."""
        drives = self.state.drives
        affect = self.state.core_affect
        intero_signals = intero_signals or {}
        energy = float(intero_signals.get("energy", 1.0))
        fatigue = float(intero_signals.get("fatigue", 0.0))
        ts_reward = float(intero_signals.get("time_since_reward", 0.0))
        ts_social = float(intero_signals.get("time_since_social_contact", 0.0))
        ts_reflection = float(intero_signals.get("time_since_reflection", 0.0))
        ts_sleep = float(intero_signals.get("time_since_sleep", 0.0))

        # Passive drift toward mild depletion.
        drives.social_hunger = _clamp(drives.social_hunger + self.drive_decay)
        drives.curiosity_drive = _clamp(drives.curiosity_drive - self.drive_decay * 0.5)
        drives.safety_drive = _clamp(drives.safety_drive - self.drive_decay * 0.25)
        drives.rest_drive = _clamp(drives.rest_drive + self.drive_decay * 0.2)
        drives.novelty_drive = _clamp(drives.novelty_drive - self.drive_decay * 0.2)
        drives.self_reflection_drive = _clamp(drives.self_reflection_drive + self.drive_decay * 0.3)
        drives.sleep_drive = _clamp(drives.sleep_drive + self.drive_decay * 0.2)

        # Social contact reduces social hunger.
        if mirror_contact:
            drives.social_hunger = _clamp(drives.social_hunger - 0.1)
            drives.self_reflection_drive = _clamp(drives.self_reflection_drive - 0.2)

        # Novel stimuli nudge curiosity.
        drives.curiosity_drive = _clamp(drives.curiosity_drive + 0.5 * novelty)
        drives.novelty_drive = _clamp(drives.novelty_drive + 0.6 * novelty)

        # Time-based modulation.
        drives.social_hunger = _clamp(drives.social_hunger + 0.2 * ts_social)
        drives.self_reflection_drive = _clamp(drives.self_reflection_drive + 0.2 * ts_reflection)
        drives.novelty_drive = _clamp(drives.novelty_drive + 0.2 * ts_reward)
        drives.sleep_drive = _clamp(drives.sleep_drive + 0.5 * ts_sleep)

        # Energy/fatigue effects.
        drives.rest_drive = _clamp(drives.rest_drive + max(0.0, 0.5 - energy) + fatigue * 0.2)
        drives.curiosity_drive = _clamp(drives.curiosity_drive - fatigue * 0.1)
        drives.sleep_drive = _clamp(drives.sleep_drive + fatigue * 0.3 + max(0.0, 0.5 - energy) * 0.6)

        # Prediction error reduces safety sense.
        drives.safety_drive = _clamp(drives.safety_drive - 0.3 * prediction_error)

        # Reward improves valence; error reduces it.
        reward_scaled = max(-1.0, min(1.0, reward))
        affect.valence = _clamp(
            0.85 * affect.valence + 0.5 * reward_scaled - 0.2 * prediction_error + 0.1 * (energy - fatigue - 0.5)
        )
        affect.arousal = _clamp(affect.arousal * 0.8 + 0.6 * novelty + 0.3 * abs(reward) + 0.1 * fatigue)

        # Slow-moving mood that integrates valence.
        self.state.mood = _clamp(self.mood_tau * self.state.mood + (1 - self.mood_tau) * affect.valence)

        # Compose the latent vector (keep it small and interpretable).
        latent = [
            float(affect.valence),
            float(affect.arousal),
            float(drives.curiosity_drive),
            float(drives.safety_drive),
            float(drives.novelty_drive),
            float(drives.self_reflection_drive),
            float(drives.sleep_drive),
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
            "novelty_drive": self.state.drives.novelty_drive,
            "self_reflection_drive": self.state.drives.self_reflection_drive,
            "sleep_drive": self.state.drives.sleep_drive,
        }

    def drives_dict(self) -> Dict[str, float]:
        """Return drives only."""
        return {
            "social_hunger": self.state.drives.social_hunger,
            "curiosity_drive": self.state.drives.curiosity_drive,
            "safety_drive": self.state.drives.safety_drive,
            "rest_drive": self.state.drives.rest_drive,
            "novelty_drive": self.state.drives.novelty_drive,
            "self_reflection_drive": self.state.drives.self_reflection_drive,
            "sleep_drive": self.state.drives.sleep_drive,
        }

    def core_affect_dict(self) -> Dict[str, float]:
        """Return valence/arousal only."""
        return {
            "valence": self.state.core_affect.valence,
            "arousal": self.state.core_affect.arousal,
            "mood": self.state.mood,
        }
