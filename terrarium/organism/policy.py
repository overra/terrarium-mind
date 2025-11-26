"""Simple placeholder policy head."""

from __future__ import annotations

import random
from typing import Any, Sequence


class PolicyHead:
    """Random or heuristic policy placeholder."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def select_action(self, observation: Any, emotion_latent: Sequence[float], action_space: Sequence[str]) -> str:
        """Select an action based on observation and emotion latent (random for now)."""
        return self.rng.choice(list(action_space))
