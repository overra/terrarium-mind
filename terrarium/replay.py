"""Simple replay buffer for storing transitions."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass
class Transition:
    observation: Dict[str, Any]
    action: str
    reward: float
    next_observation: Dict[str, Any]
    done: bool
    emotion_latent: Sequence[float]
    drives: Dict[str, float]
    core_affect: Dict[str, float]
    expression: Dict[str, Any]
    novelty: float
    prediction_error: float
    priority: float
    info: Dict[str, Any]


class ReplayBuffer:
    """Ring buffer with uniform or priority-aware sampling later."""

    def __init__(self, capacity: int = 10000, seed: int | None = None) -> None:
        self.capacity = capacity
        self.storage: List[Transition] = []
        self.position = 0
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: Transition) -> None:
        """Store a transition, overwriting old entries when full."""
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
        else:
            self.storage[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample uniformly for now."""
        if batch_size <= 0:
            return []
        return self.rng.sample(self.storage, min(batch_size, len(self.storage)))
