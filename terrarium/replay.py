"""Simple replay buffer for storing transitions."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Transition:
    observation: Dict[str, Any]
    action: str
    action_idx: int
    reward: float
    next_observation: Dict[str, Any]
    done: bool
    brain_state: Sequence[float]
    next_brain_state: Sequence[float]
    emotion_latent: Sequence[float]
    next_emotion_latent: Optional[Sequence[float]] = None
    hidden_left: Sequence[float] = ()
    hidden_right: Sequence[float] = ()
    hidden_left_in: Sequence[float] = ()
    hidden_right_in: Sequence[float] = ()
    next_hidden_left: Sequence[float] = ()
    next_hidden_right: Sequence[float] = ()
    next_hidden_left_in: Sequence[float] = ()
    next_hidden_right_in: Sequence[float] = ()
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
        self.priorities: List[float] = []
        self.position = 0
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, transition: Transition) -> None:
        """Store a transition, overwriting old entries when full."""
        if len(self.storage) < self.capacity:
            self.storage.append(transition)
            self.priorities.append(max(transition.priority, 1e-3))
        else:
            self.storage[self.position] = transition
            self.priorities[self.position] = max(transition.priority, 1e-3)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample uniformly for now."""
        if batch_size <= 0:
            return []
        return self.rng.sample(self.storage, min(batch_size, len(self.storage)))

    def sample_prioritized(
        self, batch_size: int, alpha: float = 0.6, beta: float = 0.4
    ) -> tuple[List[Transition], List[float], List[int]]:
        """Sample transitions proportionally to priority."""
        if not self.storage:
            return [], [], []
        scaled = [p**alpha for p in self.priorities]
        total = sum(scaled)
        probs = [s / total for s in scaled]
        k = min(batch_size, len(self.storage))
        indices = self.rng.choices(range(len(self.storage)), weights=probs, k=k)
        samples = [self.storage[i] for i in indices]
        if beta > 0:
            weights = [(1 / (len(self.storage) * probs[i])) ** beta for i in indices]
            max_w = max(weights) if weights else 1.0
            weights = [w / max_w for w in weights]
        else:
            weights = [1.0] * k
        return samples, weights, indices

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update stored priorities after learning step."""
        for idx, prio in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = max(prio, 1e-3)
