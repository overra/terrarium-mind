"""Simple metabolic core tracking energy and fatigue."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetabolicState:
    energy: float
    fatigue: float
    age_steps: int


class MetabolicCore:
    """Tracks a tiny energy/fatigue budget."""

    def __init__(self, base_cost: float = 0.005, recovery: float = 0.002, k_action: float = 0.01, k_arousal: float = 0.005, k_learning: float = 0.002):
        self.base_cost = base_cost
        self.recovery = recovery
        self.k_action = k_action
        self.k_arousal = k_arousal
        self.k_learning = k_learning
        self.state = MetabolicState(energy=0.9, fatigue=0.1, age_steps=0)

    def reset(self) -> MetabolicState:
        self.state = MetabolicState(energy=0.9, fatigue=0.1, age_steps=0)
        return self.state

    def step(self, action_cost: float, arousal: float, learning_load: float, is_sleeping: bool = False, sleep_recovery: float = 0.01, sleep_rest: float = 0.01) -> MetabolicState:
        s = self.state
        s.age_steps += 1
        if is_sleeping:
            s.energy = min(1.0, s.energy + sleep_recovery)
            s.fatigue = max(0.0, s.fatigue - sleep_rest)
        else:
            cost = self.base_cost + self.k_action * action_cost + self.k_arousal * abs(arousal) + self.k_learning * learning_load
            s.energy = max(0.0, s.energy - cost)
            s.fatigue = min(1.0, s.fatigue + cost * 0.5)
            s.energy = min(1.0, s.energy + self.recovery)
        self.state = s
        return self.state
