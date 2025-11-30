"""Homeostatic intrinsic reward utilities."""

from __future__ import annotations

import numpy as np

# Target bands for E_t components
HOMEOSTASIS_TARGETS = {
    "tiredness": {"low": 0.1, "high": 0.4, "idx": 2},
    "social_satiation": {"low": 0.5, "high": 1.0, "idx": 3},
    "sleep_urge": {"low": 0.0, "high": 0.5, "idx": 6},
    "confusion_level": {"low": 0.0, "high": 0.4, "idx": 7},
}


def compute_homeostasis_reward(E_t: np.ndarray) -> float:
    """Compute intrinsic reward from distance to target bands.

    For each band:
      inside band  -> +1 scaled toward center
      outside band -> negative proportional to distance outside
    Returns mean score across bands.
    """
    scores = []
    for cfg in HOMEOSTASIS_TARGETS.values():
        idx = cfg["idx"]
        val = float(E_t[idx])
        low, high = cfg["low"], cfg["high"]
        if val < low:
            score = - (low - val)
        elif val > high:
            score = - (val - high)
        else:
            center = (low + high) / 2.0
            half = (high - low) / 2.0
            score = 1.0 - abs(val - center) / max(half, 1e-6)
        scores.append(score)
    return float(np.mean(scores)) if scores else 0.0


class HomeostasisTracker:
    """Track long-term overload for penalties."""

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self.tired_long = 0.0
        self.confuse_long = 0.0

    def update(self, tiredness: float, confusion: float) -> None:
        self.tired_long = self.decay * self.tired_long + (1 - self.decay) * tiredness
        self.confuse_long = self.decay * self.confuse_long + (1 - self.decay) * confusion

    def overload_penalty(self, tired_thresh: float = 0.8, confuse_thresh: float = 0.7, penalty: float = 0.01) -> float:
        p = 0.0
        if self.tired_long > tired_thresh:
            p -= penalty
        if self.confuse_long > confuse_thresh:
            p -= penalty
        return p
