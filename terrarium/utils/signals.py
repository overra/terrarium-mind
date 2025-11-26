"""Signal utilities for novelty and prediction error estimation."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def compute_novelty(current_obs: Dict[str, Any], previous_obs: Dict[str, Any] | None) -> float:
    """Very small novelty heuristic based on ego_patch changes."""
    if previous_obs is None:
        return 1.0
    current_patch = _flatten_patch(current_obs.get("ego_patch", []))
    prev_patch = _flatten_patch(previous_obs.get("ego_patch", []))
    if not current_patch or not prev_patch or len(current_patch) != len(prev_patch):
        return 0.5
    differences = sum(1 for a, b in zip(current_patch, prev_patch) if a != b)
    return min(1.0, differences / len(current_patch))


def compute_prediction_error(reward: float, expected_reward: float) -> float:
    """Absolute reward surprise."""
    return abs(reward - expected_reward)


def _flatten_patch(patch: Sequence[Sequence[str]]) -> List[str]:
    flat: List[str] = []
    for row in patch:
        flat.extend(row)
    return flat
