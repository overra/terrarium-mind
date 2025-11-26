"""Signal utilities for novelty and prediction error estimation."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


def compute_novelty(current_obs: Dict[str, Any], previous_obs: Dict[str, Any] | None) -> float:
    """Very small novelty heuristic based on ego_patch changes."""
    if previous_obs is None:
        return 1.0
    # Stage 2: structured obs with positions; use L2 distance of self + objects/peers.
    if "self" in current_obs:
        cur_self = current_obs.get("self", {}).get("pos", [0.0, 0.0])
        prev_self = previous_obs.get("self", {}).get("pos", [0.0, 0.0])
        ds = ( (cur_self[0]-prev_self[0])**2 + (cur_self[1]-prev_self[1])**2 ) ** 0.5
        def _positions(obs: Dict[str, Any], key: str):
            return [(e.get("rel_x",0.0), e.get("rel_y",0.0)) for e in obs.get(key, [])]
        cur_objs = _positions(current_obs, "objects")
        prev_objs = _positions(previous_obs, "objects")
        delta_objs = sum(( (cx-px)**2 + (cy-py)**2 )**0.5 for (cx,cy),(px,py) in zip(cur_objs, prev_objs[:len(cur_objs)])) if prev_objs else 0.0
        cur_peers = _positions(current_obs, "peers")
        prev_peers = _positions(previous_obs, "peers")
        delta_peers = sum(( (cx-px)**2 + (cy-py)**2 )**0.5 for (cx,cy),(px,py) in zip(cur_peers, prev_peers[:len(cur_peers)])) if prev_peers else 0.0
        novelty = ds + 0.5*delta_objs + 0.5*delta_peers
        return max(0.0, min(1.0, novelty))
    # Stage 1 grid fallback
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
