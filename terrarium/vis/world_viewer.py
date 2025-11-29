"""Top-down world rendering helpers."""

from __future__ import annotations

from typing import List

import numpy as np


def render_snapshot_topdown(snapshot: dict, size: int = 128) -> np.ndarray:
    """Render a simple top-down RGB image from a world snapshot."""
    img = np.ones((size, size, 3), dtype=np.float32)
    img *= 0.9
    world_size = float(snapshot.get("world_size", 8.0))

    def to_px(pos):
        x = int(pos[0] / world_size * (size - 1))
        y = int(pos[1] / world_size * (size - 1))
        return x, y

    # Mirrors
    for m in snapshot.get("mirrors", []):
        x = to_px([m["p1"][0], 0])[0]
        img[:, x : x + 1, :] = np.array([1.0, 1.0, 0.3])

    # Screens
    for s in snapshot.get("screens", []):
        x, y = to_px(s["pos"])
        size_px = int(s["size"][0] / world_size * size)
        img[max(0, y - size_px) : min(size, y + size_px), max(0, x - size_px) : min(size, x + size_px), :] = np.array(
            [0.7, 0.2, 0.9]
        )

    # Objects
    for o in snapshot.get("objects", []):
        x, y = to_px(o["pos"])
        size_px = int(o["size"][0] / world_size * size)
        img[max(0, y - size_px) : min(size, y + size_px), max(0, x - size_px) : min(size, x + size_px), :] = 0.5

    # Peers
    for p in snapshot.get("peers", []):
        x, y = to_px(p["pos"])
        img[max(0, y - 2) : min(size, y + 2), max(0, x - 2) : min(size, x + 2), :] = np.array([0.2, 0.8, 0.2])

    # Agent
    for a in snapshot.get("agents", []):
        x, y = to_px(a["pos"])
        img[max(0, y - 3) : min(size, y + 3), max(0, x - 3) : min(size, x + 3), :] = np.array([0.2, 0.2, 1.0])
    return (img * 255).astype(np.uint8)


def replay_episode_topdown(snapshots: List[dict], size: int = 128) -> List[np.ndarray]:
    """Render a sequence of snapshots into frames."""
    return [render_snapshot_topdown(s, size=size) for s in snapshots]
