"""Simple salient episodic memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import numpy as np


@dataclass
class MemoryEntry:
    core_summary: np.ndarray
    emotion: np.ndarray
    task_id: str
    reward: float
    confusion: float
    timestamp: int
    info: Dict[str, Any]


class SalientMemory:
    def __init__(self, capacity: int = 256, salience_threshold: float = 0.5):
        self.capacity = capacity
        self.salience_threshold = salience_threshold
        self.entries: List[MemoryEntry] = []

    def consider(
        self,
        core_summary: np.ndarray,
        emotion: np.ndarray,
        reward: float,
        confusion: float,
        task_id: str,
        timestamp: int,
        info: Dict[str, Any],
    ) -> None:
        salience = max(abs(reward), confusion, float(np.abs(emotion).max()))
        if salience < self.salience_threshold:
            return
        if len(self.entries) >= self.capacity:
            self.entries.pop(0)
        self.entries.append(
            MemoryEntry(
                core_summary=np.array(core_summary, dtype=np.float32),
                emotion=np.array(emotion, dtype=np.float32),
                task_id=task_id,
                reward=reward,
                confusion=confusion,
                timestamp=timestamp,
                info=info,
            )
        )

    def sample(self, k: int) -> List[MemoryEntry]:
        if k <= 0:
            return []
        if k >= len(self.entries):
            return list(self.entries)
        idx = np.random.choice(len(self.entries), size=k, replace=False)
        return [self.entries[i] for i in idx]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "size": len(self.entries),
            "by_task": {t: sum(1 for e in self.entries if e.task_id == t) for t in set(e.task_id for e in self.entries)},
        }
