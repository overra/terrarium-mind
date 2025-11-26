"""Stubbed split-brain world cores and bridge."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class WorldCoreState:
    """Lightweight representation of a world core output."""

    name: str
    summary: Dict[str, Any]


class WorldCore:
    """Placeholder for a hemisphere-specific processing block."""

    def __init__(self, name: str) -> None:
        self.name = name

    def process(self, observation: Dict[str, Any]) -> WorldCoreState:
        """Return a simple summary that could be replaced by a real encoder."""
        pose = observation.get("agent_pose", {})
        patch = observation.get("ego_patch", [])
        summary = {
            "pose": pose,
            "patch_center": patch[len(patch) // 2][len(patch[0]) // 2] if patch and patch[0] else None,
        }
        return WorldCoreState(name=self.name, summary=summary)


class Bridge:
    """Limited-bandwidth connector between the two hemispheres."""

    def exchange(self, left: WorldCoreState, right: WorldCoreState) -> Dict[str, Any]:
        """Return a compact merged view."""
        return {
            "left_summary": left.summary,
            "right_summary": right.summary,
            "callosal_signal": {
                "pose_agreement": left.summary.get("pose") == right.summary.get("pose"),
            },
        }
