"""Lightweight 2D grid environment for Stage 0."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Action = str
Position = Tuple[int, int]


@dataclass
class GridWorldConfig:
    """Configuration for the gridworld."""

    width: int = 8
    height: int = 8
    patch_radius: int = 1
    max_steps: int = 50
    num_objects: int = 3
    mirror_position: Optional[Position] = None
    seed: Optional[int] = None
    step_penalty: float = -0.01
    object_reward: float = 1.0
    mirror_reward: float = 0.05

    def __post_init__(self) -> None:
        if self.width <= 2 or self.height <= 2:
            raise ValueError("Grid must be larger than 2x2.")
        if self.patch_radius < 0:
            raise ValueError("patch_radius must be non-negative.")


@dataclass
class Observation:
    """Structured observation returned by the environment."""

    ego_patch: List[List[str]]
    agent_pose: Dict[str, object]
    objects: List[Dict[str, object]]
    mirror: Dict[str, object]
    step_count: int


class GridWorldEnv:
    """Simple gridworld that can run headless."""

    ACTIONS: Sequence[Action] = ("up", "down", "left", "right", "stay")
    EMPTY = "."
    WALL = "#"
    AGENT = "A"
    MIRROR = "M"
    OBJECT = "O"

    def __init__(self, config: GridWorldConfig):
        self.cfg = config
        self.rng = random.Random(config.seed)
        self.width = config.width
        self.height = config.height
        self.patch_radius = config.patch_radius
        self.max_steps = config.max_steps

        self.agent_pos: Position = (0, 0)
        self.facing: Action = "up"
        self.objects: List[Position] = []
        self.mirror_pos: Position = (
            config.mirror_position
            if config.mirror_position is not None
            else (self.width - 2, self.height - 2)
        )
        self.steps: int = 0
        self.reset()

    def reset(self) -> Observation:
        """Reset world to a starting state."""
        self.steps = 0
        self.facing = "up"
        self.agent_pos = (self.width // 2, self.height // 2)
        self.objects = self._spawn_objects()
        return self._observe()

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, object]]:
        """Advance one step in the grid."""
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}. Valid: {self.ACTIONS}")

        self.steps += 1
        self._move_agent(action)
        reward = self.cfg.step_penalty
        done = False
        info: Dict[str, object] = {}

        if self.agent_pos in self.objects:
            self.objects.remove(self.agent_pos)
            reward += self.cfg.object_reward
            info["collected_object"] = True

        if self.agent_pos == self.mirror_pos:
            info["mirror_contact"] = True
            reward += self.cfg.mirror_reward

        if self.steps >= self.max_steps or not self.objects:
            done = True

        obs = self._observe()
        return obs, reward, done, info

    def _spawn_objects(self) -> List[Position]:
        """Place reward objects away from the agent and mirror."""
        positions: List[Position] = []
        attempts = 0
        required = max(0, self.cfg.num_objects)
        while len(positions) < required and attempts < 1000:
            attempts += 1
            pos = (self.rng.randrange(1, self.width - 1), self.rng.randrange(1, self.height - 1))
            if pos in positions or pos == self.agent_pos or pos == self.mirror_pos:
                continue
            positions.append(pos)
        return positions

    def _move_agent(self, action: Action) -> None:
        """Move the agent if within bounds; always update facing."""
        dx, dy = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
            "stay": (0, 0),
        }[action]
        self.facing = action if action != "stay" else self.facing

        new_x = max(0, min(self.width - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.height - 1, self.agent_pos[1] + dy))
        self.agent_pos = (new_x, new_y)

    def _observe(self) -> Observation:
        """Return egocentric and allocentric features."""
        ego_patch = self._local_patch()
        objects_payload = [{"type": "reward", "position": pos} for pos in self.objects]
        mirror_payload = {"position": self.mirror_pos}
        pose = {"x": self.agent_pos[0], "y": self.agent_pos[1], "facing": self.facing}
        return Observation(
            ego_patch=ego_patch,
            agent_pose=pose,
            objects=objects_payload,
            mirror=mirror_payload,
            step_count=self.steps,
        )

    def _local_patch(self) -> List[List[str]]:
        """Return a small grid centered on the agent."""
        patch: List[List[str]] = []
        r = self.patch_radius
        cx, cy = self.agent_pos
        for dy in range(-r, r + 1):
            row: List[str] = []
            for dx in range(-r, r + 1):
                x, y = cx + dx, cy + dy
                if not (0 <= x < self.width and 0 <= y < self.height):
                    row.append(self.WALL)
                else:
                    row.append(self._token_at((x, y)))
            patch.append(row)
        return patch

    def _token_at(self, pos: Position) -> str:
        """Return a symbolic token for a cell."""
        if pos == self.agent_pos:
            return self.AGENT
        if pos == self.mirror_pos:
            return self.MIRROR
        if pos in self.objects:
            return self.OBJECT
        return self.EMPTY

    def render_ascii(self) -> str:
        """Render the full grid as ASCII."""
        rows: List[str] = []
        for y in range(self.height):
            row_cells: List[str] = []
            for x in range(self.width):
                row_cells.append(self._token_at((x, y)))
            rows.append("".join(row_cells))
        return "\n".join(rows)

    @property
    def action_space(self) -> Sequence[Action]:
        """Expose available actions."""
        return self.ACTIONS
