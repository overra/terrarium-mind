"""World wrapper around the Stage 2 environment."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from terrarium.env.world import Stage2Env


class World:
    """Wraps the environment and tracks world time."""

    def __init__(self, env: Stage2Env):
        self.env = env
        self.cfg = env.cfg
        self.world_time = 0
        self.episode_step = 0

    def reset(self, task_id: Optional[str] = None) -> Dict[str, object]:
        self.episode_step = 0
        obs = self.env.reset(task_id=task_id)
        return self._augment_obs(obs)

    def step(self, actions) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        self.world_time += 1
        self.episode_step += 1
        if isinstance(actions, dict):
            action = actions.get("agent", "stay")
        else:
            action = actions
        obs, reward, done, info = self.env.step(action)
        obs = self._augment_obs(obs)
        return obs, reward, done, info

    def _augment_obs(self, obs: Dict[str, object]) -> Dict[str, object]:
        obs = dict(obs)
        obs["time"] = {
            "episode_step_norm": self.episode_step / max(1, self.env.cfg.max_steps),
            "world_time_norm": (self.world_time % 10000) / 10000.0,
        }
        return obs

    def get_snapshot(self) -> Dict[str, object]:
        """Minimal snapshot for viewers (placeholder)."""
        return {
            "agent": {"pos": [self.env.agent.x, self.env.agent.y], "orientation": self.env.agent.orientation},
            "objects": [{"pos": [o.x, o.y], "size": o.size, "type": o.type_id} for o in self.env.objects],
            "peers": [{"pos": [p.x, p.y], "orientation": p.orientation, "expression": p.expression} for p in self.env.peers],
            "mirrors": [{"x": m.x} for m in self.env.mirrors],
            "time": {"world_time": self.world_time, "episode_step": self.episode_step},
        }
