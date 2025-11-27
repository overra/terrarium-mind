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
        action = actions
        if isinstance(actions, dict):
            # pick the first (single) agent action if provided as dict
            action = list(actions.values())[0] if actions else "stay"
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

    def get_snapshot(self, agent_status: Dict[str, dict] | None = None) -> Dict[str, object]:
        """Snapshot for viewers/teachers; agent_status can enrich emotion/expression."""
        agent_camera = {
            "pos": [self.env.agent.x, self.env.agent.y, 1.0],
            "orientation": [0.0, 0.0, self.env.agent.orientation],
            "fov_deg": 60.0,
            "near": 0.1,
            "far": 15.0,
        }
        base_agent = {
            "id": "agent-1",
            "pos": [self.env.agent.x, self.env.agent.y],
            "orientation": self.env.agent.orientation,
            "velocity": [0.0, 0.0],
            "expression": {},
            "emotion": {},
            "task_state": {"task_id": getattr(self.env, "task_id", None), "task_success": False},
            "camera": agent_camera,
        }
        if agent_status and "agent-1" in agent_status:
            status = agent_status["agent-1"]
            base_agent["expression"] = status.get("expression", {})
            base_agent["emotion"] = status.get("emotion", {})
            base_agent["task_state"]["task_id"] = status.get("task_id", base_agent["task_state"]["task_id"])
            base_agent["task_state"]["task_success"] = status.get("task_success", False)
            base_agent["sleeping"] = status.get("sleeping", False)

        snapshot = {
            "t": self.world_time,
            "episode_step": self.episode_step,
            "agents": [base_agent],
            "peers": [
                {"id": f"peer-{i}", "pos": [p.x, p.y], "orientation": p.orientation, "velocity": [0.0, 0.0], "expression": p.expression}
                for i, p in enumerate(self.env.peers)
            ],
            "objects": [
                {"id": f"obj-{i}", "type": o.type_id, "pos": [o.x, o.y], "size": [o.size, o.size], "state": {"seen": getattr(o, 'seen', False)}}
                for i, o in enumerate(self.env.objects)
            ],
            "mirrors": [{"id": f"mirror-{i}", "p1": [m.x, 0.0], "p2": [m.x, self.env.cfg.world_size]} for i, m in enumerate(self.env.mirrors)],
        }
        return snapshot
