"""World wrapper around the Stage 2 environment."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from terrarium.env.world import Stage2Env
import numpy as np
import math


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
        obs["retina"] = self.render_retina(grid_size=16)
        return obs

    def render_retina(self, grid_size: int = 16) -> list:
        """Render an egocentric proto-retina as a list for JSON friendliness.

        Channels (C=6):
        0: occupancy (objects|peers|mirrors|screens)
        1: objects
        2: peers
        3: mirrors
        4: screens
        5: intensity/pattern (e.g., screen brightness)
        """
        H = W = grid_size
        C = 6  # occupancy, objects, peers, mirrors, screens, intensity
        retina = np.zeros((C, H, W), dtype=np.float32)
        # coordinates in agent frame: forward = +x
        view_range = 4.0
        half_w = view_range
        xs = np.linspace(0.0, view_range, H)
        ys = np.linspace(-half_w, half_w, W)
        cos_o = math.cos(self.env.agent.orientation)
        sin_o = math.sin(self.env.agent.orientation)

        def world_pos(x_local: float, y_local: float) -> Tuple[float, float]:
            wx = self.env.agent.x + cos_o * x_local - sin_o * y_local
            wy = self.env.agent.y + sin_o * x_local + cos_o * y_local
            return wx, wy

        # Fill mirrors (as vertical lines)
        for m in self.env.mirrors:
            for i, x_local in enumerate(xs):
                for j, y_local in enumerate(ys):
                    wx, wy = world_pos(x_local, y_local)
                    if abs(wx - m.x) < 0.1:
                        retina[3, i, j] = 1.0

        # Fill screens
        for screen in self.env.screens:
            for i, x_local in enumerate(xs):
                for j, y_local in enumerate(ys):
                    wx, wy = world_pos(x_local, y_local)
                    if abs(wx - screen.x) < screen.size and abs(wy - screen.y) < screen.size:
                        retina[4, i, j] = 1.0
                        pattern = screen.brightness * (1.0 if (int(self.world_time / 10 + screen.content_id) % 2 == 0) else 0.5)
                        retina[5, i, j] = pattern

        # Fill objects
        for obj in self.env.objects:
            for i, x_local in enumerate(xs):
                for j, y_local in enumerate(ys):
                    wx, wy = world_pos(x_local, y_local)
                    if abs(wx - obj.x) < obj.size and abs(wy - obj.y) < obj.size:
                        retina[1, i, j] = 1.0

        # Fill peers
        for peer in self.env.peers:
            for i, x_local in enumerate(xs):
                for j, y_local in enumerate(ys):
                    wx, wy = world_pos(x_local, y_local)
                    if abs(wx - peer.x) < peer.size and abs(wy - peer.y) < peer.size:
                        retina[2, i, j] = 1.0

        # Occupancy (mirror & objects & peers)
        retina[0] = np.clip(retina[1] + retina[2] + retina[3] + retina[4], 0, 1)
        return retina.tolist()

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
            "world_size": self.env.cfg.world_size,
            "agents": [base_agent],
            "peers": [
                {"id": f"peer-{i}", "pos": [p.x, p.y], "orientation": p.orientation, "velocity": [0.0, 0.0], "expression": p.expression}
                for i, p in enumerate(self.env.peers)
            ],
            "objects": [
                {"id": f"obj-{i}", "type": o.type_id, "pos": [o.x, o.y], "size": [o.size, o.size], "state": {"seen": getattr(o, 'seen', False)}}
                for i, o in enumerate(self.env.objects)
            ],
            "screens": [
                {"id": f"screen-{i}", "pos": [s.x, s.y], "size": [s.size, s.size], "content_id": s.content_id, "brightness": s.brightness}
                for i, s in enumerate(getattr(self.env, "screens", []))
            ],
            "mirrors": [{"id": f"mirror-{i}", "p1": [m.x, 0.0], "p2": [m.x, self.env.cfg.world_size]} for i, m in enumerate(self.env.mirrors)],
            "retina_info": {
                "grid_size": 16,
                "channels": ["occupancy", "objects", "peers", "mirrors", "screens", "intensity"],
                "agent_centric": True,
            },
        }
        return snapshot
