"""Stage 2: 2.5D headless environment with objects, peers, and mirrors."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class Stage2Config:
    world_size: float = 8.0
    max_steps: int = 80
    max_objects: int = 5
    max_peers: int = 1
    max_reflections: int = 2
    max_screens: int = 1
    step_size: float = 0.5
    turn_step: float = math.pi / 8
    seed: Optional[int] = None
    include_vision_task: bool = False
    include_go_to_sound: bool = False
    enable_head_yaw: bool = False
    tasks: Tuple[str, ...] = (
        "goto_mirror",
        "touch_object",
        "examine_reflection",
        "follow_peer",
        "social_gaze",
        "novel_object_investigation",
        "cooperative_goal",
    )
    step_penalty: float = -0.01
    success_reward: float = 1.0
    object_reward: float = 0.5
    mirror_reward: float = 0.1


@dataclass
class Entity:
    x: float
    y: float
    size: float = 0.4
    orientation: float = 0.0  # body yaw radians
    head_offset: float = 0.0  # head yaw relative to body


@dataclass
class Object(Entity):
    type_id: int = 0
    seen: bool = False


@dataclass
class Peer(Entity):
    wander: bool = True
    follow: bool = False
    expression: float = 0.0  # placeholder scalar


@dataclass
class Screen(Entity):
    content_id: int = 0
    brightness: float = 1.0


@dataclass
class MirrorSurface:
    x: float  # vertical line at x


class Stage2Env:
    ACTIONS: Sequence[str] = (
        "forward",
        "backward",
        "left",
        "right",
        "turn_left",
        "turn_right",
        "head_left",
        "head_right",
        "head_center",
        "stay",
        "sleep",
    )

    def __init__(self, config: Stage2Config):
        self.cfg = config
        self.rng = random.Random(config.seed)
        self.agent = Entity(0.0, 0.0, size=0.5, orientation=0.0)
        self.objects: List[Object] = []
        self.peers: List[Peer] = []
        self.mirrors: List[MirrorSurface] = []
        self.screens: List[Screen] = []
        self.steps = 0
        self.task_id = "goto_mirror"
        self.success = False
        self.gaze_hold = 0
        self.task_list = list(self.cfg.tasks)
        if getattr(self.cfg, "include_vision_task", False) and "vision_object_discrim" not in self.task_list:
            self.task_list.append("vision_object_discrim")
        if getattr(self.cfg, "include_go_to_sound", False) and "go_to_sound" not in self.task_list:
            self.task_list.append("go_to_sound")
        self.coop_goal = (self.cfg.world_size * 0.8, self.cfg.world_size * 0.8)

    def reset(self, task_id: Optional[str] = None) -> Dict[str, object]:
        self.steps = 0
        self.success = False
        self.task_id = task_id or self.rng.choice(self.task_list)
        self.gaze_hold = 0
        self.agent = Entity(
            x=self.rng.uniform(0.5, self.cfg.world_size - 0.5),
            y=self.rng.uniform(0.5, self.cfg.world_size - 0.5),
            size=0.5,
            orientation=self.rng.uniform(-math.pi, math.pi),
            head_offset=0.0,
        )
        self.objects = self._spawn_objects()
        self.peers = self._spawn_peers()
        self.mirrors = [MirrorSurface(x=self.cfg.world_size / 2), MirrorSurface(x=0.5)][: self.cfg.max_reflections]
        self.screens = self._spawn_screens()
        if self.task_id == "vision_object_discrim":
            # ensure at least two objects and mark one as glowing target
            if len(self.objects) < 2:
                self.objects.extend(self._spawn_objects())
            self.objects = self.objects[:2]
            for o in self.objects:
                o.glow = False
            target_idx = self.rng.randrange(len(self.objects))
            self.objects[target_idx].glow = True
        self.sound_source = None
        if self.task_id == "go_to_sound":
            if not self.objects:
                self.objects = self._spawn_objects()
            self.sound_source = self.objects[0]
        return self._observe()

    @property
    def action_space(self) -> Sequence[str]:
        return self.ACTIONS

    def step(self, action: str) -> Tuple[Dict[str, object], float, bool, Dict[str, object]]:
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}")
        self.steps += 1
        if action == "sleep":
            pass  # no movement during sleep
        elif action in ("head_left", "head_right", "head_center"):
            if self.cfg.enable_head_yaw:
                self._update_head(action)
        else:
            self._update_agent(action)
        self._update_peers()

        reward = self.cfg.step_penalty
        info: Dict[str, object] = {"task_id": self.task_id}

        dist_mirror = self._dist_to_nearest_mirror()
        if dist_mirror < 0.5:
            info["mirror_contact"] = True

        if self.task_id == "goto_mirror":
            if dist_mirror < 0.5:
                reward += self.cfg.success_reward
                info["task_success"] = True
        elif self.task_id == "touch_object":
            obj = self._closest_object()
            if obj and self._distance(self.agent, obj) < (self.agent.size + obj.size):
                reward += self.cfg.success_reward
                info["task_success"] = True
        elif self.task_id == "examine_reflection":
            if self._is_facing_mirror(threshold=0.2) and self._dist_to_nearest_mirror() < 1.0:
                reward += self.cfg.success_reward
                info["task_success"] = True
        elif self.task_id == "follow_peer":
            for peer in self.peers:
                d = self._distance(self.agent, peer)
                if 1.0 <= d <= 2.0:
                    reward += self.cfg.success_reward
                    info["task_success"] = True
                    break
        elif self.task_id == "social_gaze":
            success_here = False
            facing_any = False
            for peer in self.peers:
                if self._is_facing_target(peer, angle_thresh=0.3, dist_thresh=3.0):
                    facing_any = True
                    self.gaze_hold += 1
                    if self.gaze_hold >= 2:
                        reward += self.cfg.success_reward
                        info["task_success"] = True
                        success_here = True
                        break
            if not facing_any:
                self.gaze_hold = 0
        # Social contact flag (proximity or gaze)
        social_contact = False
        for peer in self.peers:
            if self._distance(self.agent, peer) <= 1.5:
                social_contact = True
                break
            if self._is_facing_target(peer, angle_thresh=0.3, dist_thresh=3.0):
                social_contact = True
                break
        if social_contact:
            info["social_contact"] = True
        if self.task_id == "novel_object_investigation":
            obj = self._closest_object()
            if obj and not obj.seen and self._distance(self.agent, obj) < 1.0:
                reward += self.cfg.success_reward
                obj.seen = True
                info["task_success"] = True
        elif self.task_id == "cooperative_goal":
            goal_x, goal_y = self.coop_goal
            agent_near = math.hypot(self.agent.x - goal_x, self.agent.y - goal_y) < 0.7
            peer_near = any(math.hypot(p.x - goal_x, p.y - goal_y) < 0.7 for p in self.peers)
            if agent_near and peer_near:
                reward += self.cfg.success_reward
                info["task_success"] = True
        elif self.task_id == "vision_object_discrim":
            obj = self._closest_object()
            if obj and self._distance(self.agent, obj) < (self.agent.size + obj.size):
                if getattr(obj, "glow", False):
                    reward += self.cfg.success_reward
                    info["task_success"] = True
                else:
                    reward += self.cfg.step_penalty  # touching wrong object gives no success
        elif self.task_id == "go_to_sound":
            if self.sound_source and self._distance(self.agent, self.sound_source) < (self.agent.size + self.sound_source.size):
                reward += self.cfg.success_reward
                info["task_success"] = True

        done = info.get("task_success", False) or self.steps >= self.cfg.max_steps
        obs = self._observe()
        return obs, reward, done, info

    # Internal helpers
    def _update_agent(self, action: str) -> None:
        if action == "turn_left":
            self.agent.orientation += self.cfg.turn_step
        elif action == "turn_right":
            self.agent.orientation -= self.cfg.turn_step
        elif action == "forward":
            dx, dy = self._dir_vector()
            self.agent.x += dx * self.cfg.step_size
            self.agent.y += dy * self.cfg.step_size
        elif action == "backward":
            dx, dy = self._dir_vector()
            self.agent.x -= dx * self.cfg.step_size
            self.agent.y -= dy * self.cfg.step_size
        elif action == "left":
            dx, dy = self._dir_vector()
            self.agent.x += -dy * self.cfg.step_size
            self.agent.y += dx * self.cfg.step_size
        elif action == "right":
            dx, dy = self._dir_vector()
            self.agent.x += dy * self.cfg.step_size
            self.agent.y += -dx * self.cfg.step_size
        self.agent.x = max(0.0, min(self.cfg.world_size, self.agent.x))
        self.agent.y = max(0.0, min(self.cfg.world_size, self.agent.y))

    def _update_head(self, action: str) -> None:
        max_offset = math.pi / 3
        if action == "head_left":
            self.agent.head_offset = min(max_offset, self.agent.head_offset + self.cfg.turn_step)
        elif action == "head_right":
            self.agent.head_offset = max(-max_offset, self.agent.head_offset - self.cfg.turn_step)
        elif action == "head_center":
            self.agent.head_offset = 0.0

    def _update_peers(self) -> None:
        for peer in self.peers:
            if peer.follow:
                dx = self.agent.x - peer.x
                dy = self.agent.y - peer.y
                peer.orientation = math.atan2(dy, dx)
                dist = math.hypot(dx, dy)
                if dist > 2.5:
                    peer.x += math.cos(peer.orientation) * self.cfg.step_size * 0.7
                    peer.y += math.sin(peer.orientation) * self.cfg.step_size * 0.7
            elif peer.wander:
                peer.orientation += self.rng.uniform(-0.2, 0.2)
                peer.x += math.cos(peer.orientation) * self.cfg.step_size * 0.5
                peer.y += math.sin(peer.orientation) * self.cfg.step_size * 0.5
            peer.x = max(0.0, min(self.cfg.world_size, peer.x))
            peer.y = max(0.0, min(self.cfg.world_size, peer.y))
            peer.expression = self.rng.uniform(-1.0, 1.0)

    def _dir_vector(self) -> Tuple[float, float]:
        return math.cos(self.agent.orientation), math.sin(self.agent.orientation)

    def _spawn_objects(self) -> List[Object]:
        objs: List[Object] = []
        for i in range(self.cfg.max_objects):
            objs.append(
                Object(
                    x=self.rng.uniform(0.5, self.cfg.world_size - 0.5),
                    y=self.rng.uniform(0.5, self.cfg.world_size - 0.5),
                    size=self.rng.uniform(0.2, 0.6),
                    orientation=0.0,
                    type_id=i % 3,
                )
            )
        return objs

    def _spawn_peers(self) -> List[Peer]:
        peers: List[Peer] = []
        for _ in range(self.cfg.max_peers):
            peers.append(
                Peer(
                    x=self.rng.uniform(0.5, self.cfg.world_size - 0.5),
                    y=self.rng.uniform(0.5, self.cfg.world_size - 0.5),
                    size=0.5,
                    orientation=self.rng.uniform(-math.pi, math.pi),
                    wander=True,
                    follow=True,
                )
            )
        return peers

    def _spawn_screens(self) -> List[Screen]:
        screens: List[Screen] = []
        for i in range(self.cfg.max_screens):
            screens.append(
                Screen(
                    x=self.rng.uniform(0.5, self.cfg.world_size - 0.5),
                    y=self.rng.uniform(0.5, self.cfg.world_size - 0.5),
                    size=0.6,
                    orientation=0.0,
                    content_id=i % 3,
                    brightness=1.0,
                )
            )
        return screens

    def _observe(self) -> Dict[str, object]:
        """Structured egocentric observation."""
        ego = {
            "pos": [self.agent.x, self.agent.y],
            "orientation": self.agent.orientation + (self.agent.head_offset if self.cfg.enable_head_yaw else 0.0),
            "body_orientation": self.agent.orientation,
            "head_orientation": self.agent.orientation + (self.agent.head_offset if self.cfg.enable_head_yaw else 0.0),
            "velocity": [0.0, 0.0],  # placeholder
        }
        objects = []
        for obj in self.objects[: self.cfg.max_objects]:
            rel = self._to_ego(obj.x, obj.y)
            objects.append(
                {
                    "type_id": obj.type_id,
                    "rel_x": rel[0],
                    "rel_y": rel[1],
                    "size": obj.size,
                    "visible": 1.0,
                }
            )
        while len(objects) < self.cfg.max_objects:
            objects.append({"type_id": 0, "rel_x": 0.0, "rel_y": 0.0, "size": 0.0, "visible": 0.0})
        screens_obs = []
        for screen in self.screens[: self.cfg.max_screens]:
            rel = self._to_ego(screen.x, screen.y)
            screens_obs.append(
                {
                    "rel_x": rel[0],
                    "rel_y": rel[1],
                    "size": screen.size,
                    "content_id": screen.content_id,
                    "brightness": screen.brightness,
                    "visible": 1.0,
                }
            )
        while len(screens_obs) < self.cfg.max_screens:
            screens_obs.append({"rel_x": 0.0, "rel_y": 0.0, "size": 0.0, "content_id": 0, "brightness": 0.0, "visible": 0.0})

        peers_obs = []
        for peer in self.peers[: self.cfg.max_peers]:
            rel = self._to_ego(peer.x, peer.y)
            heading = self.agent.orientation + (self.agent.head_offset if self.cfg.enable_head_yaw else 0.0)
            peers_obs.append(
                {
                    "rel_x": rel[0],
                    "rel_y": rel[1],
                    "orientation": self._angle_diff(peer.orientation, heading),
                    "expression": peer.expression,
                }
            )
        while len(peers_obs) < self.cfg.max_peers:
            peers_obs.append({"rel_x": 0.0, "rel_y": 0.0, "orientation": 0.0, "expression": 0.0})

        reflections = []
        for mirror in self.mirrors[: self.cfg.max_reflections]:
            rx = 2 * mirror.x - self.agent.x
            ry = self.agent.y
            rel = self._to_ego(rx, ry)
            # Reflect orientation across vertical axis: theta_reflect = pi - theta
            theta_reflect = math.pi - self.agent.orientation
            # Express in ego frame
            theta_rel = self._angle_diff(theta_reflect, self.agent.orientation)
            reflections.append({"rel_x": rel[0], "rel_y": rel[1], "orientation": theta_rel})
        while len(reflections) < self.cfg.max_reflections:
            reflections.append({"rel_x": 0.0, "rel_y": 0.0, "orientation": 0.0})

        return {
            "task_id": self.task_id,
            "step_count": self.steps,
            "self": ego,
            "objects": objects,
            "peers": peers_obs,
            "mirror_reflections": reflections,
            "screens": screens_obs,
        }

    def _distance(self, a: Entity, b: Entity) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def _closest_object(self) -> Optional[Object]:
        if not self.objects:
            return None
        return min(self.objects, key=lambda o: self._distance(self.agent, o))

    def _dist_to_nearest_mirror(self) -> float:
        return min(abs(self.agent.x - m.x) for m in self.mirrors) if self.mirrors else float("inf")

    def _is_facing_mirror(self, threshold: float = 0.2) -> bool:
        if not self.mirrors:
            return False
        # Use nearest mirror for heading check
        mirror_x = min(self.mirrors, key=lambda m: abs(m.x - self.agent.x)).x
        dir_x, _ = self._dir_vector()
        if mirror_x > self.agent.x:
            return dir_x > threshold
        return dir_x < -threshold

    def _is_facing_target(self, target: Entity, angle_thresh: float = 0.3, dist_thresh: float = 3.0) -> bool:
        dx = target.x - self.agent.x
        dy = target.y - self.agent.y
        dist = math.hypot(dx, dy)
        if dist > dist_thresh:
            return False
        angle_to_target = math.atan2(dy, dx)
        return abs(self._angle_diff(angle_to_target, self.agent.orientation)) < angle_thresh

    def _to_ego(self, x: float, y: float) -> Tuple[float, float]:
        dx = x - self.agent.x
        dy = y - self.agent.y
        heading = self.agent.orientation + (self.agent.head_offset if self.cfg.enable_head_yaw else 0.0)
        cos_o = math.cos(-heading)
        sin_o = math.sin(-heading)
        rel_x = cos_o * dx - sin_o * dy
        rel_y = sin_o * dx + cos_o * dy
        return rel_x, rel_y

    def _angle_diff(self, a: float, b: float) -> float:
        d = (a - b + math.pi) % (2 * math.pi) - math.pi
        return d
