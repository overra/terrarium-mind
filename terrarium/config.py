"""Run-level configuration for simulations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunConfig:
    seed: int = 42
    num_episodes: int = 5
    max_steps_per_episode: int = 50
    backend: str = "torch"  # future: "mlx"
    env_size: int = 8
    patch_radius: int = 1
    num_objects: int = 3
    step_penalty: float = -0.01
    object_reward: float = 1.0
    mirror_reward: float = 0.05
    log_interval_steps: int = 10
    wandb_mode: str = "offline"
    wandb_project: str = "digital-organism"
