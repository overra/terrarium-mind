"""Run-level configuration for simulations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunConfig:
    seed: int = 42
    num_episodes: int = 50
    max_steps_per_episode: int = 60
    backend: str = "torch"  # future: "mlx"
    device: str = "cpu"  # "auto", "cuda", "mps", or "cpu" - CPU faster for this architecture
    env_size: int = 8  # Stage 1 grid; for Stage 2, world_size = env_size (float)
    patch_radius: int = 1
    num_objects: int = 3
    step_penalty: float = -0.01
    object_reward: float = 1.0
    mirror_reward: float = 0.05
    log_interval_steps: int = 10
    wandb_mode: str = "online"
    wandb_project: str = "digital-organism"
    bridge_dim: int = 16
    hidden_dim: int = 64  # balanced size - fast on CPU
    epsilon_start: float = 0.8
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    gamma: float = 0.95
    lr: float = 1e-3
    batch_size: int = 32
    train_start: int = 200
    train_interval: int = 1
    target_update_interval: int = 50
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    curiosity_epsilon_scale: float = 0.3
    max_buffer_size: int = 10000
    # Stage 2 environment settings
    max_stage2_objects: int = 5
    max_stage2_peers: int = 1
    max_stage2_reflections: int = 2
    max_stage2_screens: int = 1
    # Sleep/consolidation
    sleep_replay_multiplier: int = 3
    sleep_recovery_rate: float = 0.01
    sleep_rest_rate: float = 0.01
    # Vision logging
    log_retina: bool = False
    retina_log_interval_episodes: int = 10
    retina_max_snapshots_per_run: int = 20
    # Observational learning
    use_observational_learning: bool = False
    demo_fraction: float = 0.0
