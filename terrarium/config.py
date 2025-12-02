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
    # Camera vision
    vision_mode: str = "retina"  # "retina", "camera", or "both"
    camera_size: int = 32
    camera_channels: int = 3
    log_camera: bool = False
    camera_log_interval_episodes: int = 20
    camera_max_snapshots_per_run: int = 10
    use_observational_learning: bool = False
    demo_fraction: float = 0.0
    enable_vision_object_discrim: bool = False
    enable_go_to_sound: bool = False
    # Predictive head / auxiliary losses
    use_predictive_head: bool = True
    lambda_pred_emotion: float = 0.1
    lambda_pred_core: float = 0.1
    # Salient memory
    use_salient_memory: bool = True
    # Head yaw
    enable_head_yaw: bool = False
    # Homeostasis intrinsic reward
    use_homeostasis: bool = True
    homeostasis_weight: float = 0.05
    homeostasis_chronic_penalty: float = 0.01
    # Social tasks
    enable_stay_with_caregiver: bool = False
    enable_explore_and_return: bool = False
    enable_move_to_target: bool = False
    # Observational learning
    lambda_imitation: float = 0.01
    # Body variation
    use_body_variation: bool = False
    body_move_scale_range: tuple[float, float] = (0.5, 1.5)
    body_turn_scale_range: tuple[float, float] = (0.5, 1.5)
    body_noise_scale_range: tuple[float, float] = (0.0, 0.2)
    # Topdown video logging
    log_topdown_video: bool = False
    topdown_log_interval_episodes: int = 20
    topdown_max_videos_per_run: int = 3
    # Epsilon scheduling
    epsilon_mode: str = "dev"  # "dev" or "long_train"
    epsilon_long_train_final: float = 0.2
    epsilon_long_train_steps: int = 20000
