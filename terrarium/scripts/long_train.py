"""Long-run training script with long_train epsilon schedule."""

from __future__ import annotations

import random
from dataclasses import asdict

import numpy as np
import torch
import wandb

from terrarium.backend import TorchBackend
from terrarium.config import RunConfig
from terrarium.env import Stage2Config, Stage2Env
from terrarium.organism import Organism
from terrarium.plasticity import PlasticityController
from terrarium.replay import ReplayBuffer
from terrarium.training import RLTrainer
from terrarium.world import World
from terrarium.runtime import Runtime


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_env(cfg: RunConfig) -> Stage2Env:
    env_cfg = Stage2Config(
        world_size=float(cfg.env_size),
        max_steps=cfg.max_steps_per_episode,
        max_objects=cfg.max_stage2_objects,
        max_peers=cfg.max_stage2_peers,
        max_reflections=cfg.max_stage2_reflections,
        max_screens=cfg.max_stage2_screens,
        seed=cfg.seed,
        include_vision_task=cfg.enable_vision_object_discrim,
        include_go_to_sound=cfg.enable_go_to_sound,
        enable_head_yaw=cfg.enable_head_yaw,
        enable_stay_with_caregiver=cfg.enable_stay_with_caregiver,
        enable_explore_and_return=cfg.enable_explore_and_return,
        enable_move_to_target=cfg.enable_move_to_target,
        use_body_variation=cfg.use_body_variation,
        body_move_scale_range=cfg.body_move_scale_range,
        body_turn_scale_range=cfg.body_turn_scale_range,
        body_noise_scale_range=cfg.body_noise_scale_range,
    )
    return Stage2Env(env_cfg)


def main() -> None:
    config = RunConfig(
        num_episodes=500,
        epsilon_mode="long_train",
        log_retina=False,
        log_topdown_video=False,
        wandb_mode="online",
    )
    set_seeds(config.seed)
    policy_rng = random.Random(config.seed)

    wandb.init(project=config.wandb_project, mode=config.wandb_mode, config=asdict(config))

    backend = TorchBackend(device=config.device)
    wandb.config.update({"resolved_device": str(backend.device)})

    env = build_env(config)
    world = World(env)
    organism = Organism(
        action_space=env.action_space,
        backend=backend,
        hidden_dim=config.hidden_dim,
        bridge_dim=config.bridge_dim,
        grid_size=int(config.env_size),
        max_steps=config.max_steps_per_episode,
        task_ids=env.cfg.tasks,
        policy_rng=policy_rng,
        max_objects=env.cfg.max_objects,
        max_peers=env.cfg.max_peers,
        max_reflections=env.cfg.max_reflections,
    )
    replay = ReplayBuffer(capacity=config.max_buffer_size, seed=config.seed)
    plasticity = PlasticityController()

    trainer = RLTrainer(world, organism, replay, plasticity, config)
    runtime = Runtime(world, trainer)
    runtime.run()

    wandb.finish()


if __name__ == "__main__":
    main()
