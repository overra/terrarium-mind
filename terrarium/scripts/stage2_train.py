"""Stage 2 training script using the 2.5D environment."""

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


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_env(cfg: RunConfig) -> Stage2Env:
    env_cfg = Stage2Config(
        world_size=float(cfg.env_size),
        max_steps=cfg.max_steps_per_episode,
        max_objects=cfg.max_stage2_objects,
        max_peers=cfg.max_stage2_peers,
        max_reflections=cfg.max_stage2_reflections,
        seed=cfg.seed,
    )
    return Stage2Env(env_cfg)


def main() -> None:
    config = RunConfig()
    set_seeds(config.seed)
    policy_rng = random.Random(config.seed)

    wandb.init(project=config.wandb_project, mode=config.wandb_mode, config=asdict(config))

    backend = TorchBackend()
    env = build_env(config)
    organism = Organism(
        action_space=env.action_space,
        backend=backend,
        hidden_dim=config.hidden_dim,
        bridge_dim=config.bridge_dim,
        grid_size=int(config.env_size),
        max_steps=config.max_steps_per_episode,
        task_ids=env.cfg.tasks,
        policy_rng=policy_rng,
    )
    replay = ReplayBuffer(capacity=config.max_buffer_size, seed=config.seed)
    plasticity = PlasticityController()

    trainer = RLTrainer(env, organism, replay, plasticity, config)
    trainer.run()

    wandb.finish()


if __name__ == "__main__":
    main()
