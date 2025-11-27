"""Stage 1 training script with DQN-style learning and wandb logging."""

from __future__ import annotations

import random
from dataclasses import asdict

import numpy as np
import torch
import wandb

from terrarium.backend import TorchBackend
from terrarium.config import RunConfig
from terrarium.env import GridWorldConfig, GridWorldEnv
from terrarium.organism import Organism
from terrarium.plasticity import PlasticityController
from terrarium.replay import ReplayBuffer
from terrarium.training import RLTrainer


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_env(cfg: RunConfig) -> GridWorldEnv:
    env_cfg = GridWorldConfig(
        width=cfg.env_size,
        height=cfg.env_size,
        patch_radius=cfg.patch_radius,
        max_steps=cfg.max_steps_per_episode,
        num_objects=cfg.num_objects,
        seed=cfg.seed,
        step_penalty=cfg.step_penalty,
        object_reward=cfg.object_reward,
        mirror_reward=cfg.mirror_reward,
    )
    return GridWorldEnv(env_cfg)


def main() -> None:
    config = RunConfig()
    set_seeds(config.seed)
    policy_rng = random.Random(config.seed)

    wandb.init(project=config.wandb_project, mode=config.wandb_mode, config=asdict(config))

    backend = TorchBackend(device=config.device)
    print(f"Using device: {backend.device}")
    wandb.config.update({"resolved_device": str(backend.device)})

    env = build_env(config)
    organism = Organism(
        action_space=env.action_space,
        backend=backend,
        hidden_dim=config.hidden_dim,
        bridge_dim=config.bridge_dim,
        grid_size=config.env_size,
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
