"""Stage 0.5 demo runner with config, backend, and wandb logging."""

from __future__ import annotations

import random
from dataclasses import asdict
from typing import Dict

import numpy as np
import torch
import wandb

from terrarium.backend import TorchBackend
from terrarium.config import RunConfig
from terrarium.env import GridWorldConfig, GridWorldEnv
from terrarium.organism import Organism
from terrarium.plasticity import PlasticityController
from terrarium.replay import ReplayBuffer
from terrarium.simulation import SimulationRunner


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    config.num_episodes = 3  # keep demo short
    set_seeds(config.seed)

    if config.backend != "torch":
        raise NotImplementedError(f"Backend {config.backend} not yet implemented.")
    backend = TorchBackend()
    env = build_env(config)
    organism = Organism(action_space=env.action_space, backend=backend)
    replay = ReplayBuffer(capacity=5000, seed=config.seed)
    plasticity = PlasticityController()
    runner = SimulationRunner(env, organism, replay, plasticity)

    wandb.init(project=config.wandb_project, mode=config.wandb_mode, config=asdict(config))

    for ep in range(config.num_episodes):
        print(f"=== Episode {ep} ===")
        stats = runner.run_episode(
            max_steps=config.max_steps_per_episode,
            verbose=False,
            collect_traces=True,
            log_interval=config.log_interval_steps,
        )

        mean_valence = float(np.mean(stats.valence_trace)) if stats.valence_trace else 0.0
        mean_arousal = float(np.mean(stats.arousal_trace)) if stats.arousal_trace else 0.0
        mean_pred_error = float(np.mean(stats.prediction_error_trace)) if stats.prediction_error_trace else 0.0

        wandb.log(
            {
                "episode": ep,
                "episode_reward": stats.cumulative_reward,
                "episode_length": stats.steps,
                "mean_valence": mean_valence,
                "mean_arousal": mean_arousal,
                "mean_prediction_error": mean_pred_error,
            },
            step=ep,
        )

        print(
            f"Episode finished in {stats.steps} steps, reward {stats.cumulative_reward:.2f}, "
            f"valence {mean_valence:+.2f}, arousal {mean_arousal:+.2f}"
        )
        print(env.render_ascii())
        print()

    print(f"Stored transitions: {len(replay)}")
    wandb.finish()


if __name__ == "__main__":
    main()
