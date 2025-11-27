"""Exist mode: run a persistent world server with an organism client."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from terrarium.backend import TorchBackend
from terrarium.config import RunConfig
from terrarium.env import Stage2Config, Stage2Env
from terrarium.organism import Organism
from terrarium.replay import ReplayBuffer
from terrarium.plasticity import PlasticityController
from terrarium.agents import OrganismClient
from terrarium.world import World
from terrarium.world_server import WorldServer


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_world(cfg: RunConfig) -> World:
    env_cfg = Stage2Config(
        world_size=float(cfg.env_size),
        max_steps=cfg.max_steps_per_episode,
        max_objects=cfg.max_stage2_objects,
        max_peers=cfg.max_stage2_peers,
        max_reflections=cfg.max_stage2_reflections,
        max_screens=cfg.max_stage2_screens,
        seed=cfg.seed,
    )
    env = Stage2Env(env_cfg)
    return World(env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a persistent world server with an organism client.")
    parser.add_argument("--steps", type=int, default=-1, help="Number of steps to run (default infinite; set a positive int for finite).")
    parser.add_argument("--learn", action="store_true", default=False, help="Enable online learning in exist mode.")
    parser.add_argument("--log-interval", type=int, default=100, help="Print a status line every N steps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    cfg = RunConfig()
    cfg.seed = args.seed
    set_seeds(cfg.seed)
    backend = TorchBackend(device=cfg.device)
    print(f"Using device: {backend.device}")
    world = build_world(cfg)
    organism = Organism(
        action_space=world.env.action_space,
        backend=backend,
        hidden_dim=cfg.hidden_dim,
        bridge_dim=cfg.bridge_dim,
        grid_size=int(cfg.env_size),
        max_steps=cfg.max_steps_per_episode,
        task_ids=world.env.cfg.tasks,
        max_objects=world.env.cfg.max_objects,
        max_peers=world.env.cfg.max_peers,
        max_reflections=world.env.cfg.max_reflections,
    )
    replay = ReplayBuffer(capacity=cfg.max_buffer_size, seed=cfg.seed)
    plasticity = PlasticityController()
    client = OrganismClient(organism, cfg, replay=replay, plasticity=plasticity, learn=args.learn)

    server = WorldServer(world, learn=args.learn)
    server.add_agent("agent-1", client)

    print(f"[exist] starting run: steps={args.steps if args.steps>=0 else 'infinite'} learn={args.learn} seed={args.seed}")
    try:
        t = 0
        while True:
            if args.steps >= 0 and t >= args.steps:
                break
            server.step(t)
            if args.log_interval > 0 and t % args.log_interval == 0:
                snap = server.get_snapshot()
                agent_meta = snap.get("agents_status", [{}])[0] if snap.get("agents_status") else {}
                pos = agent_meta.get("pos", [world.env.agent.x, world.env.agent.y]) if agent_meta else [world.env.agent.x, world.env.agent.y]
                task = agent_meta.get("task_id", getattr(world.env, "task_id", None))
                sleeping = agent_meta.get("sleeping", False)
                print(f"[exist] t={t} pos={pos} task={task} sleeping={sleeping}")
            t += 1
    except KeyboardInterrupt:
        print("[exist] stopped.")


if __name__ == "__main__":
    main()
