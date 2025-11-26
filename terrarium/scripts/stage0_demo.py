"""Stage 0 demo runner."""

from __future__ import annotations

from terrarium.env import GridWorldConfig, GridWorldEnv
from terrarium.organism import Organism
from terrarium.plasticity import PlasticityController
from terrarium.replay import ReplayBuffer
from terrarium.simulation import SimulationRunner


def main() -> None:
    env = GridWorldEnv(GridWorldConfig(width=8, height=8, patch_radius=1, max_steps=40, num_objects=3, seed=42))
    organism = Organism(action_space=env.action_space)
    replay = ReplayBuffer(capacity=2000, seed=123)
    plasticity = PlasticityController()
    runner = SimulationRunner(env, organism, replay, plasticity)

    episodes = 3
    for ep in range(episodes):
        print(f"=== Episode {ep} ===")
        stats = runner.run_episode(verbose=True)
        print(f"Episode finished in {stats.steps} steps, reward {stats.cumulative_reward:.2f}")
        print(env.render_ascii())
        print()

    print(f"Stored transitions: {len(replay)}")


if __name__ == "__main__":
    main()
