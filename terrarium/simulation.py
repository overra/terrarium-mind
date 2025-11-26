"""Simulation loop wiring environment, organism, replay, and plasticity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

from .env import GridWorldEnv
from .organism import Organism
from .plasticity import PlasticityController
from .replay import ReplayBuffer, Transition
from .utils import compute_novelty, compute_prediction_error


@dataclass
class EpisodeStats:
    steps: int
    cumulative_reward: float


class SimulationRunner:
    """Runs episodes and logs transitions."""

    def __init__(
        self,
        env: GridWorldEnv,
        organism: Organism,
        replay_buffer: ReplayBuffer,
        plasticity: PlasticityController,
    ) -> None:
        self.env = env
        self.organism = organism
        self.replay = replay_buffer
        self.plasticity = plasticity
        self.expected_reward = 0.0

    def run_episode(self, max_steps: int | None = None, verbose: bool = False) -> EpisodeStats:
        """Roll out a single episode."""
        obs = self.env.reset()
        self.organism.reset()
        prev_obs = None
        last_reward = 0.0
        last_info: Dict[str, Any] = {}
        cumulative_reward = 0.0
        step_limit = max_steps or self.env.max_steps

        for step in range(step_limit):
            obs_dict = asdict(obs)
            prev_obs_dict = asdict(prev_obs) if prev_obs else None
            novelty = compute_novelty(obs_dict, prev_obs_dict)
            prediction_error = compute_prediction_error(last_reward, self.expected_reward)

            outputs = self.organism.step(
                observation=obs_dict,
                reward=last_reward,
                novelty=novelty,
                prediction_error=prediction_error,
                info=last_info,
            )

            next_obs, reward, done, info = self.env.step(outputs.action)
            next_obs_dict = asdict(next_obs)

            transition_error = compute_prediction_error(reward, self.expected_reward)
            priority = self.plasticity.compute_priority(
                reward=reward,
                novelty=novelty,
                prediction_error=transition_error,
                emotion_latent=outputs.emotion.latent,
            )
            self.replay.add(
                Transition(
                    observation=obs_dict,
                    action=outputs.action,
                    reward=reward,
                    next_observation=next_obs_dict,
                    done=done,
                    emotion_latent=outputs.emotion.latent,
                    expression=outputs.expression,
                    novelty=novelty,
                    prediction_error=transition_error,
                    priority=priority,
                    info={"core_readouts": outputs.core_readouts, "env_info": info},
                )
            )

            self.expected_reward = 0.9 * self.expected_reward + 0.1 * reward
            cumulative_reward += reward

            if verbose:
                self._print_step(step, outputs.action, reward, novelty, outputs)

            prev_obs = obs
            obs = next_obs
            last_reward = reward
            last_info = info

            if done:
                break

        return EpisodeStats(steps=step + 1, cumulative_reward=cumulative_reward)

    def _print_step(
        self,
        step: int,
        action: str,
        reward: float,
        novelty: float,
        outputs: Any,
    ) -> None:
        """Log a concise line for debugging."""
        emo = outputs.emotion
        valence = emo.core_affect.valence
        arousal = emo.core_affect.arousal
        print(
            f"[step {step:02d}] action={action:>5} reward={reward:+.2f} "
            f"novelty={novelty:.2f} valence={valence:+.2f} arousal={arousal:+.2f}"
        )
