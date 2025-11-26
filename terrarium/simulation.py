"""Simulation loop wiring environment, organism, replay, and plasticity."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

from .env import GridWorldEnv
from .organism import Organism
from .plasticity import PlasticityController
from .replay import ReplayBuffer, Transition
from .utils import compute_novelty, compute_prediction_error


@dataclass
class EpisodeStats:
    steps: int
    cumulative_reward: float
    valence_trace: List[float] = field(default_factory=list)
    arousal_trace: List[float] = field(default_factory=list)
    prediction_error_trace: List[float] = field(default_factory=list)
    last_drives: Dict[str, float] | None = None
    last_core_affect: Dict[str, float] | None = None
    last_expression: Dict[str, Any] | None = None


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

    def run_episode(
        self,
        max_steps: int | None = None,
        verbose: bool = False,
        collect_traces: bool = False,
        log_interval: int | None = None,
    ) -> EpisodeStats:
        """Roll out a single episode."""
        obs = self.env.reset()
        self.organism.reset()
        last_reward = 0.0
        last_info: Dict[str, Any] = {}
        cumulative_reward = 0.0
        step_limit = max_steps or self.env.max_steps
        valence_trace: List[float] = []
        arousal_trace: List[float] = []
        prediction_errors: List[float] = []
        last_drives: Dict[str, float] | None = None
        last_core_affect: Dict[str, float] | None = None
        last_expression: Dict[str, Any] | None = None

        state = self.organism.encode_observation(asdict(obs), last_reward, 1.0, 0.0, last_info)

        for step in range(step_limit):
            obs_dict = asdict(obs)
            prediction_error = compute_prediction_error(last_reward, self.expected_reward)

            epsilon = 0.2
            action, _ = self.organism.select_action(state.brain_state_tensor, epsilon)

            next_obs, reward, done, info = self.env.step(action)
            next_obs_dict = asdict(next_obs)

            novelty_transition = compute_novelty(next_obs_dict, obs_dict)
            transition_error = compute_prediction_error(reward, self.expected_reward)
            next_state = self.organism.encode_observation(next_obs_dict, reward, novelty_transition, transition_error, info)

            priority = self.plasticity.compute_priority(
                reward=reward,
                novelty=novelty_transition,
                prediction_error=transition_error,
                emotion_latent=state.emotion.latent,
                core_affect=state.core_affect,
            )
            self.replay.add(
                Transition(
                    observation=obs_dict,
                    action=action,
                    action_idx=self.organism.action_to_idx[action],
                    reward=reward,
                    next_observation=next_obs_dict,
                    done=done,
                    brain_state=state.brain_state,
                    next_brain_state=next_state.brain_state,
                    emotion_latent=state.emotion.latent,
                    drives=state.drives,
                    core_affect=state.core_affect,
                    expression=state.expression,
                    novelty=novelty_transition,
                    prediction_error=transition_error,
                    priority=priority,
                    info={"env_info": info},
                )
            )

            self.expected_reward = 0.9 * self.expected_reward + 0.1 * reward
            cumulative_reward += reward
            if collect_traces:
                valence_trace.append(state.core_affect["valence"])
                arousal_trace.append(state.core_affect["arousal"])
                prediction_errors.append(transition_error)
            last_drives = state.drives
            last_core_affect = state.core_affect
            last_expression = state.expression

            if verbose:
                self._print_step(step, action, reward, novelty_transition, state.emotion)
            elif log_interval and collect_traces and step % log_interval == 0:
                self._print_step(step, action, reward, novelty_transition, state.emotion)

            obs = next_obs
            state = next_state
            last_reward = reward
            last_info = info

            if done:
                break

        return EpisodeStats(
            steps=step + 1,
            cumulative_reward=cumulative_reward,
            valence_trace=valence_trace,
            arousal_trace=arousal_trace,
            prediction_error_trace=prediction_errors,
            last_drives=last_drives,
            last_core_affect=last_core_affect,
            last_expression=last_expression,
        )

    def _print_step(
        self,
        step: int,
        action: str,
        reward: float,
        novelty: float,
        emotion_state: Any,
    ) -> None:
        """Log a concise line for debugging."""
        valence = emotion_state.core_affect.valence
        arousal = emotion_state.core_affect.arousal
        print(
            f"[step {step:02d}] action={action:>5} reward={reward:+.2f} "
            f"novelty={novelty:.2f} valence={valence:+.2f} arousal={arousal:+.2f}"
        )
