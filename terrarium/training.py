"""RL training loop wiring env, organism, and prioritized replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import wandb

from .config import RunConfig
from .env import GridWorldEnv
from .organism import EncodedState, Organism
from .metabolism import MetabolicCore
from .plasticity import PlasticityController
from .replay import ReplayBuffer, Transition
from .utils import compute_novelty, compute_prediction_error


@dataclass
class EpisodeMetrics:
    steps: int
    reward: float
    valence_trace: List[float] = field(default_factory=list)
    arousal_trace: List[float] = field(default_factory=list)
    prediction_error_trace: List[float] = field(default_factory=list)
    task_id: str = ""
    success: bool = False
    mean_energy: float = 0.0
    mean_fatigue: float = 0.0
    mean_ts_reward: float = 0.0
    mean_ts_social: float = 0.0
    mean_ts_reflection: float = 0.0
    valence_positive_fraction: float = 0.0
    sleep_fraction: float = 0.0
    avg_sleep_length: float = 0.0
    mean_sleep_drive: float = 0.0


class RLTrainer:
    """DQN-style trainer with emotion-modulated exploration and priorities."""

    def __init__(
        self,
        env: Any,
        organism: Organism,
        replay: ReplayBuffer,
        plasticity: PlasticityController,
        config: RunConfig,
    ) -> None:
        self.env = env
        self.organism = organism
        self.replay = replay
        self.plasticity = plasticity
        self.cfg = config
        self.optimizer: torch.optim.Optimizer | None = None
        self.global_step: int = 0
        self.expected_reward: float = 0.0
        self.epsilon: float = config.epsilon_start
        cfg_tasks = getattr(env, "cfg", getattr(env, "env", None) and getattr(env.env, "cfg", None))
        task_list = cfg_tasks.tasks if cfg_tasks else []
        self.success_counters: Dict[str, List[bool]] = {task: [] for task in task_list}
        self.metabolic = MetabolicCore()
        self.time_since_reward = 0
        self.time_since_social = 0
        self.time_since_reflection = 0
        self.time_since_sleep = 0

    def run(self) -> None:
        for ep in range(self.cfg.num_episodes):
            metrics = self._run_episode()
            self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)
            self._log_episode(ep, metrics)
            if (ep + 1) % self.cfg.target_update_interval == 0:
                self.organism.update_target()

    def _run_episode(self) -> EpisodeMetrics:
        self.organism.reset()
        obs = self.env.reset()
        prev_obs = None
        last_reward = 0.0
        last_info: Dict[str, Any] = {}
        novelty = 1.0
        prediction_error = 0.0
        self.metabolic.reset()
        self.time_since_reward = 0
        self.time_since_social = 0
        self.time_since_reflection = 0
        self.time_since_sleep = 0
        sleep_steps = 0
        sleep_segments: List[int] = []
        current_sleep_len = 0

        # Prime state.
        obs_dict = obs if isinstance(obs, dict) else asdict(obs)
        intero_signals = self._build_intero_signals()
        state = self.organism.encode_observation(obs_dict, last_reward, novelty, prediction_error, last_info, intero_signals)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.organism.parameters_for_learning(), lr=self.cfg.lr)  # type: ignore[arg-type]

        valence_trace: List[float] = []
        arousal_trace: List[float] = []
        pred_errors: List[float] = []
        task_id = obs_dict.get("task_id", "unknown")
        if task_id not in self.success_counters:
            self.success_counters[task_id] = []
        success = False
        cumulative_reward = 0.0
        energies: List[float] = []
        fatigues: List[float] = []
        ts_rewards: List[float] = []
        ts_socials: List[float] = []
        ts_reflections: List[float] = []
        valence_positive_steps = 0

        for step in range(self.cfg.max_steps_per_episode):
            epsilon_mod = self._modulate_epsilon(state)
            action, _ = self.organism.select_action(state.brain_state_tensor, epsilon_mod)
            sleeping = action == "sleep" or self.organism.is_sleeping

            next_obs, reward, done, info = self.env.step(action if not sleeping else "sleep")
            self.global_step += 1
            cumulative_reward += reward
            if sleeping:
                sleep_steps += 1
                current_sleep_len += 1
            elif current_sleep_len > 0:
                sleep_segments.append(current_sleep_len)
                current_sleep_len = 0

            obs_dict = obs if isinstance(obs, dict) else asdict(obs)
            prediction_error = compute_prediction_error(reward, self.expected_reward)
            next_obs_dict = next_obs if isinstance(next_obs, dict) else asdict(next_obs)
            novelty_transition = compute_novelty(next_obs_dict, obs_dict)
            self._update_time_counters(reward, info, sleeping)
            metabolic_state = self.metabolic.step(
                action_cost=self._action_cost(action),
                arousal=state.emotion.core_affect.arousal,
                learning_load=abs(prediction_error),
                is_sleeping=sleeping,
                sleep_recovery=self.cfg.sleep_recovery_rate,
                sleep_rest=self.cfg.sleep_rest_rate,
            )
            intero_signals = self._build_intero_signals(metabolic_state)

            next_state = self.organism.encode_observation(
                next_obs_dict, reward, novelty_transition, prediction_error, info, intero_signals
            )

            priority = self.plasticity.compute_priority(
                reward=reward,
                novelty=novelty_transition,
                prediction_error=prediction_error,
                emotion_latent=state.emotion.latent,
                core_affect=state.core_affect,
            )

            transition = Transition(
                observation=obs_dict,
                action=action,
                action_idx=self.organism.action_to_idx[action],
                reward=reward,
                next_observation=next_obs_dict,
                done=done,
                brain_state=state.brain_state,
                next_brain_state=next_state.brain_state,
                emotion_latent=state.emotion.latent,
                next_emotion_latent=next_state.emotion.latent,
                hidden_left_in=state.hidden_left_in,
                hidden_right_in=state.hidden_right_in,
                hidden_left=state.hidden_left,
                hidden_right=state.hidden_right,
                next_hidden_left_in=next_state.hidden_left_in,
                next_hidden_right_in=next_state.hidden_right_in,
                next_hidden_left=next_state.hidden_left,
                next_hidden_right=next_state.hidden_right,
                drives=state.drives,
                core_affect=state.core_affect,
                expression=state.expression,
                novelty=novelty_transition,
                prediction_error=prediction_error,
                priority=priority,
                info={"task_id": task_id, "env_info": info},
            )
            self.replay.add(transition)
            train_out = self._train_step()
            if train_out is not None:
                td_errors = train_out["td_errors"]
                self.replay.update_priorities(train_out["indices"], td_errors)

            self.expected_reward = 0.9 * self.expected_reward + 0.1 * reward

            valence_trace.append(state.core_affect["valence"])
            arousal_trace.append(state.core_affect["arousal"])
            pred_errors.append(prediction_error)
            energies.append(metabolic_state.energy)
            fatigues.append(metabolic_state.fatigue)
            ts_rewards.append(self.time_since_reward / max(1, self.cfg.max_steps_per_episode))
            ts_socials.append(self.time_since_social / max(1, self.cfg.max_steps_per_episode))
            ts_reflections.append(self.time_since_reflection / max(1, self.cfg.max_steps_per_episode))
            if state.core_affect["valence"] > 0:
                valence_positive_steps += 1
            if sleeping:
                for _ in range(self.cfg.sleep_replay_multiplier - 1):
                    train_out = self._train_step()
                    if train_out is not None:
                        td_errors = train_out["td_errors"]
                        self.replay.update_priorities(train_out["indices"], td_errors)

            prev_obs = obs
            obs = next_obs
            state = next_state
            last_reward = reward
            last_info = info

            if info.get("task_success"):
                success = True
            if info.get("mirror_contact") and task_id == "goto_mirror":
                success = True
            if info.get("touched_special") and task_id == "touch_object":
                success = True

            if done:
                break

        self.success_counters[task_id].append(success)
        return EpisodeMetrics(
            steps=step + 1,
            reward=cumulative_reward,
            valence_trace=valence_trace,
            arousal_trace=arousal_trace,
            prediction_error_trace=pred_errors,
            task_id=task_id,
            success=success,
            mean_energy=float(np.mean(energies)) if energies else 0.0,
            mean_fatigue=float(np.mean(fatigues)) if fatigues else 0.0,
            mean_ts_reward=float(np.mean(ts_rewards)) if ts_rewards else 0.0,
            mean_ts_social=float(np.mean(ts_socials)) if ts_socials else 0.0,
            mean_ts_reflection=float(np.mean(ts_reflections)) if ts_reflections else 0.0,
            valence_positive_fraction=valence_positive_steps / max(1, len(valence_trace)),
            sleep_fraction=sleep_steps / max(1, step + 1),
            avg_sleep_length=float(np.mean(sleep_segments)) if sleep_segments else (current_sleep_len or 0.0),
            mean_sleep_drive=state.drives.get("sleep_drive", 0.0),
        )

    def _train_step(self) -> Dict[str, Any] | None:
        if self.optimizer is None or len(self.replay) < self.cfg.train_start:
            return None
        if self.global_step % self.cfg.train_interval != 0:
            return None

        samples, weights, indices = self.replay.sample_prioritized(
            self.cfg.batch_size, alpha=self.cfg.priority_alpha, beta=self.cfg.priority_beta
        )
        if not samples:
            return None

        device = self.organism.device
        actions = torch.tensor([t.action_idx for t in samples], dtype=torch.int64, device=device)
        rewards = torch.tensor([t.reward for t in samples], dtype=torch.float32, device=device)
        dones = torch.tensor([float(t.done) for t in samples], dtype=torch.float32, device=device)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(-1)

        states = torch.cat(
            [
                self.organism.encode_replay_state(
                    t.observation, t.emotion_latent, t.hidden_left_in, t.hidden_right_in
                )
                for t in samples
            ],
            dim=0,
        )
        with torch.no_grad():
            next_states = torch.cat(
                [
                    self.organism.encode_replay_state(
                        t.next_observation,
                        t.next_emotion_latent or t.emotion_latent,
                        t.next_hidden_left_in,
                        t.next_hidden_right_in,
                    )
                    for t in samples
                ],
                dim=0,
            )

        q_values = self.organism.q_network(states)  # type: ignore[arg-type]
        q_sa = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            target_q = self.organism.target_network(next_states)  # type: ignore[arg-type]
            max_next = torch.max(target_q, dim=1).values
            targets = rewards + self.cfg.gamma * (1 - dones) * max_next
            targets = torch.clamp(targets, -50.0, 50.0)

        td_error = targets - q_sa
        loss = ((td_error.pow(2)) * weights_t.squeeze()).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.organism.q_network.parameters(), 1.0)  # type: ignore[arg-type]
        self.optimizer.step()

        td_errors_abs = td_error.detach().abs().cpu().tolist()
        wandb.log({"train/q_loss": loss.item()}, step=self.global_step)
        return {"td_errors": td_errors_abs, "indices": indices}

    def _modulate_epsilon(self, state: EncodedState) -> float:
        curiosity = max(0.0, state.drives.get("curiosity_drive", 0.0))
        scale = 1.0 + self.cfg.curiosity_epsilon_scale * curiosity
        return max(self.cfg.epsilon_end, min(1.0, self.epsilon * scale))

    def _log_episode(self, episode_idx: int, metrics: EpisodeMetrics) -> None:
        val_mean = float(np.mean(metrics.valence_trace)) if metrics.valence_trace else 0.0
        arousal_mean = float(np.mean(metrics.arousal_trace)) if metrics.arousal_trace else 0.0
        pred_error_mean = float(np.mean(metrics.prediction_error_trace)) if metrics.prediction_error_trace else 0.0

        success_rates = {
            f"success_rate/{task}": float(np.mean(self.success_counters[task][-10:])) if self.success_counters[task] else 0.0
            for task in self.success_counters
        }

        wandb.log(
            {
                "episode": episode_idx,
                "episode_length": metrics.steps,
                "episode_reward": metrics.reward,
                "mean_valence": val_mean,
                "mean_arousal": arousal_mean,
                "mean_prediction_error": pred_error_mean,
                "task_id": metrics.task_id,
                "task_success": float(metrics.success),
                "epsilon": self.epsilon,
                "mean_energy": metrics.mean_energy,
                "mean_fatigue": metrics.mean_fatigue,
                "mean_time_since_reward": metrics.mean_ts_reward,
                "mean_time_since_social_contact": metrics.mean_ts_social,
                "mean_time_since_reflection": metrics.mean_ts_reflection,
                "valence_positive_fraction": metrics.valence_positive_fraction,
                "sleep_fraction": metrics.sleep_fraction,
                "avg_sleep_length": metrics.avg_sleep_length,
                "mean_sleep_drive": metrics.mean_sleep_drive,
                **success_rates,
            },
            step=self.global_step,
        )

    def _action_cost(self, action: str) -> float:
        if action in ("forward", "backward", "left", "right", "turn_left", "turn_right"):
            return 0.05
        return 0.0

    def _update_time_counters(self, reward: float, info: Dict[str, Any], sleeping: bool) -> None:
        self.time_since_reward = 0 if reward > 0 else self.time_since_reward + 1
        if info.get("mirror_contact"):
            self.time_since_reflection = 0
        else:
            self.time_since_reflection += 1
        if info.get("task_success") and info.get("task_id") in ("social_gaze", "follow_peer"):
            self.time_since_social = 0
        else:
            self.time_since_social += 1
        if sleeping:
            self.time_since_sleep = 0
        else:
            self.time_since_sleep += 1

    def _build_intero_signals(self, metabolic_state=None) -> Dict[str, float]:
        if metabolic_state is None:
            metabolic_state = self.metabolic.state
        return {
            "energy": metabolic_state.energy,
            "fatigue": metabolic_state.fatigue,
            "time_since_reward": min(1.0, self.time_since_reward / max(1, self.cfg.max_steps_per_episode)),
            "time_since_social_contact": min(1.0, self.time_since_social / max(1, self.cfg.max_steps_per_episode)),
            "time_since_reflection": min(1.0, self.time_since_reflection / max(1, self.cfg.max_steps_per_episode)),
            "time_since_sleep": min(1.0, self.time_since_sleep / max(1, self.cfg.max_steps_per_episode)),
        }
