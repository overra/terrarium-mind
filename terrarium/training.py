"""RL training loop wiring env, organism, and prioritized replay."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple
import random

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
from .qlearning import QTrainer, TransitionBatch
from .organism.memory import SalientMemory
from .homeostasis import compute_homeostasis_reward, HomeostasisTracker
from .vis.retina_logging import retina_to_image


@dataclass
class EpisodeMetrics:
    steps: int
    reward: float
    valence_trace: List[float] = field(default_factory=list)
    arousal_trace: List[float] = field(default_factory=list)
    tiredness_trace: List[float] = field(default_factory=list)
    social_satiation_trace: List[float] = field(default_factory=list)
    curiosity_trace: List[float] = field(default_factory=list)
    safety_trace: List[float] = field(default_factory=list)
    sleep_urge_trace: List[float] = field(default_factory=list)
    confusion_trace: List[float] = field(default_factory=list)
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
    retina_sample: Any | None = None
    mean_audio_left: float = 0.0
    mean_audio_right: float = 0.0
    mean_head_offset: float = 0.0


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
        task_list = list(cfg_tasks.tasks) if cfg_tasks else []
        env_task_list = getattr(env, "task_list", [])
        for t in env_task_list:
            if t not in task_list:
                task_list.append(t)
        self.success_counters: Dict[str, List[bool]] = {task: [] for task in task_list}
        self.metabolic = MetabolicCore()
        self.time_since_reward = 0
        self.time_since_social = 0
        self.time_since_reflection = 0
        self.time_since_sleep = 0
        self.retina_logged = 0
        self.q_trainer = QTrainer(config.gamma)
        self.rng = random.Random(config.seed)
        self.last_pred_error = 0.0
        self.memory = SalientMemory() if config.use_salient_memory else None
        self.homeo = HomeostasisTracker()

    def run(self) -> None:
        for ep in range(self.cfg.num_episodes):
            demo_episode = self.cfg.use_observational_learning and (self.rng.random() < self.cfg.demo_fraction)
            metrics = self._run_episode(demo_episode=demo_episode)
            self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)
            self._log_episode(ep, metrics)
            if (ep + 1) % self.cfg.target_update_interval == 0:
                self.organism.update_target()

    def _run_episode(self, demo_episode: bool = False) -> EpisodeMetrics:
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
        tiredness_trace: List[float] = []
        social_sat_trace: List[float] = []
        curiosity_trace: List[float] = []
        safety_trace: List[float] = []
        sleep_urge_trace: List[float] = []
        confusion_trace: List[float] = []
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
        retina_sample = obs_dict.get("retina")
        audio_lefts: List[float] = []
        audio_rights: List[float] = []
        head_offsets: List[float] = []

        for step in range(self.cfg.max_steps_per_episode):
            epsilon_mod = self._modulate_epsilon(state)
            if demo_episode:
                action = "stay"
            else:
                action, _ = self.organism.select_action(state.brain_state_tensor, epsilon_mod)
            sleeping = action == "sleep" or self.organism.is_sleeping

            next_obs, env_reward, done, info = self.env.step(action if not sleeping else "sleep")
            self.global_step += 1
            h_reward = 0.0
            if self.cfg.use_homeostasis:
                h_reward = compute_homeostasis_reward(np.array(state.emotion.latent))
                self.homeo.update(state.drives.get("tiredness", 0.0), state.drives.get("confusion_level", 0.0))
                h_reward += self.homeo.overload_penalty(penalty=self.cfg.homeostasis_chronic_penalty)
            reward = env_reward + (self.cfg.homeostasis_weight * h_reward if self.cfg.use_homeostasis else 0.0)
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
                core_summary=state.core_summary,
                next_core_summary=next_state.core_summary,
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
                info={"task_id": task_id, "env_info": info, "is_demo": demo_episode},
            )
            self.replay.add(transition)
            if self.memory is not None:
                try:
                    self.memory.consider(
                        np.array(state.core_summary),
                        np.array(state.emotion.latent),
                        reward=reward,
                        confusion=state.drives.get("confusion_level", 0.0),
                        task_id=task_id,
                        timestamp=self.global_step,
                        info=info,
                    )
                except Exception:
                    pass
            train_out = self._train_step()
            if train_out is not None:
                td_errors = train_out["td_errors"]
                self.replay.update_priorities(train_out["indices"], td_errors)

            self.expected_reward = 0.9 * self.expected_reward + 0.1 * reward

            valence_trace.append(state.core_affect["valence"])
            arousal_trace.append(state.core_affect["arousal"])
            tiredness_trace.append(state.drives.get("tiredness", 0.0))
            social_sat_trace.append(state.drives.get("social_satiation", 0.0))
            curiosity_trace.append(state.drives.get("curiosity_drive", 0.0))
            safety_trace.append(state.drives.get("safety_drive", 0.0))
            sleep_urge_trace.append(state.drives.get("sleep_urge", 0.0))
            confusion_trace.append(state.drives.get("confusion_level", 0.0))
            pred_errors.append(prediction_error)
            energies.append(metabolic_state.energy)
            fatigues.append(metabolic_state.fatigue)
            ts_rewards.append(self.time_since_reward / max(1, self.cfg.max_steps_per_episode))
            ts_socials.append(self.time_since_social / max(1, self.cfg.max_steps_per_episode))
            ts_reflections.append(self.time_since_reflection / max(1, self.cfg.max_steps_per_episode))
            audio = obs_dict.get("audio", {})
            audio_lefts.append(float(audio.get("left", 0.0)))
            audio_rights.append(float(audio.get("right", 0.0)))
            head_angle = obs_dict.get("self", {}).get("head_orientation", obs_dict.get("self", {}).get("orientation", 0.0))
            body_angle = obs_dict.get("self", {}).get("body_orientation", head_angle)
            head_offsets.append(abs(head_angle - body_angle))
            if state.core_affect["valence"] > 0:
                valence_positive_steps += 1
            if "retina" in next_obs_dict:
                retina_sample = next_obs_dict["retina"]
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
            tiredness_trace=tiredness_trace,
            social_satiation_trace=social_sat_trace,
            curiosity_trace=curiosity_trace,
            safety_trace=safety_trace,
            sleep_urge_trace=sleep_urge_trace,
            confusion_trace=confusion_trace,
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
            retina_sample=retina_sample,
            mean_audio_left=float(np.mean(audio_lefts)) if audio_lefts else 0.0,
            mean_audio_right=float(np.mean(audio_rights)) if audio_rights else 0.0,
            mean_head_offset=float(np.mean(head_offsets)) if head_offsets else 0.0,
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

        batch = TransitionBatch(
            states=states,
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            dones=dones,
            weights=weights_t,
        )
        loss, td_abs = self.q_trainer.compute_td_loss(self.organism, batch)

        aux_loss = torch.tensor(0.0, device=device)
        pred_em_err = torch.tensor(0.0, device=device)
        pred_core_err = torch.tensor(0.0, device=device)
        if self.cfg.use_predictive_head and hasattr(self.organism, "predictive_head"):
            action_onehot = torch.nn.functional.one_hot(actions, num_classes=len(self.organism.action_space)).float()
            emo_t = torch.tensor([t.emotion_latent for t in samples], dtype=torch.float32, device=device)
            emo_tp1 = torch.tensor(
                [t.next_emotion_latent if t.next_emotion_latent is not None else t.emotion_latent for t in samples],
                dtype=torch.float32,
                device=device,
            )
            core_t = torch.tensor([t.core_summary or t.brain_state[: self.organism.hidden_dim * 2] for t in samples], dtype=torch.float32, device=device)
            core_tp1 = torch.tensor(
                [t.next_core_summary or t.next_brain_state[: self.organism.hidden_dim * 2] for t in samples],
                dtype=torch.float32,
                device=device,
            )
            pred_emo, pred_core = self.organism.predictive_head(emo_t, core_t, action_onehot)
            pred_em_err = (pred_emo - emo_tp1)
            pred_core_err = (pred_core - core_tp1)
            loss_em = (pred_em_err.pow(2)).mean()
            loss_core = (pred_core_err.pow(2)).mean()
            aux_loss = self.cfg.lambda_pred_emotion * loss_em + self.cfg.lambda_pred_core * loss_core
            loss = loss + aux_loss

        # Imitation loss on observational (demo) transitions
        demo_mask = torch.tensor([1 if t.info.get("is_demo") else 0 for t in samples], device=device).bool()
        if self.cfg.use_observational_learning and demo_mask.any():
            logits = self.organism.q_network(states)  # type: ignore[arg-type]
            imitation = torch.nn.functional.cross_entropy(logits[demo_mask], actions[demo_mask])
            loss = loss + self.cfg.lambda_imitation * imitation

        self.q_trainer.apply_gradients(self.optimizer, loss, self.organism.parameters_for_learning())
        td_errors_abs = td_abs.cpu().tolist()
        self.last_pred_error = (
            float((pred_em_err.abs().mean() + pred_core_err.abs().mean()).item())
            if aux_loss.numel() > 0 and aux_loss.item() > 0
            else 0.0
        )
        wandb.log(
            {
                "train/q_loss": loss.item(),
                "train/pred_emotion_loss": float(aux_loss.item()) if aux_loss.numel() > 0 else 0.0,
            },
            step=self.global_step,
        )
        return {"td_errors": td_errors_abs, "indices": indices}

    def _modulate_epsilon(self, state: EncodedState) -> float:
        curiosity = max(0.0, state.drives.get("curiosity_drive", 0.0))
        scale = 1.0 + self.cfg.curiosity_epsilon_scale * curiosity
        return max(self.cfg.epsilon_end, min(1.0, self.epsilon * scale))

    def _log_episode(self, episode_idx: int, metrics: EpisodeMetrics) -> None:
        val_mean = float(np.mean(metrics.valence_trace)) if metrics.valence_trace else 0.0
        arousal_mean = float(np.mean(metrics.arousal_trace)) if metrics.arousal_trace else 0.0
        tired_mean = float(np.mean(metrics.tiredness_trace)) if metrics.tiredness_trace else 0.0
        social_sat_mean = float(np.mean(metrics.social_satiation_trace)) if metrics.social_satiation_trace else 0.0
        curiosity_mean = float(np.mean(metrics.curiosity_trace)) if metrics.curiosity_trace else 0.0
        safety_mean = float(np.mean(metrics.safety_trace)) if metrics.safety_trace else 0.0
        sleep_urge_mean = float(np.mean(metrics.sleep_urge_trace)) if metrics.sleep_urge_trace else 0.0
        confusion_mean = float(np.mean(metrics.confusion_trace)) if metrics.confusion_trace else 0.0
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
                "reward/total": metrics.reward,
                "mean_valence": val_mean,
                "mean_arousal": arousal_mean,
                "mean_tiredness": tired_mean,
                "mean_social_satiation": social_sat_mean,
                "mean_curiosity_drive": curiosity_mean,
                "mean_safety_drive": safety_mean,
                "mean_sleep_urge": sleep_urge_mean,
                "mean_confusion_level": confusion_mean,
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
                "mean_audio_left": metrics.mean_audio_left,
                "mean_audio_right": metrics.mean_audio_right,
                "mean_head_offset": metrics.mean_head_offset,
                **success_rates,
            },
            step=self.global_step,
        )
        if (
            self.cfg.log_retina
            and metrics.retina_sample is not None
            and episode_idx % max(1, self.cfg.retina_log_interval_episodes) == 0
            and self.retina_logged < self.cfg.retina_max_snapshots_per_run
        ):
            retina_np = np.array(metrics.retina_sample, dtype=np.float32)
            img = retina_to_image(retina_np)
            wandb.log({"retina/last_frame": wandb.Image(img)}, step=self.global_step)
            self.retina_logged += 1
        if metrics.retina_sample is not None:
            retina_np = np.array(metrics.retina_sample, dtype=np.float32)
            if retina_np.shape[0] <= retina_np.shape[-1]:  # C,H,W
                channels = retina_np
            else:
                channels = np.transpose(retina_np, (2, 0, 1))
            intensity_mean = float(channels[5].mean()) if channels.shape[0] > 5 else 0.0
            motion_mean = float(channels[6].mean()) if channels.shape[0] > 6 else 0.0
            wandb.log({"vision/mean_intensity": intensity_mean, "vision/mean_motion": motion_mean}, step=self.global_step)
        if self.memory is not None:
            wandb.log({"memory/size": len(self.memory.entries)}, step=self.global_step)

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
        if info.get("social_contact") or (info.get("task_success") and info.get("task_id") in ("social_gaze", "follow_peer")):
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
            "confusion_extra": self.last_pred_error,
        }
