"""Agent client abstractions for interacting with the WorldServer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol
import random
import numpy as np

from terrarium.config import RunConfig
from terrarium.metabolism import MetabolicCore
from terrarium.organism import Organism
from terrarium.plasticity import PlasticityController
from terrarium.replay import ReplayBuffer, Transition
from terrarium.utils import compute_novelty, compute_prediction_error
from terrarium.qlearning import QTrainer, TransitionBatch
from terrarium.organism.memory import SalientMemory
import torch


class AgentClient(Protocol):
    def init_episode(self, obs: dict) -> None: ...
    def act(self, obs: dict, t: int) -> Any: ...
    def observe(self, obs: dict, reward: float, done: bool, info: dict) -> None: ...
    def on_world_step(self, t: int) -> None: ...
    def get_status(self) -> dict: ...


@dataclass
class AgentState:
    obs: dict
    encoded: Any


class OrganismClient:
    """Wraps an Organism with lightweight RL/replay logic for the world server."""

    def __init__(
        self,
        organism: Organism,
        config: RunConfig,
        replay: Optional[ReplayBuffer] = None,
        plasticity: Optional[PlasticityController] = None,
        learn: bool = True,
        policy_rng: Optional[random.Random] = None,
    ):
        self.org = organism
        self.cfg = config
        self.replay = replay or ReplayBuffer(capacity=config.max_buffer_size, seed=config.seed)
        self.plasticity = plasticity or PlasticityController()
        self.learn = learn
        self.optimizer: torch.optim.Optimizer | None = None
        self.expected_reward = 0.0
        self.epsilon = config.epsilon_start
        self.metabolic = MetabolicCore()
        self.time_since_reward = 0
        self.time_since_social = 0
        self.time_since_reflection = 0
        self.time_since_sleep = 0
        self.state: Optional[AgentState] = None
        self.global_step = 0
        self.current_task = ""
        self.last_action = None
        self.q_trainer = QTrainer(config.gamma)
        self.policy_rng = policy_rng or random.Random(config.seed)
        self.policy_rng.seed(config.seed)
        self.memory = SalientMemory() if config.use_salient_memory else None
        self.last_pred_error = 0.0

    def init_episode(self, obs: dict) -> None:
        self.org.reset()
        self.metabolic.reset()
        self.time_since_reward = 0
        self.time_since_social = 0
        self.time_since_reflection = 0
        self.time_since_sleep = 0
        self.expected_reward = 0.0
        self.current_task = obs.get("task_id", "")
        intero = self._build_intero_signals()
        encoded = self.org.encode_observation(obs, 0.0, 1.0, 0.0, {}, intero)
        self.state = AgentState(obs=obs, encoded=encoded)
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.org.parameters_for_learning(), lr=self.cfg.lr)  # type: ignore[arg-type]

    def act(self, obs: dict, t: int) -> str:
        if self.state is None:
            self.init_episode(obs)
        self.state.obs = obs
        epsilon_mod = self._modulate_epsilon(self.state.encoded)
        action, _ = self.org.select_action(self.state.encoded.brain_state_tensor, epsilon_mod)
        self.last_action = action
        return action

    def observe(self, obs: dict, reward: float, done: bool, info: dict) -> None:
        if self.state is None:
            return
        prev_obs = self.state.obs
        prediction_error = compute_prediction_error(reward, self.expected_reward)
        novelty = compute_novelty(obs, prev_obs)
        sleeping = self.org.is_sleeping
        self._update_time_counters(reward, info, sleeping)
        metabolic_state = self.metabolic.step(
            action_cost=self._action_cost(self.last_action or "stay"),
            arousal=self.state.encoded.emotion.core_affect.arousal,
            learning_load=abs(prediction_error),
            is_sleeping=sleeping,
            sleep_recovery=self.cfg.sleep_recovery_rate,
            sleep_rest=self.cfg.sleep_rest_rate,
        )
        intero = self._build_intero_signals(metabolic_state)
        next_state = self.org.encode_observation(obs, reward, novelty, prediction_error, info, intero)

        if self.learn:
            priority = self.plasticity.compute_priority(
                reward=reward,
                novelty=novelty,
                prediction_error=prediction_error,
                emotion_latent=self.state.encoded.emotion.latent,
                core_affect=self.state.encoded.core_affect,
            )
            transition = Transition(
                observation=prev_obs,
                action=self.last_action or "stay",
                action_idx=self.org.action_to_idx[self.last_action or "stay"],
                reward=reward,
                next_observation=obs,
                done=done,
                brain_state=self.state.encoded.brain_state,
                next_brain_state=next_state.brain_state,
                emotion_latent=self.state.encoded.emotion.latent,
                next_emotion_latent=next_state.emotion.latent,
                core_summary=self.state.encoded.core_summary,
                next_core_summary=next_state.core_summary,
                hidden_left_in=self.state.encoded.hidden_left_in,
                hidden_right_in=self.state.encoded.hidden_right_in,
                hidden_left=self.state.encoded.hidden_left,
                hidden_right=self.state.encoded.hidden_right,
                next_hidden_left_in=next_state.hidden_left_in,
                next_hidden_right_in=next_state.hidden_right_in,
                next_hidden_left=next_state.hidden_left,
                next_hidden_right=next_state.hidden_right,
                drives=self.state.encoded.drives,
                core_affect=self.state.encoded.core_affect,
                expression=self.state.encoded.expression,
                novelty=novelty,
                prediction_error=prediction_error,
                priority=priority,
                info={"task_id": self.current_task, "env_info": info},
            )
            all_vals = [
                reward,
                *self.state.encoded.brain_state,
                *next_state.brain_state,
                *self.state.encoded.emotion.latent,
                *next_state.emotion.latent,
            ]
            if any((not np.isfinite(v)) for v in all_vals):
import wandb
            else:
                self.replay.add(transition)
            if self.memory is not None:
                try:
                    self.memory.consider(
                        np.array(self.state.encoded.core_summary),
                        np.array(self.state.encoded.emotion.latent),
                        reward=reward,
                        confusion=self.state.encoded.drives.get("confusion_level", 0.0),
                        task_id=self.current_task,
                        timestamp=self.global_step,
                        info=info,
                    )
                except Exception:
                    pass
            train_out = self._train_step()
            if train_out is not None:
                self.replay.update_priorities(train_out["indices"], train_out["td_errors"])
            # Periodically sync target network
            if self.global_step % self.cfg.target_update_interval == 0:
                self.org.update_target()
            if sleeping:
                for _ in range(self.cfg.sleep_replay_multiplier - 1):
                    extra = self._train_step()
                    if extra is not None:
                        self.replay.update_priorities(extra["indices"], extra["td_errors"])

        self.expected_reward = 0.9 * self.expected_reward + 0.1 * reward
        self.state = AgentState(obs=obs, encoded=next_state)
        if done:
            self.init_episode(obs)

    def on_world_step(self, t: int) -> None:
        self.global_step += 1
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)

    def get_status(self) -> dict:
        if self.state is None:
            return {}
        return {
            "emotion": self.state.encoded.core_affect,
            "drives": self.state.encoded.drives,
            "energy": self.metabolic.state.energy,
            "fatigue": self.metabolic.state.fatigue,
            "task_id": self.current_task,
            "expression": self.state.encoded.expression,
            "sleeping": self.org.is_sleeping,
            "task_success": False,
        }

    # Helpers
    def _action_cost(self, action: str) -> float:
        if action in ("forward", "backward", "left", "right", "turn_left", "turn_right"):
            return 0.05
        return 0.0

    def _modulate_epsilon(self, encoded) -> float:
        curiosity = max(0.0, encoded.drives.get("curiosity_drive", 0.0))
        scale = 1.0 + self.cfg.curiosity_epsilon_scale * curiosity
        return max(self.cfg.epsilon_end, min(1.0, self.epsilon * scale))

    def _train_step(self) -> Optional[Dict[str, Any]]:
        if self.optimizer is None or len(self.replay) < self.cfg.train_start:
            return None
        samples, weights, indices = self.replay.sample_prioritized(
            self.cfg.batch_size, alpha=self.cfg.priority_alpha, beta=self.cfg.priority_beta
        )
        if not samples:
            return None
        device = self.org.device
        actions = torch.tensor([t.action_idx for t in samples], dtype=torch.int64, device=device)
        rewards = torch.tensor([t.reward for t in samples], dtype=torch.float32, device=device)
        dones = torch.tensor([float(t.done) for t in samples], dtype=torch.float32, device=device)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(-1)

        states = torch.cat(
            [
                self.org.encode_replay_state(
                    t.observation, t.emotion_latent, t.hidden_left_in, t.hidden_right_in
                )
                for t in samples
            ],
            dim=0,
        )
        with torch.no_grad():
            next_states = torch.cat(
                [
                    self.org.encode_replay_state(
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
        loss, td_abs = self.q_trainer.compute_td_loss(self.org, batch)
        aux_loss = torch.tensor(0.0, device=device)
        pred_em_err = torch.tensor(0.0, device=device)
        pred_core_err = torch.tensor(0.0, device=device)
        if self.cfg.use_predictive_head and hasattr(self.org, "predictive_head"):
            action_onehot = torch.nn.functional.one_hot(actions, num_classes=len(self.org.action_space)).float()
            emo_t = torch.tensor([t.emotion_latent for t in samples], dtype=torch.float32, device=device)
            emo_tp1 = torch.tensor(
                [t.next_emotion_latent if t.next_emotion_latent is not None else t.emotion_latent for t in samples],
                dtype=torch.float32,
                device=device,
            )
            core_t = torch.tensor([t.core_summary or t.brain_state[: self.org.hidden_dim * 2] for t in samples], dtype=torch.float32, device=device)
            core_tp1 = torch.tensor(
                [t.next_core_summary or t.next_brain_state[: self.org.hidden_dim * 2] for t in samples],
                dtype=torch.float32,
                device=device,
            )
            pred_emo, pred_core = self.org.predictive_head(emo_t, core_t, action_onehot)
            pred_em_err = (pred_emo - emo_tp1)
            pred_core_err = (pred_core - core_tp1)
            loss_em = (pred_em_err.pow(2)).mean()
            loss_core = (pred_core_err.pow(2)).mean()
            aux_loss = self.cfg.lambda_pred_emotion * loss_em + self.cfg.lambda_pred_core * loss_core
            loss = loss + aux_loss

        demo_mask = torch.tensor([1 if t.info.get("is_demo") else 0 for t in samples], device=device).bool()
        if self.cfg.use_observational_learning and demo_mask.any():
            logits = self.org.q_network(states)  # type: ignore[arg-type]
            imitation = torch.nn.functional.cross_entropy(logits[demo_mask], actions[demo_mask])
            loss = loss + self.cfg.lambda_imitation * imitation

        self.q_trainer.apply_gradients(self.optimizer, loss, self.org.parameters_for_learning())
        td_errors_abs = td_abs.cpu().tolist()
        self.last_pred_error = float((pred_em_err.abs().mean() + pred_core_err.abs().mean()).item()) if aux_loss.numel() > 0 else 0.0
        return {"td_errors": td_errors_abs, "indices": indices}

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
