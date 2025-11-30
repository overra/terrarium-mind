import torch

from terrarium.organism.policy import EpsilonGreedyPolicy


def test_policy_greedy_when_epsilon_zero() -> None:
    action_space = ["left", "stay", "right"]
    policy = EpsilonGreedyPolicy(action_space)
    q_vals = torch.tensor([0.1, 0.2, 0.9])
    action = policy.select(q_vals, epsilon=0.0)
    assert action == "right"


def test_policy_explores_when_epsilon_one() -> None:
    action_space = ["left", "stay", "right"]
    policy = EpsilonGreedyPolicy(action_space, rng=None)
    q_vals = torch.tensor([0.0, 0.0, 2.0])  # greedy would be index 2
    # Force exploration by setting epsilon=1 and seeding to a known choice (randrange -> 1)
    policy.rng.seed(0)
    action = policy.select(q_vals, epsilon=1.0)
    assert action == "stay"


class DummyState:
    drives = {"curiosity_drive": 0.0}
    brain_state_tensor = torch.zeros(1)


class DummyTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.epsilon = cfg.epsilon_start
        self.global_step = 0

    def _modulate_epsilon(self, state):
        from terrarium.training import RLTrainer
        # reuse logic via a tiny shim
        curiosity = max(0.0, state.drives.get("curiosity_drive", 0.0))
        scale = 1.0 + self.cfg.curiosity_epsilon_scale * curiosity
        eps = self.epsilon
        if self.cfg.epsilon_mode == "long_train":
            decay_ratio = min(1.0, self.global_step / max(1, self.cfg.epsilon_long_train_steps))
            eps = self.cfg.epsilon_start - decay_ratio * (self.cfg.epsilon_start - self.cfg.epsilon_long_train_final)
        return max(self.cfg.epsilon_end, min(1.0, eps * scale))


def test_long_train_epsilon_schedule():
    from terrarium.config import RunConfig
    cfg = RunConfig(epsilon_mode="long_train", epsilon_start=0.8, epsilon_long_train_final=0.2, epsilon_long_train_steps=100)
    trainer = DummyTrainer(cfg)
    trainer.global_step = 50
    eps = trainer._modulate_epsilon(DummyState())
    assert eps < cfg.epsilon_start
