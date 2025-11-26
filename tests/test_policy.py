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
