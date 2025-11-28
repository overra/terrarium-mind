import numpy as np

from terrarium.homeostasis import compute_homeostasis_reward


def test_homeostasis_reward_penalizes_extremes() -> None:
    et = np.array([0, 0, 0.9, 0.0, 0, 0, 0.9, 0.9])
    r = compute_homeostasis_reward(et)
    assert r < 0


def test_homeostasis_reward_near_targets() -> None:
    et = np.array([0, 0, 0.2, 0.8, 0, 0, 0.2, 0.1])
    r = compute_homeostasis_reward(et)
    assert r > -0.1
