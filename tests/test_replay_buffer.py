from typing import Dict, List

from terrarium.replay import ReplayBuffer, Transition


def _make_transition(priority: float, action: str = "up") -> Transition:
    return Transition(
        observation={"x": 0},
        action=action,
        action_idx=0,
        reward=0.0,
        next_observation={"x": 1},
        done=False,
        brain_state=[0.0],
        next_brain_state=[0.1],
        emotion_latent=[0.0, 0.0],
        drives={"curiosity_drive": 0.5},
        core_affect={"valence": 0.0, "arousal": 0.0},
        expression={},
        novelty=0.5,
        prediction_error=0.1,
        priority=priority,
        info={},
    )


def test_prioritized_sampling_weights() -> None:
    buf = ReplayBuffer(capacity=5, seed=0)
    priorities = [1.0, 2.0, 4.0]
    for p in priorities:
        buf.add(_make_transition(priority=p))

    samples, weights, indices = buf.sample_prioritized(batch_size=3, alpha=0.6, beta=0.4)

    assert len(samples) == 3
    assert len(weights) == 3
    assert len(indices) == 3
    # All weights normalized to <= 1
    assert all(0 < w <= 1.0 for w in weights)

    # Verify weights match expected formula for returned indices
    scaled = [p**0.6 for p in priorities]
    total = sum(scaled)
    probs = [s / total for s in scaled]
    expected_weights = []
    for idx in indices:
        prob = probs[idx]
        w = (1 / (len(priorities) * prob)) ** 0.4
        expected_weights.append(w)
    max_w = max(expected_weights)
    expected_weights = [w / max_w for w in expected_weights]

    for got, exp in zip(weights, expected_weights):
        assert abs(got - exp) < 1e-6
