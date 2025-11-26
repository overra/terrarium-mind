import torch

from terrarium.organism.hemisphere_core import HemisphereCore


def test_hemisphere_core_updates_hidden() -> None:
    obs_dim = 3
    emotion_dim = 2
    hidden_dim = 4
    core = HemisphereCore(input_dim=obs_dim + emotion_dim, hidden_dim=hidden_dim)
    obs = torch.randn(1, obs_dim)
    emotion = torch.randn(1, emotion_dim)
    hidden = torch.zeros(1, hidden_dim)

    next_hidden = core(obs, emotion, hidden)

    assert next_hidden.shape == (1, hidden_dim)
    assert not torch.allclose(next_hidden, hidden)
