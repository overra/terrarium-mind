import numpy as np
import torch

from terrarium.organism.organism import Organism
from terrarium.backend import TorchBackend


def _base_observation() -> dict:
    return {
        "self": {
            "pos": [0.0, 0.0],
            "orientation": 0.0,
            "velocity": [0.0, 0.0],
            "body": {},
            "head_orientation": 0.0,
        },
        "objects": [],
        "peers": [],
        "mirror_reflections": [],
        "time": {"episode_step_norm": 0.0, "world_time_norm": 0.0},
        "audio": {"left": 0.0, "right": 0.0},
    }


def test_encode_with_camera_mode() -> None:
    obs = _base_observation()
    obs["camera"] = np.zeros((8, 8, 3), dtype=np.uint8)
    org = Organism(
        action_space=["stay"],
        backend=TorchBackend(),
        vision_mode="camera",
        camera_channels=3,
    )
    state = org.encode_observation(obs, reward=0.0, novelty=0.0, prediction_error=0.0, info={}, intero_signals={})
    assert torch.isfinite(state.brain_state_tensor).all()


def test_encode_with_retina_mode() -> None:
    obs = _base_observation()
    obs["retina"] = np.zeros((7, 4, 4), dtype=np.float32)
    org = Organism(
        action_space=["stay"],
        backend=TorchBackend(),
        vision_mode="retina",
    )
    state = org.encode_observation(obs, reward=0.0, novelty=0.0, prediction_error=0.0, info={}, intero_signals={})
    assert torch.isfinite(state.brain_state_tensor).all()
