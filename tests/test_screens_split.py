import torch

from terrarium.backend import TorchBackend
from terrarium.organism import Organism


def make_org(max_objects: int = 3) -> Organism:
    return Organism(
        action_space=["stay"],
        backend=TorchBackend(),
        hidden_dim=8,
        bridge_dim=8,
        grid_size=8,
        max_steps=10,
        task_ids=["t"],
        policy_rng=None,
        max_objects=max_objects,
        max_peers=1,
        max_reflections=1,
    )


def extract_objects(org: Organism, obs):
    tensors = org._split_observation(obs)
    return tensors["objects"].squeeze(0)  # [slots, feat_dim]


def test_screens_priority_over_objects_when_exceeding_capacity() -> None:
    org = make_org(max_objects=2)
    obs = {
        "self": {"pos": [0, 0], "orientation": 0.0, "velocity": [0, 0]},
        "time": {},
        "objects": [
            {"rel_x": 1.0, "rel_y": 0.0, "size": 1.0, "visible": 1.0, "type_id": 1.0},
            {"rel_x": 2.0, "rel_y": 0.0, "size": 1.0, "visible": 1.0, "type_id": 2.0},
        ],
        "screens": [
            {"rel_x": 3.0, "rel_y": 0.0, "size": 1.0, "visible": 1.0, "content_id": 5.0, "brightness": 0.0},
        ],
    }
    objs = extract_objects(org, obs)
    # Expect first slot to be screen (rel_x=3), second slot first object (rel_x=1)
    assert torch.isclose(objs[0, 0], torch.tensor(3.0))
    assert torch.isclose(objs[1, 0], torch.tensor(1.0))


def test_all_screens_included_when_room_and_objects_fill_remaining() -> None:
    org = make_org(max_objects=3)
    obs = {
        "self": {"pos": [0, 0], "orientation": 0.0, "velocity": [0, 0]},
        "time": {},
        "objects": [{"rel_x": 1.0, "rel_y": 0.0, "size": 1.0, "visible": 1.0, "type_id": 1.0}],
        "screens": [
            {"rel_x": 2.0, "rel_y": 0.0, "size": 1.0, "visible": 1.0, "content_id": 5.0, "brightness": 0.0},
            {"rel_x": 3.0, "rel_y": 0.0, "size": 1.0, "visible": 1.0, "content_id": 6.0, "brightness": 0.0},
        ],
    }
    objs = extract_objects(org, obs)
    # Two screens then object
    assert torch.isclose(objs[0, 0], torch.tensor(2.0))
    assert torch.isclose(objs[1, 0], torch.tensor(3.0))
    assert torch.isclose(objs[2, 0], torch.tensor(1.0))


def test_only_objects_when_no_screens() -> None:
    org = make_org(max_objects=2)
    obs = {
        "self": {"pos": [0, 0], "orientation": 0.0, "velocity": [0, 0]},
        "time": {},
        "objects": [
            {"rel_x": 1.0, "rel_y": 0.0, "size": 1.0, "visible": 1.0, "type_id": 1.0},
            {"rel_x": 2.0, "rel_y": 0.0, "size": 1.0, "visible": 1.0, "type_id": 2.0},
        ],
        "screens": [],
    }
    objs = extract_objects(org, obs)
    assert torch.isclose(objs[0, 0], torch.tensor(1.0))
    assert torch.isclose(objs[1, 0], torch.tensor(2.0))


def test_exact_capacity_mix() -> None:
    org = make_org(max_objects=2)
    obs = {
        "self": {"pos": [0, 0], "orientation": 0.0, "velocity": [0, 0]},
        "time": {},
        "objects": [{"rel_x": 1.0, "rel_y": 0.0, "size": 1.0, "visible": 1.0, "type_id": 1.0}],
        "screens": [{"rel_x": 2.0, "rel_y": 0.0, "size": 1.0, "visible": 1.0, "content_id": 5.0, "brightness": 0.0}],
    }
    objs = extract_objects(org, obs)
    assert torch.isclose(objs[0, 0], torch.tensor(2.0))
    assert torch.isclose(objs[1, 0], torch.tensor(1.0))


def test_zero_max_objects_or_no_screens_handled() -> None:
    org = make_org(max_objects=0)
    obs = {
        "self": {"pos": [0, 0], "orientation": 0.0, "velocity": [0, 0]},
        "time": {},
        "objects": [{"rel_x": 1.0, "rel_y": 0.0, "size": 1.0, "visible": 1.0, "type_id": 1.0}],
        "screens": [{"rel_x": 2.0, "rel_y": 0.0, "size": 1.0, "visible": 1.0, "content_id": 5.0, "brightness": 0.0}],
    }
    objs = extract_objects(org, obs)
    assert objs.shape[0] == 0

