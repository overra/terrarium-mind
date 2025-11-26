import math

from terrarium.env import Stage2Config, Stage2Env


def _make_env(task_id: str) -> Stage2Env:
    cfg = Stage2Config(world_size=6.0, max_steps=10, max_objects=2, max_peers=1, max_reflections=1, seed=0)
    env = Stage2Env(cfg)
    env.reset(task_id=task_id)
    return env


def test_examine_reflection_success() -> None:
    env = _make_env("examine_reflection")
    env.agent.x = env.mirrors[0].x - 0.4
    env.agent.y = 1.0
    env.agent.orientation = 0.0  # face +x toward mirror
    _, reward, done, info = env.step("stay")
    assert info.get("task_success") is True
    assert reward > 0.9
    assert done is True


def test_follow_peer_success() -> None:
    env = _make_env("follow_peer")
    peer = env.peers[0]
    peer.x, peer.y = 2.0, 2.0
    env.agent.x, env.agent.y = 2.0, 3.2  # distance ~1.2
    _, reward, done, info = env.step("stay")
    assert info.get("task_success") is True
    assert reward > 0.9
    assert done is True


def test_social_gaze_success() -> None:
    env = _make_env("social_gaze")
    peer = env.peers[0]
    peer.x, peer.y = 3.0, 3.0
    env.agent.x, env.agent.y = 2.0, 3.0
    env.agent.orientation = 0.0  # facing +x toward peer
    _, reward, done, info = env.step("stay")
    assert info.get("task_success") is True
    assert reward > 0.9
    assert done is True


def test_novel_object_investigation_success_marks_seen() -> None:
    env = _make_env("novel_object_investigation")
    obj = env.objects[0]
    env.agent.x, env.agent.y = obj.x - 0.5, obj.y
    _, reward, done, info = env.step("right")  # move closer
    assert info.get("task_success") is True
    assert reward > 0.9
    assert done is True
    assert obj.seen is True


def test_reflection_relative_position() -> None:
    env = _make_env("examine_reflection")
    mirror_x = env.mirrors[0].x
    env.agent.x = mirror_x - 1.0
    env.agent.y = 1.0
    obs, _, _, _ = env.step("stay")
    ref = obs["mirror_reflections"][0]
    # Reflection should be roughly 2 units to the right in ego frame
    assert ref["rel_x"] > 1.5
