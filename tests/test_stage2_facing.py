import math

from terrarium.env.world import Stage2Env, Stage2Config


def _env_with_mirrors(mirror_positions):
    cfg = Stage2Config(world_size=10.0, max_steps=5, max_objects=0, max_peers=0, max_reflections=len(mirror_positions), seed=0)
    env = Stage2Env(cfg)
    env.reset(task_id="examine_reflection")
    env.mirrors = []
    for x in mirror_positions:
        env.mirrors.append(type("M", (), {"x": x}))
    return env


def test_is_facing_mirror_no_mirrors() -> None:
    env = _env_with_mirrors([])
    env.agent.x = 5.0
    env.agent.orientation = 0.0
    assert env._is_facing_mirror() is False


def test_is_facing_mirror_chooses_nearest() -> None:
    env = _env_with_mirrors([1.0, 9.0])
    env.agent.x = 1.2
    env.agent.orientation = math.pi  # facing -x toward mirror at x=1.0
    assert env._is_facing_mirror() is True
    env.agent.orientation = 0.0  # facing +x away from mirror at x=1.0
    assert env._is_facing_mirror() is False


def test_is_facing_mirror_positive_x_direction() -> None:
    env = _env_with_mirrors([6.0])
    env.agent.x = 4.0
    env.agent.orientation = 0.0  # facing +x toward mirror
    assert env._is_facing_mirror() is True


def test_is_facing_mirror_negative_x_direction() -> None:
    env = _env_with_mirrors([2.0])
    env.agent.x = 4.0
    env.agent.orientation = 0.0  # facing +x away from mirror (mirror is to the left)
    assert env._is_facing_mirror() is False
