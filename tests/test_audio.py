import math

from terrarium.env.world import Stage2Config, Stage2Env
from terrarium.world import World


def _env_sound() -> Stage2Env:
    cfg = Stage2Config(world_size=4.0, max_steps=5, max_objects=1, include_go_to_sound=True, tasks=("go_to_sound",))
    env = Stage2Env(cfg)
    env.reset("go_to_sound")
    return env


def test_audio_panning_left_vs_right() -> None:
    env = _env_sound()
    world = World(env)
    # place agent at center, orientation 0 (facing +x)
    env.agent.x, env.agent.y, env.agent.orientation = 2.0, 2.0, 0.0
    env.sound_source.x, env.sound_source.y = 2.0, 1.0  # below -> left ear (negative sin)
    obs = world._augment_obs(env._observe())
    audio = obs["audio"]
    left1, right1 = audio["left"], audio["right"]
    env.sound_source.y = 3.0  # above -> right ear
    obs2 = world._augment_obs(env._observe())
    left2, right2 = obs2["audio"]["left"], obs2["audio"]["right"]
    assert left1 > right1
    assert right2 > left2


def test_go_to_sound_success() -> None:
    env = _env_sound()
    # Move agent onto sound source
    env.agent.x, env.agent.y = env.sound_source.x, env.sound_source.y
    _, reward, done, info = env.step("stay")
    assert info.get("task_success") is True
    assert reward > 0
    assert done is True


def test_audio_panning_right_source() -> None:
    env = _env_sound()
    world = World(env)
    env.agent.x, env.agent.y, env.agent.orientation = 2.0, 2.0, 0.0
    env.sound_source.x, env.sound_source.y = 3.0, 2.0  # to the right
    audio = world._augment_obs(env._observe())["audio"]
    assert audio["right"] > 0 and audio["left"] > 0


def test_audio_panning_left_source() -> None:
    env = _env_sound()
    world = World(env)
    env.agent.x, env.agent.y, env.agent.orientation = 2.0, 2.0, 0.0
    env.sound_source.x, env.sound_source.y = 1.0, 2.0  # to the left
    audio = world._augment_obs(env._observe())["audio"]
    assert audio["left"] > 0 and audio["right"] > 0


def test_audio_panning_front_source() -> None:
    env = _env_sound()
    world = World(env)
    env.agent.x, env.agent.y, env.agent.orientation = 2.0, 2.0, 0.0
    env.sound_source.x, env.sound_source.y = 3.0, 2.0  # front (+x)
    audio = world._augment_obs(env._observe())["audio"]
    assert audio["left"] > 0 or audio["right"] > 0


def test_audio_panning_back_source() -> None:
    env = _env_sound()
    world = World(env)
    env.agent.x, env.agent.y, env.agent.orientation = 2.0, 2.0, 0.0
    env.sound_source.x, env.sound_source.y = 1.0, 2.0  # back (-x)
    audio = world._augment_obs(env._observe())["audio"]
    assert audio["left"] > 0 or audio["right"] > 0


def test_audio_panning_side_source() -> None:
    env = _env_sound()
    world = World(env)
    env.agent.x, env.agent.y, env.agent.orientation = 2.0, 2.0, 0.0
    env.sound_source.x, env.sound_source.y = 2.0, 3.0  # side (+y)
    audio = world._augment_obs(env._observe())["audio"]
    assert audio["left"] > 0 and audio["right"] > 0
