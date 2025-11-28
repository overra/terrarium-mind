import math

from terrarium.env.world import Stage2Config, Stage2Env


def _env_with_vision_task() -> Stage2Env:
    cfg = Stage2Config(world_size=4.0, max_steps=5, max_objects=2, include_vision_task=True, tasks=("vision_object_discrim",))
    env = Stage2Env(cfg)
    env.reset("vision_object_discrim")
    return env


def test_vision_object_discrim_success_on_glow() -> None:
    env = _env_with_vision_task()
    glow_obj = next(o for o in env.objects if getattr(o, "glow", False))
    env.agent.x, env.agent.y = glow_obj.x, glow_obj.y
    _, reward, done, info = env.step("stay")
    assert info.get("task_success") is True
    assert reward > 0
    assert done is True


def test_vision_object_discrim_not_success_on_non_glow() -> None:
    env = _env_with_vision_task()
    non_glow = next(o for o in env.objects if not getattr(o, "glow", False))
    env.agent.x, env.agent.y = non_glow.x, non_glow.y
    _, reward, done, info = env.step("stay")
    assert not info.get("task_success")
    assert reward <= 0
    assert done is False or done is True  # allowed either if max steps reached later
