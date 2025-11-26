from terrarium.env import GridWorldConfig, GridWorldEnv


def test_goto_mirror_task_reward_and_done() -> None:
    env = GridWorldEnv(GridWorldConfig(seed=0))
    env.reset(task_id="goto_mirror")
    # Place agent left of mirror and step right onto it.
    mx, my = env.mirror_pos
    env.agent_pos = (max(0, mx - 1), my)
    obs, reward, done, info = env.step("right")
    assert done is True
    assert reward > 0.9  # step penalty (-0.01) + task reward (+1.0)
    assert info.get("mirror_contact") is True
    assert obs.task_id == "goto_mirror"


def test_touch_object_task_reward_and_done() -> None:
    env = GridWorldEnv(GridWorldConfig(seed=1))
    env.reset(task_id="touch_object")
    sx, sy = env.special_object  # type: ignore[assignment]
    # Place agent left of special object if possible, else right.
    if sx > 0:
        env.agent_pos = (sx - 1, sy)
        action = "right"
    else:
        env.agent_pos = (sx + 1, sy)
        action = "left"
    obs, reward, done, info = env.step(action)
    assert done is True
    assert reward > 0.9  # step penalty (-0.01) + task reward (+1.0)
    assert info.get("touched_special") is True
    assert obs.task_id == "touch_object"
