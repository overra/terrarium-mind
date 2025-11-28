from terrarium.env.world import Stage2Config, Stage2Env


def test_body_config_sampling_and_scaling() -> None:
    cfg = Stage2Config(world_size=4.0, max_steps=2, use_body_variation=True, body_move_scale_range=(1.5, 1.5))
    env = Stage2Env(cfg)
    obs = env.reset()
    move_scale = obs["body"]["move_scale"]
    assert move_scale == 1.5
    env.agent.orientation = 0.0
    prev_x = env.agent.x
    env.step("forward")
    assert env.agent.x > prev_x  # scaled up move
