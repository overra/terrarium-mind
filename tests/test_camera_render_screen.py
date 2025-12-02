import numpy as np

from terrarium.env.world import Stage2Config, Stage2Env, Screen
from terrarium.world import World


def test_render_camera_with_screen() -> None:
    cfg = Stage2Config(enable_camera=True, camera_size=16, seed=123, max_screens=1)
    env = Stage2Env(cfg)
    world = World(env)
    env.agent.x = cfg.world_size / 2
    env.agent.y = cfg.world_size / 2
    env.agent.orientation = 0.0
    env.screens = [Screen(x=env.agent.x + 1.0, y=env.agent.y, size=0.4, brightness=1.0)]
    img = world.render_camera(size=16)
    assert img.shape == (16, 16, 3)
    assert img.dtype == np.uint8
    assert img.sum() > 0
