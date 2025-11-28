import numpy as np

from terrarium.env.world import Stage2Config, Stage2Env
from terrarium.world import World


def test_head_orientation_shifts_retina() -> None:
    cfg = Stage2Config(world_size=4.0, max_steps=1, max_objects=1, enable_head_yaw=True)
    env = Stage2Env(cfg)
    env.reset()
    obj = env.objects[0]
    env.agent.x, env.agent.y = 2.0, 2.0
    obj.x, obj.y = 2.0, 3.0  # ahead in +y relative to body if head faces +y
    env.agent.orientation = 0.0  # body facing +x
    env.agent.head_offset = 0.0
    world = World(env)
    obs_center = world._augment_obs(env._observe())
    env.agent.head_offset = 0.5  # turn head left (toward +y)
    obs_left = world._augment_obs(env._observe())
    env.agent.head_offset = -0.5  # turn head right (toward -y)
    obs_right = world._augment_obs(env._observe())

    def obj_energy(obs):
        ret = np.array(obs["retina"])
        # objects channel
        if ret.shape[0] <= ret.shape[-1]:
            ch = ret[1]
        else:
            ch = np.transpose(ret, (2, 0, 1))[1]
        return ch.sum()

    assert obj_energy(obs_left) > obj_energy(obs_right)
