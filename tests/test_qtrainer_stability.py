import torch

from terrarium.qlearning import QTrainer, TransitionBatch
from terrarium.organism import Organism
from terrarium.backend import TorchBackend
from terrarium.env import Stage2Config, Stage2Env
from terrarium.world import World
from terrarium.organism import Organism


def test_qtrainer_loss_finite_on_synthetic_batch() -> None:
    trainer = QTrainer(gamma=0.95)
    # Build tiny organism for shape only
    env_cfg = Stage2Config(world_size=4.0, max_steps=1)
    env = Stage2Env(env_cfg)
    backend = TorchBackend()
    org = Organism(
        action_space=env.action_space,
        backend=backend,
        hidden_dim=8,
        bridge_dim=8,
        grid_size=4,
        max_steps=1,
        task_ids=env.cfg.tasks,
        policy_rng=None,
        max_objects=env.cfg.max_objects,
        max_peers=env.cfg.max_peers,
        max_reflections=env.cfg.max_reflections,
    )
    # initialize modules
    org._ensure_modules(obs_dim=org.slot_input_dim, emotion_dim=8)
    in_dim = org.q_network.net[0].in_features
    batch = TransitionBatch(
        states=torch.zeros(2, in_dim),
        next_states=torch.zeros(2, in_dim),
        actions=torch.tensor([0, 0]),
        rewards=torch.tensor([0.0, 0.0]),
        dones=torch.tensor([0.0, 0.0]),
        weights=torch.ones(2, 1),
    )
    loss, td_abs = trainer.compute_td_loss(org, batch)
    assert torch.isfinite(loss)
    assert torch.isfinite(td_abs).all()
