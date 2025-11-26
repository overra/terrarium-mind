"""Runtime wrapper that wires world, organism, and trainer."""

from __future__ import annotations

from terrarium.training import RLTrainer
from terrarium.world import World


class Runtime:
    """Thin runtime orchestrator."""

    def __init__(self, world: World, trainer: RLTrainer):
        self.world = world
        self.trainer = trainer

    def run(self) -> None:
        """Delegate to trainer loop."""
        self.trainer.run()
