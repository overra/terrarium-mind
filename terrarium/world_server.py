"""WorldServer orchestrates the world and agent clients."""

from __future__ import annotations

from typing import Dict, Optional

from terrarium.world import World
from terrarium.agents import AgentClient


class WorldServer:
    def __init__(self, world: World, learn: bool = True):
        self.world = world
        self.learn = learn
        self.clients: Dict[str, AgentClient] = {}
        self.last_obs: Dict[str, dict] = {}

    def add_agent(self, agent_id: str, client: AgentClient) -> None:
        self.clients[agent_id] = client
        # Initialize with a fresh obs
        init_obs = self.world.reset()
        self.last_obs[agent_id] = init_obs
        client.init_episode(init_obs)

    def remove_agent(self, agent_id: str) -> None:
        self.clients.pop(agent_id, None)
        self.last_obs.pop(agent_id, None)

    def step(self, t: int) -> None:
        # Gather actions
        actions = {}
        for agent_id, client in self.clients.items():
            obs = self.last_obs.get(agent_id, {})
            actions[agent_id] = client.act(obs, t)
        # Step world (single-agent for now: pick its action directly)
        action_value = list(actions.values())[0] if actions else "stay"
        obs, reward, done, info = self.world.step(action_value)
        # Dispatch results
        for agent_id, client in self.clients.items():
            client.observe(obs, reward, done, info)
            self.last_obs[agent_id] = obs
            client.on_world_step(t)
            if done:
                new_obs = self.world.reset()
                self.last_obs[agent_id] = new_obs
                client.init_episode(new_obs)

    def run(self, num_steps: Optional[int] = None) -> None:
        try:
            if num_steps is None:
                t = 0
                while True:
                    self.step(t)
                    t += 1
            else:
                for t in range(num_steps):
                    self.step(t)
        except KeyboardInterrupt:
            return

    def get_snapshot(self) -> dict:
        status = {aid: client.get_status() for aid, client in self.clients.items()}
        snap = self.world.get_snapshot(agent_status=status)
        snap["agents_status"] = [
            dict(meta, id=aid) for aid, meta in status.items()
        ]
        return snap
