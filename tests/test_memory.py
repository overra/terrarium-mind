from terrarium.organism.memory import SalientMemory
import numpy as np


def test_salient_memory_threshold_and_capacity() -> None:
    mem = SalientMemory(capacity=2, salience_threshold=0.5)
    mem.consider(np.zeros(2), np.zeros(2), reward=0.1, confusion=0.1, task_id="t", timestamp=0, info={})
    assert len(mem.entries) == 0  # below threshold
    mem.consider(np.zeros(2), np.zeros(2), reward=1.0, confusion=0.0, task_id="t", timestamp=1, info={})
    mem.consider(np.zeros(2), np.zeros(2), reward=0.0, confusion=0.6, task_id="t", timestamp=2, info={})
    assert len(mem.entries) == 2  # capacity respected
    assert mem.entries[0].timestamp == 1
    mem.consider(np.zeros(2), np.zeros(2), reward=1.0, confusion=0.6, task_id="t", timestamp=3, info={})
    assert len(mem.entries) == 2
    assert mem.entries[0].timestamp == 2  # oldest dropped
