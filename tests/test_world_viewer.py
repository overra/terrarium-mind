import numpy as np

from terrarium.vis.world_viewer import render_snapshot_topdown


def test_render_snapshot_topdown_shape() -> None:
    snap = {
        "world_size": 4.0,
        "agents": [{"id": "a", "pos": [1.0, 1.0], "orientation": 0.0}],
        "peers": [],
        "objects": [],
        "mirrors": [],
        "screens": [],
    }
    img = render_snapshot_topdown(snap, size=64)
    assert img.shape == (64, 64, 3)
    assert img.dtype == np.uint8
