"""Utilities for logging retina grids to wandb."""

from __future__ import annotations

import numpy as np


def retina_to_image(retina: np.ndarray) -> np.ndarray:
    """Convert a retina grid (H,W,C or C,H,W) into an RGB uint8 image.

    Channel mapping (simple):
    0: occupancy -> red
    1: objects -> green
    2: peers -> blue
    3: mirrors -> cyan
    4: screens -> magenta
    5: intensity -> grayscale
    6: motion -> orange tint
    """
    if retina.ndim != 3:
        raise ValueError("retina must be 3D")
    if retina.shape[0] <= 6:  # assume C,H,W
        retina = np.transpose(retina, (1, 2, 0))
    H, W, C = retina.shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    channels = [retina[..., i] if i < C else np.zeros((H, W)) for i in range(7)]
    img[..., 0] += channels[0]  # occupancy -> R
    img[..., 1] += channels[1]  # objects -> G
    img[..., 2] += channels[2]  # peers -> B
    img[..., 1] += channels[3] * 0.5  # mirrors -> G tint
    img[..., 0] += channels[4] * 0.5  # screens -> R tint
    img += np.stack([channels[5]] * 3, axis=-1) * 0.2  # intensity as gray tint
    img[..., 0] += channels[6] * 0.3  # motion tint
    img[..., 1] += channels[6] * 0.1
    img = np.clip(img, 0.0, 1.0)
    return (img * 255).astype(np.uint8)
