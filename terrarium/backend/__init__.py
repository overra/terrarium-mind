"""Backend abstraction layer (Torch for now, MLX later)."""

from .torch_backend import TorchBackend, get_best_device

__all__ = ["TorchBackend", "get_best_device"]
