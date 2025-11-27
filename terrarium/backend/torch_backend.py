"""Minimal torch backend shim for tensors and device handling."""

from __future__ import annotations

from typing import Any, Iterable

import torch


def get_best_device() -> str:
    """Return best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TorchBackend:
    """Lightweight wrapper to ease backend swapping later."""

    def __init__(self, device: str = "auto"):
        if device == "auto":
            device = get_best_device()
        self.device = torch.device(device)

    def tensor(self, x: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
        return torch.tensor(x, dtype=dtype).to(self.device)

    def zeros(self, shape: Iterable[int], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def randn(self, shape: Iterable[int], dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(shape, dtype=dtype, device=self.device)

    @property
    def float_dtype(self) -> torch.dtype:
        return torch.float32
