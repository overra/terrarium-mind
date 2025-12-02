"""Proto vision encoder turning retina grids into hemisphere features."""

from __future__ import annotations

import torch
from torch import nn


class VisionEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 32, out_dim: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Linear(hidden_dim * 4 * 4 // 2, out_dim)

    def forward(self, retina: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        retina: [B, C, H, W]
        returns: (vision_left, vision_right) each [B, out_dim]
        """
        feat = self.conv(retina)
        pooled = self.pool(feat)  # [B, H', W']
        B, C, H, W = pooled.shape
        left = pooled[:, :, :, : W // 2].reshape(B, -1)
        right = pooled[:, :, :, W // 2 :].reshape(B, -1)
        left_vec = self.proj(left)
        right_vec = self.proj(right)
        return left_vec, right_vec


class CameraVisionEncoder(nn.Module):
    """Simple CNN over a small camera image, split into left/right halves."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.proj = nn.Linear(hidden_dim * 4 * 4 // 2, out_dim)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        image: [B, C, H, W]
        returns: (vision_left, vision_right) each [B, out_dim]
        """
        feat = self.encoder(image)
        B, C, H, W = feat.shape
        left = feat[:, :, :, : W // 2].reshape(B, -1)
        right = feat[:, :, :, W // 2 :].reshape(B, -1)
        left_vec = self.proj(left)
        right_vec = self.proj(right)
        return left_vec, right_vec
