"""Lightweight multi-camera vision encoder for edge VLA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minipi05.vla.mobilenet_blocks import DepthwiseSeparableConv, make_divisible


class SmallMultiCamEncoder(nn.Module):
    """
    Shared-weight CNN over multiple camera views.

    Each view is encoded independently, then mean-pooled across cameras.
    Designed for fixed S100 camera rigs (e.g. front + wrist).
    """

    def __init__(
        self,
        *,
        num_cameras: int,
        image_size: int,
        out_dim: int,
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.image_size = image_size
        c1 = make_divisible(32 * width_mult)
        c2 = make_divisible(64 * width_mult)
        c3 = make_divisible(128 * width_mult)
        c4 = make_divisible(256 * width_mult)

        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU6(inplace=True),
        )
        self.blocks = nn.Sequential(
            DepthwiseSeparableConv(c1, c2, 2),
            DepthwiseSeparableConv(c2, c3, 2),
            DepthwiseSeparableConv(c3, c4, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(c4, out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C_cam, 3, H, W) or (B, 3, H, W) for a single camera.

        Returns:
            (B, out_dim) visual embedding.
        """
        if images.dim() == 4:
            images = images.unsqueeze(1)
        if images.dim() != 5:
            raise ValueError(f"expected 4D or 5D images, got shape {tuple(images.shape)}")

        b, n_cam, _, h, w = images.shape
        if h != self.image_size or w != self.image_size:
            images = F.interpolate(
                images.reshape(b * n_cam, 3, h, w),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(b, n_cam, 3, self.image_size, self.image_size)

        x = images.reshape(b * n_cam, 3, self.image_size, self.image_size)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        x = self.proj(x)
        x = x.reshape(b, n_cam, -1).mean(dim=1)
        return x
