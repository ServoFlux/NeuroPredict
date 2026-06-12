"""3D CNN architecture for white matter disease classification."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Conv3d -> GroupNorm -> ReLU -> MaxPool.

    GroupNorm (instead of BatchNorm) keeps train/eval behaviour consistent with
    the small batch sizes typical of 3D MRI training on limited hardware.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        num_groups = min(8, out_channels)
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WMDClassifier3D(nn.Module):
    """A compact 3D CNN for binary/multi-class volume classification.

    Designed to be trainable on CPU for demos while remaining a sensible
    starting point for real GPU training on FLAIR MRI volumes.
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            ConvBlock(in_channels, 8),
            ConvBlock(8, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
        )
        # Global MAX pooling makes the head independent of the input size and,
        # crucially, is sensitive to small bright focal lesions (white-matter
        # hyperintensities) that average pooling would wash out.
        self.pool = nn.AdaptiveMaxPool3d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_model(num_classes: int = 2) -> WMDClassifier3D:
    return WMDClassifier3D(num_classes=num_classes)
