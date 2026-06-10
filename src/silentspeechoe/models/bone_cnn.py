"""Bone‑acceleration CNN models for binaural silent speech recognition."""

from __future__ import annotations

import torch
import torch.nn as nn


class BoneBinauralCNN(nn.Module):
    """Simple 1‑D CNN for binaural bone‑acceleration classification.

    Expected input shape: ``[B, 2, T]`` where channel 0 is the left‑ear
    magnitude and channel 1 is the right‑ear magnitude.

    Architecture::

        Conv1d → ReLU → Conv1d → ReLU → GlobalAvgPool → Linear → C
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 36,
        conv1_channels: int = 64,
        conv2_channels: int = 128,
        kernel_size_1: int = 7,
        kernel_size_2: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            conv1_channels,
            kernel_size=kernel_size_1,
            padding=kernel_size_1 // 2,
        )
        self.conv2 = nn.Conv1d(
            conv1_channels,
            conv2_channels,
            kernel_size=kernel_size_2,
            padding=kernel_size_2 // 2,
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(conv2_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: ``[B, C, T]`` float tensor.

        Returns:
            ``[B, num_classes]`` logits.
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        # Global average pooling over the time dimension
        x = x.mean(dim=-1)  # [B, conv2_channels]
        return self.classifier(x)
