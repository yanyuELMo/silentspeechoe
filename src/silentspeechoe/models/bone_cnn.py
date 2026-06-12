"""Bone‑acceleration CNN models for binaural silent speech recognition."""

from __future__ import annotations

import torch
import torch.nn as nn


class BoneBinauralCNN(nn.Module):
    """1‑D CNN for binaural bone‑acceleration classification.

    Input shape: ``[B, C, T]`` where *C* is the number of channels
    (2 for raw magnitude, 60 for engineered features) and *T* is the
    variable‑length time / frame dimension.

    Architecture::

        Conv1d → ReLU → Conv1d → ReLU → MaskedAvgPool → Dropout → Linear → C
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

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional length‑aware pooling.

        Args:
            x: ``[B, C, T]`` float tensor.
            lengths: ``[B]`` long tensor of valid time steps.
                If ``None``, a simple global mean is used (backward
                compatible with equal‑length or pre‑padded inputs).

        Returns:
            ``[B, num_classes]`` logits.
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.dropout(x)

        if lengths is not None:
            # Masked mean pooling — only average over valid steps.
            max_t = x.shape[-1]
            mask = (
                torch.arange(max_t, device=x.device)[None, :] < lengths[:, None]
            )  # [B, T]
            mask = mask.unsqueeze(1)  # [B, 1, T]
            x = (x * mask).sum(dim=-1) / lengths[:, None].clamp(min=1)  # [B, C]
        else:
            x = x.mean(dim=-1)  # [B, conv2_channels]

        return self.classifier(x)
