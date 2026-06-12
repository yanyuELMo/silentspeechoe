"""Temporal Convolutional Network for raw binaural bone‑acceleration classification.

Input: ``[B, 6, T]`` where the 6 channels are
``(left_x, left_y, left_z, right_x, right_y, right_z)``.

No CTC — the model uses masked mean + max pooling across the time
dimension, followed by a linear classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ResidualTCNBlock(nn.Module):
    """A single residual TCN block.

    Structure::

        Conv1d -> BatchNorm1d -> ReLU -> Dropout -> Conv1d -> BatchNorm1d
        └── residual (1x1 conv if channel dims differ) ──┘
        -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()

        # Depthwise‑separable style: keep same temporal resolution.
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 1x1 projection when channel dimensions differ.
        self.proj: nn.Module | None = None
        if in_channels != out_channels:
            self.proj = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.proj is None else self.proj(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)
        return out


class BoneRawTCN(nn.Module):
    """Temporal Convolutional Network for raw binaural bone‑acc signals.

    Input shape: ``[B, 6, T]`` (6 raw acc channels).

    Architecture::

        1x1 Conv projection: 6 -> hidden_channels
        Residual TCN block (dilation=1)
        Residual TCN block (dilation=2)
        Residual TCN block (dilation=4)
        Residual TCN block (dilation=8)
        Residual TCN block (dilation=16)
        Masked mean pooling + Masked max pooling -> concat
        Dropout
        Linear -> num_classes

    All convolutions are causal‑ish (same padding, no future leakage).
    """

    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels: int = 64,
        num_classes: int = 36,
        kernel_size: int = 7,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.2,
    ):
        super().__init__()

        # Input projection.
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.input_bn = nn.BatchNorm1d(hidden_channels)
        self.input_act = nn.ReLU(inplace=True)

        # Stack of residual TCN blocks.
        blocks: list[_ResidualTCNBlock] = []
        for d in dilations:
            blocks.append(
                _ResidualTCNBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout,
                )
            )
        self.tcn_blocks = nn.Sequential(*blocks)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels * 2, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with length‑aware masked pooling.

        Args:
            x: ``[B, C, T]`` float tensor (C = 6).
            lengths: ``[B]`` long tensor of valid time steps.

        Returns:
            ``[B, num_classes]`` logits.
        """
        B, C, max_t = x.shape
        device = x.device

        # Input projection.
        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.input_act(x)  # [B, hidden, T]

        # TCN stack.
        x = self.tcn_blocks(x)  # [B, hidden, T]

        # Build mask: True for valid time steps.
        mask = torch.arange(max_t, device=device)[None, :] < lengths[:, None]  # [B, T]
        mask_float = mask.unsqueeze(1).float()  # [B, 1, T]

        # Masked mean pooling.
        sum_pooled = (x * mask_float).sum(dim=-1)  # [B, hidden]
        mean_pooled = sum_pooled / lengths[:, None].clamp(min=1).float()  # [B, hidden]

        # Masked max pooling (set padded positions to -inf).
        x_masked = x.masked_fill(~mask_float.bool(), float("-inf"))
        max_pooled = x_masked.max(dim=-1).values  # [B, hidden]

        # Concatenate and classify.
        pooled = torch.cat([mean_pooled, max_pooled], dim=-1)  # [B, 2*hidden]
        pooled = self.dropout(pooled)

        return self.classifier(pooled)
