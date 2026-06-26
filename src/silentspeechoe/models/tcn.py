"""Temporal CNN models for raw sensor-window classification.

The core model is sensor-agnostic: it accepts ``[B, C, T]`` windows and uses a
small residual temporal convolution stack followed by masked mean + max pooling.

Three explicit entry points are provided for the current processed windows:

* :class:`BoneAccTemporalCNN` for single-ear ``bone_acc`` windows, ``C=3``.
* :class:`IMUTemporalCNN` for single-ear IMU windows, ``C=9``.

The older :class:`BoneRawTCN` name remains available for binaural 6-channel
bone-acc experiments.
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


class SensorTemporalCNN(nn.Module):
    """Residual temporal CNN for variable-length raw sensor windows.

    Input shape: ``[B, C, T]``.

    Architecture::

        1x1 Conv projection: C -> hidden_channels
        Residual TCN block (dilation=1)
        Residual TCN block (...)
        Masked mean pooling + Masked max pooling -> concat
        Dropout
        Linear -> num_classes
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

        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)

        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.input_bn = nn.BatchNorm1d(hidden_channels)
        self.input_act = nn.ReLU(inplace=True)

        blocks: list[_ResidualTCNBlock] = []
        for dilation in dilations:
            blocks.append(
                _ResidualTCNBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.tcn_blocks = nn.Sequential(*blocks)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_channels * 2, num_classes)

    @property
    def embedding_dim(self) -> int:
        """Dimension of the pooled pre-classifier representation."""
        return self.hidden_channels * 2

    def extract_features(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return pooled pre-classifier features.

        Args:
            x: ``[B, C, T]`` float tensor.
            lengths: Optional ``[B]`` long tensor of valid time steps.

        Returns:
            ``[B, 2 * hidden_channels]`` float tensor.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input shape [B, C, T], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        max_t = x.shape[-1]
        device = x.device

        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.input_act(x)  # [B, hidden, T]

        x = self.tcn_blocks(x)  # [B, hidden, T]

        if lengths is None:
            mean_pooled = x.mean(dim=-1)
            max_pooled = x.max(dim=-1).values
            return torch.cat([mean_pooled, max_pooled], dim=-1)

        mask = torch.arange(max_t, device=device)[None, :] < lengths[:, None]  # [B, T]
        mask_float = mask.unsqueeze(1).float()  # [B, 1, T]

        sum_pooled = (x * mask_float).sum(dim=-1)  # [B, hidden]
        mean_pooled = sum_pooled / lengths[:, None].clamp(min=1).float()  # [B, hidden]

        x_masked = x.masked_fill(~mask_float.bool(), float("-inf"))
        max_pooled = x_masked.max(dim=-1).values  # [B, hidden]
        return torch.cat([mean_pooled, max_pooled], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return class logits for raw sensor windows."""
        features = self.extract_features(x, lengths=lengths)
        features = self.dropout(features)
        return self.classifier(features)


class BoneAccTemporalCNN(SensorTemporalCNN):
    """Temporal CNN entry point for processed single-ear bone_acc windows.

    Expected input shape: ``[B, 3, T]`` from
    ``data/processed/bone_acc/bone_acc_windows/bone_acc_1000*``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_channels: int = 64,
        num_classes: int = 36,
        kernel_size: int = 9,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16),
        dropout: float = 0.2,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )


class IMUTemporalCNN(SensorTemporalCNN):
    """Temporal CNN entry point for processed single-ear IMU windows.

    Expected input shape: ``[B, 9, T]`` from
    ``data/processed/imu/imu_windows/imu_189*``.
    """

    def __init__(
        self,
        in_channels: int = 9,
        hidden_channels: int = 96,
        num_classes: int = 36,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.3,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )


class IMUBinauralLRDiffTemporalCNN(SensorTemporalCNN):
    """Temporal CNN for 27-channel binaural IMU windows.

    Expected input layout is ``left 9 + right 9 + (left - right) 9``.
    """

    def __init__(
        self,
        in_channels: int = 27,
        hidden_channels: int = 96,
        num_classes: int = 36,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.3,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )


class BoneRawTCN(SensorTemporalCNN):
    """Backward-compatible TCN for raw binaural bone-acc windows.

    Expected input shape: ``[B, 6, T]`` with left xyz + right xyz channels.
    For the new single-ear processed windows, prefer
    :class:`BoneAccTemporalCNN`.
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
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )
