"""1‑D Temporal CNN for IMU sensor sequence classification.

Input: ``[B, C, T]`` with C=9 channels (acc.xyz + gyro.xyz + mag.xyz)
at 200 Hz.  Supports variable‑length sequences via masked average pooling.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class IMUCNN(nn.Module):
    """1‑D CNN for single‑side IMU utterance classification.

    Input shape: ``[B, C, T]`` where *C* is the number of IMU channels
    (9 for one side, 18 for left+right concatenated) and *T* is the
    variable‑length time dimension after 200 Hz resampling.

    Architecture::

        Conv1d → BN → ReLU → Dropout
        → Conv1d → BN → ReLU → Dropout
        → Conv1d → BN → ReLU
        → MaskedAvgPool (or global mean)
        → Linear → num_classes
    """

    def __init__(
        self,
        in_channels: int = 9,
        num_classes: int = 36,
        conv1_channels: int = 64,
        conv2_channels: int = 128,
        conv3_channels: int = 128,
        kernel_size_1: int = 7,
        kernel_size_2: int = 5,
        kernel_size_3: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Block 1
        self.conv1 = nn.Conv1d(
            in_channels,
            conv1_channels,
            kernel_size=kernel_size_1,
            padding=kernel_size_1 // 2,
        )
        self.bn1 = nn.BatchNorm1d(conv1_channels)

        # Block 2
        self.conv2 = nn.Conv1d(
            conv1_channels,
            conv2_channels,
            kernel_size=kernel_size_2,
            padding=kernel_size_2 // 2,
        )
        self.bn2 = nn.BatchNorm1d(conv2_channels)

        # Block 3
        self.conv3 = nn.Conv1d(
            conv2_channels,
            conv3_channels,
            kernel_size=kernel_size_3,
            padding=kernel_size_3 // 2,
        )
        self.bn3 = nn.BatchNorm1d(conv3_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(conv3_channels, num_classes)

    def _pool(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply masked or simple global average pooling.

        Args:
            x: ``[B, C, T]`` post-conv features.
            lengths: ``[B]`` long tensor of valid steps, or ``None``.

        Returns:
            ``[B, C]`` pooled features.
        """
        if lengths is not None:
            max_t = x.shape[-1]
            mask = (
                torch.arange(max_t, device=x.device)[None, :] < lengths[:, None]
            )  # [B, T]
            mask = mask.unsqueeze(1)  # [B, 1, T]
            return (x * mask).sum(dim=-1) / lengths[:, None].clamp(min=1)  # [B, C]
        return x.mean(dim=-1)  # [B, C]

    def extract_features(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the conv stack and pooling, returning pre‑classifier features.

        Args:
            x: ``[B, C, T]`` float tensor.
            lengths: ``[B]`` long tensor of valid time steps, or ``None``.

        Returns:
            ``[B, embedding_dim]`` float tensor (128 for default config).
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return self._pool(x, lengths)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional length‑aware pooling.

        All convolutions use ``padding = kernel_size // 2`` and stride 1,
        so the time dimension is preserved through the conv stack.

        Args:
            x: ``[B, C, T]`` float tensor.
            lengths: ``[B]`` long tensor of valid time steps.
                If ``None``, a simple global mean is used (backward
                compatible with equal‑length or pre‑padded inputs).

        Returns:
            ``[B, num_classes]`` logits.
        """
        features = self.extract_features(x, lengths)
        return self.classifier(features)


class IMUDoubleCNN(nn.Module):
    """Three-layer 1-D CNN over fixed-length binaural IMU features.

    This model is intended for ordered handcrafted feature vectors such as
    binaural temporal-envelope features ``[left, right, left-right]``.  The
    input is a fixed vector ``[B, D]`` and is reshaped internally to
    ``[B, 1, D]`` before convolution.
    """

    def __init__(
        self,
        in_features: int = 1296,
        num_classes: int = 36,
        conv1_channels: int = 64,
        conv2_channels: int = 128,
        conv3_channels: int = 128,
        kernel_size_1: int = 7,
        kernel_size_2: int = 5,
        kernel_size_3: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_features = int(in_features)

        self.conv1 = nn.Conv1d(
            1,
            conv1_channels,
            kernel_size=kernel_size_1,
            padding=kernel_size_1 // 2,
        )
        self.bn1 = nn.BatchNorm1d(conv1_channels)
        self.conv2 = nn.Conv1d(
            conv1_channels,
            conv2_channels,
            kernel_size=kernel_size_2,
            padding=kernel_size_2 // 2,
        )
        self.bn2 = nn.BatchNorm1d(conv2_channels)
        self.conv3 = nn.Conv1d(
            conv2_channels,
            conv3_channels,
            kernel_size=kernel_size_3,
            padding=kernel_size_3 // 2,
        )
        self.bn3 = nn.BatchNorm1d(conv3_channels)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(conv3_channels, num_classes)

    def _check_input(self, x: torch.Tensor) -> None:
        """Validate fixed-vector input shape."""
        if x.dim() != 2:
            raise ValueError(f"Expected input shape [B, D], got {tuple(x.shape)}")
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected feature dimension {self.in_features}, got {x.shape[1]}"
            )

    def extract_features(
        self,
        x: torch.Tensor,
        lengths=None,
    ) -> torch.Tensor:
        """Return pre-classifier embeddings for fixed IMU features."""
        del lengths
        self._check_input(x)
        x = x.unsqueeze(1)  # [B, 1, D]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(self.dropout(x))

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(self.dropout(x))

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.global_pool(x).squeeze(-1)
        return x

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """Return class logits for fixed IMU feature vectors."""
        features = self.extract_features(x, lengths=lengths)
        return self.classifier(features)


# Backward-compatible alias for earlier local experiments.
IMUFeatureCNN = IMUDoubleCNN
