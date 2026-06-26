"""Standard 1-D CNN models for sensor-window classification.

The core model handles ``[B, C, T]`` variable-length sensor windows via a
stack of strided Conv1d blocks followed by masked mean + max pooling.  It
is intentionally simpler than the residual TCN — no dilated convolutions,
no residual connections — so it serves as a lighter, easier-to-tune baseline.

Three entry points are provided:

* :class:`IMUBinauralCNN` for binaural 18‑channel IMU raw windows.
* :class:`IMUBinauralFeatureCNN` for binaural temporal‑envelope features
  reshaped to ``[B, C_feat, T_feat]``.
* :class:`SensorCNN` — the generic base class usable with any sensor or
  reshaped feature input.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    """A single Conv1d → BatchNorm1d → ReLU → Dropout block.

    When *stride* > 1 the block also temporally downsamples.  Padding is
    chosen to preserve approximately ``T / stride`` output length.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.relu(self.bn(self.conv(x))))


class SensorCNN(nn.Module):
    """Standard 1‑D CNN for variable-length sensor windows.

    Input shape: ``[B, C, T]``.

    Architecture::

        1×1 Conv projection: C → hidden_channels
        ConvBlock (hidden → 2×hidden, stride=1)
        ConvBlock (2×hidden → 4×hidden, stride=1)
        ConvBlock (4×hidden → 4×hidden, stride=1)
        ConvBlock (4×hidden → 4×hidden, stride=1)
        Masked mean pooling + masked max pooling → concat
        Dropout
        Linear → num_classes
    """

    def __init__(
        self,
        in_channels: int = 18,
        hidden_channels: int = 64,
        num_classes: int = 36,
        kernel_size: int = 7,
        num_conv_blocks: int = 4,
        dropout: float = 0.2,
        *,
        return_features_for_training: bool = False,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.hidden_channels = int(hidden_channels)
        self.return_features_for_training = bool(return_features_for_training)

        # Channel progression per block (e.g. 64 → 128 → 256 → 256).
        self._block_channels: list[int] = []
        for i in range(num_conv_blocks):
            if i == 0:
                self._block_channels.append(hidden_channels * 2)
            elif i == 1:
                self._block_channels.append(hidden_channels * 4)
            else:
                self._block_channels.append(hidden_channels * 4)

        # 1×1 input projection.
        self.input_proj = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.input_bn = nn.BatchNorm1d(hidden_channels)
        self.input_act = nn.ReLU(inplace=True)

        # Conv blocks.
        blocks: list[_ConvBlock] = []
        prev_ch = hidden_channels
        for out_ch in self._block_channels:
            blocks.append(
                _ConvBlock(
                    in_channels=prev_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dropout=dropout,
                )
            )
            prev_ch = out_ch
        self.conv_blocks = nn.Sequential(*blocks)

        # Final channel count determines classifier input.
        final_ch = self._block_channels[-1] if self._block_channels else hidden_channels

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(final_ch * 2, num_classes)
        self._embedding_dim = final_ch * 2

    @property
    def embedding_dim(self) -> int:
        """Dimension of the pooled pre-classifier representation."""
        return self._embedding_dim

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
            ``[B, 2 * final_channels]`` float tensor.
        """
        if x.dim() == 2:
            x = self._reshape_features(x)
            lengths = torch.full(
                (x.shape[0],),
                x.shape[-1],
                dtype=torch.long,
                device=x.device,
            )
        self._validate_input(x)

        max_t = x.shape[-1]
        device = x.device

        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.input_act(x)  # [B, hidden, T]

        x = self.conv_blocks(x)  # [B, final_ch, T]

        if lengths is None:
            mean_pooled = x.mean(dim=-1)
            max_pooled = x.max(dim=-1).values
            return torch.cat([mean_pooled, max_pooled], dim=-1)

        # Scale lengths proportionally for stride (all strides are 1 here,
        # so no scaling needed — T is preserved).  The mask is computed on
        # the current temporal dimension.
        cur_t = x.shape[-1]
        scale = cur_t / max(max_t, 1)
        scaled_lengths = (lengths.float() * scale).long().clamp(min=1)

        mask = (
            torch.arange(cur_t, device=device)[None, :] < scaled_lengths[:, None]
        )  # [B, cur_t]
        mask_float = mask.unsqueeze(1).float()  # [B, 1, cur_t]

        # Masked mean.
        sum_pooled = (x * mask_float).sum(dim=-1)
        mean_pooled = sum_pooled / scaled_lengths[:, None].clamp(min=1).float()

        # Masked max.
        x_masked = x.masked_fill(~mask_float.bool(), float("-inf"))
        max_pooled = x_masked.max(dim=-1).values

        return torch.cat([mean_pooled, max_pooled], dim=-1)

    @property
    def requires_labels_for_training(self) -> bool:
        """Signal to the trainer that this model wants labels during
        ``forward()`` so it can return ``(features, logits)`` for
        contrastive losses."""
        return self.return_features_for_training

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return class logits, or ``(features, logits)`` during training.

        When ``return_features_for_training`` is ``True`` and *labels* are
        provided the method returns a ``(features, logits)`` tuple suitable
        for :class:`~silentspeechoe.training.losses.SupConCrossEntropyLoss`.

        Automatically reshapes ``[B, D]`` feature vectors to
        ``[B, C, D//C]`` when there is no temporal dimension.  When
        the input is reshaped, *lengths* are updated to reflect the
        new temporal dimension (equal for all samples in the batch).
        """
        if x.dim() == 2:
            x = self._reshape_features(x)
            # After reshape [B, D] → [B, C, T], all samples share the same T.
            lengths = torch.full(
                (x.shape[0],), x.shape[-1], dtype=torch.long, device=x.device
            )
        features = self.extract_features(x, lengths=lengths)
        logits = self.classifier(self.dropout(features))

        if self.return_features_for_training and labels is not None:
            return features, logits
        return logits

    def _reshape_features(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape ``[B, D]`` → ``[B, in_channels, D // in_channels]``."""
        if x.dim() != 2:
            return x
        d = x.shape[1]
        if d % self.in_channels != 0:
            raise ValueError(
                f"Feature dimension {d} is not divisible by "
                f"in_channels={self.in_channels}.  Set in_channels to a "
                f"divisor of {d}, or reshape manually before calling forward()."
            )
        return x.view(x.shape[0], self.in_channels, d // self.in_channels)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(f"Expected input shape [B, C, T], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )


class IMUBinauralCNN(SensorCNN):
    """CNN entry point for binaural 18‑channel IMU raw windows.

    Expected input: ``[B, 18, T]`` produced by
    :class:`~silentspeechoe.scripts.train._BinauralWindowDataset`.
    """

    def __init__(
        self,
        in_channels: int = 18,
        hidden_channels: int = 64,
        num_classes: int = 36,
        kernel_size: int = 7,
        num_conv_blocks: int = 4,
        dropout: float = 0.2,
        *,
        return_features_for_training: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            kernel_size=kernel_size,
            num_conv_blocks=num_conv_blocks,
            dropout=dropout,
            return_features_for_training=return_features_for_training,
        )


class IMUBinauralFeatureCNN(SensorCNN):
    """CNN entry point for reshaped binaural IMU feature vectors.

    Designed for fixed‑length temporal‑envelope or MFCC feature vectors
    that are reshaped from ``[B, D]`` into ``[B, C_feat, T_feat]``.

    Defaults target the 1296‑dim binaural temporal‑envelope features
    reshaped to ``[B, 36, 36]`` (36 feature channels × 36 time‑like steps).
    """

    def __init__(
        self,
        in_channels: int = 36,
        hidden_channels: int = 64,
        num_classes: int = 36,
        kernel_size: int = 5,
        num_conv_blocks: int = 3,
        dropout: float = 0.3,
        *,
        return_features_for_training: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_classes=num_classes,
            kernel_size=kernel_size,
            num_conv_blocks=num_conv_blocks,
            dropout=dropout,
            return_features_for_training=return_features_for_training,
        )
