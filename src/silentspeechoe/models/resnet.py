"""1-D ResNet models for raw sensor-window classification.

The model uses strided residual blocks to progressively downsample the
temporal axis while expanding channel capacity.  Variable‑length inputs
are handled via masked mean + max pooling (same strategy as the TCN).

Architecture::

    [B, C, T]
    → 1×1 Conv proj  → BN → ReLU
    → ResBlock (base → base×2,  stride=2)
    → ResBlock (base×2 → base×2)
    → ResBlock (base×2 → base×4,  stride=2)
    → ResBlock (base×4 → base×4)
    → ResBlock (base×4 → base×8,  stride=2)
    → ResBlock (base×8 → base×8)
    → Masked mean + max pooling → concat
    → Dropout → Linear → num_classes

Entry points:

* :class:`IMUBinauralResNet` for binaural 18‑channel IMU raw windows.
* :class:`SensorResNet` — generic base class.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _ResBlock(nn.Module):
    """A single 1‑D residual block.

    Structure::

        Conv1d → BatchNorm1d → ReLU → Conv1d → BatchNorm1d
        └── shortcut (1×1 Conv if stride≠1 or channel change) ──┘
        → ReLU
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

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut: nn.Module | None = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )

        self.relu_out = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu_out(out)
        return out


class SensorResNet(nn.Module):
    """1‑D ResNet for variable-length raw sensor windows.

    Input shape: ``[B, C, T]``.
    """

    def __init__(
        self,
        in_channels: int = 18,
        base_channels: int = 64,
        num_classes: int = 36,
        kernel_size: int = 7,
        num_stages: int = 3,
        blocks_per_stage: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.num_stages = int(num_stages)

        # Input projection.
        self.input_proj = nn.Conv1d(
            in_channels,
            base_channels,
            kernel_size=1,
            bias=False,
        )
        self.input_bn = nn.BatchNorm1d(base_channels)
        self.input_act = nn.ReLU(inplace=True)

        # Build stages: each stage starts with stride=2 (except first stage's
        # first block), then stride=1 blocks.
        stages: list[nn.Sequential] = []
        prev_ch = base_channels
        self._stage_channels: list[int] = [base_channels]

        for stage in range(num_stages):
            out_ch = base_channels * (2 ** (stage + 1))
            self._stage_channels.append(out_ch)

            blocks: list[_ResBlock] = []
            for block_idx in range(blocks_per_stage):
                stride = 2 if (block_idx == 0 and stage > 0) else 1
                blocks.append(
                    _ResBlock(
                        in_channels=prev_ch,
                        out_channels=out_ch,
                        kernel_size=kernel_size,
                        stride=stride,
                        dropout=dropout,
                    )
                )
                prev_ch = out_ch
            stages.append(nn.Sequential(*blocks))

        self.stages = nn.Sequential(*stages)
        final_ch = self._stage_channels[-1]

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(final_ch * 2, num_classes)

    @property
    def final_channels(self) -> int:
        return self._stage_channels[-1]

    def extract_features(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return pooled pre-classifier features ``[B, 2 * final_channels]``."""
        self._validate_input(x)

        max_t = x.shape[-1]
        device = x.device

        x = self.input_proj(x)
        x = self.input_bn(x)
        x = self.input_act(x)  # [B, base, T]

        x = self.stages(x)  # [B, final_ch, T']

        if lengths is None:
            mean_pooled = x.mean(dim=-1)
            max_pooled = x.max(dim=-1).values
            return torch.cat([mean_pooled, max_pooled], dim=-1)

        # Scale lengths to match the temporally-downsampled output.
        cur_t = x.shape[-1]
        scale = cur_t / max(max_t, 1)
        scaled_lengths = (lengths.float() * scale).long().clamp(min=1)

        mask = torch.arange(cur_t, device=device)[None, :] < scaled_lengths[:, None]
        mask_float = mask.unsqueeze(1).float()

        sum_pooled = (x * mask_float).sum(dim=-1)
        mean_pooled = sum_pooled / scaled_lengths[:, None].clamp(min=1).float()

        x_masked = x.masked_fill(~mask_float.bool(), float("-inf"))
        max_pooled = x_masked.max(dim=-1).values

        return torch.cat([mean_pooled, max_pooled], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return class logits."""
        features = self.extract_features(x, lengths=lengths)
        features = self.dropout(features)
        return self.classifier(features)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(f"Expected input shape [B, C, T], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )


class IMUBinauralResNet(SensorResNet):
    """1‑D ResNet entry point for binaural 18‑channel IMU raw windows.

    Expected input: ``[B, 18, T]``.
    """

    def __init__(
        self,
        in_channels: int = 18,
        base_channels: int = 64,
        num_classes: int = 36,
        kernel_size: int = 7,
        num_stages: int = 3,
        blocks_per_stage: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__(
            in_channels=in_channels,
            base_channels=base_channels,
            num_classes=num_classes,
            kernel_size=kernel_size,
            num_stages=num_stages,
            blocks_per_stage=blocks_per_stage,
            dropout=dropout,
        )
