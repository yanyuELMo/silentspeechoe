"""LSTM encoders for raw sensor-window classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class SensorBiLSTM(nn.Module):
    """Bidirectional LSTM encoder for variable-length sensor sequences.

    The model accepts sensor windows in ``[B, C, T]`` format, transposes them
    to ``[B, T, C]`` for PyTorch's LSTM, and pools the sequence output with
    masked mean and max pooling.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 36,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1

        lstm_dropout = float(dropout) if self.num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=self.in_channels,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=self.bidirectional,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    @property
    def embedding_dim(self) -> int:
        """Dimension of the pooled pre-classifier representation."""
        return self.hidden_size * self.num_directions * 2

    def extract_features(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return pooled BiLSTM features.

        Args:
            x: Input tensor with shape ``[B, C, T]``.
            lengths: Optional valid lengths with shape ``[B]``.

        Returns:
            Feature tensor with shape ``[B, embedding_dim]``.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected input shape [B, C, T], got {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        sequence = x.transpose(1, 2).contiguous()
        max_t = sequence.shape[1]
        if lengths is None:
            lengths = torch.full(
                (sequence.shape[0],),
                max_t,
                dtype=torch.long,
                device=sequence.device,
            )
        lengths = lengths.to(device=sequence.device, dtype=torch.long).clamp(
            min=1,
            max=max_t,
        )

        packed = nn.utils.rnn.pack_padded_sequence(
            sequence,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=max_t,
        )

        mask = torch.arange(max_t, device=x.device)[None, :] < lengths[:, None]
        mask_float = mask.unsqueeze(-1).float()
        mean_pooled = (output * mask_float).sum(dim=1) / lengths[:, None].float()
        max_pooled = (
            output.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        )
        return torch.cat([mean_pooled, max_pooled], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return class logits."""
        features = self.extract_features(x, lengths=lengths)
        return self.classifier(self.dropout(features))


class IMUBinauralBiLSTM(SensorBiLSTM):
    """BiLSTM entry point for 18-channel binaural IMU windows."""

    def __init__(
        self,
        in_channels: int = 18,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 36,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            bidirectional=bidirectional,
        )
