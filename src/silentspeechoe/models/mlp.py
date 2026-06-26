"""MLP models for fixed-length sensor feature vectors."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureMLP(nn.Module):
    """Small MLP classifier for pre-computed fixed-length feature vectors.

    The model is intended for handcrafted IMU features such as temporal-envelope
    vectors. Input shape is ``[B, D]``. A ``[B, C, T]`` tensor is also accepted
    and flattened to ``[B, C*T]`` for compatibility with generic training code.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: tuple[int, ...] = (512, 256),
        num_classes: int = 36,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if not hidden_features:
            raise ValueError("hidden_features must contain at least one layer")

        self.in_features = int(in_features)
        self.hidden_features = tuple(int(width) for width in hidden_features)
        self.num_classes = int(num_classes)

        layers: list[nn.Module] = []
        prev_features = self.in_features
        for width in self.hidden_features:
            if width <= 0:
                raise ValueError(f"hidden layer width must be positive, got {width}")
            layers.append(nn.Linear(prev_features, width))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_features = width

        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_features, self.num_classes)

    @property
    def embedding_dim(self) -> int:
        """Dimension of the pre-classifier representation."""
        return self.hidden_features[-1]

    def extract_features(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the pre-classifier embedding."""
        del lengths
        x = self._prepare_input(x)
        return self.encoder(x)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return class logits for fixed-length feature vectors."""
        features = self.extract_features(x, lengths=lengths)
        return self.classifier(features)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.flatten(start_dim=1)
        elif x.dim() != 2:
            raise ValueError(f"Expected input shape [B, D], got {tuple(x.shape)}")

        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected {self.in_features} input features, got {x.shape[1]}"
            )
        return x.float()


class IMUFeatureMLP(FeatureMLP):
    """MLP entry point for single-ear IMU temporal-envelope features."""

    def __init__(
        self,
        in_features: int = 432,
        hidden_features: tuple[int, ...] = (512, 256),
        num_classes: int = 36,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            num_classes=num_classes,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )


class IMUBinauralFeatureMLP(FeatureMLP):
    """MLP entry point for binaural IMU left/right/difference features."""

    def __init__(
        self,
        in_features: int = 1296,
        hidden_features: tuple[int, ...] = (1024, 512),
        num_classes: int = 36,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            num_classes=num_classes,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )
