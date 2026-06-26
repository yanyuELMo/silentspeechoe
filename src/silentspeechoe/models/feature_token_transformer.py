"""Transformer models for grouped fixed-length feature vectors."""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureTokenTransformer(nn.Module):
    """Small Transformer classifier over grouped feature tokens.

    The model treats a fixed feature vector as ``num_tokens`` contiguous feature
    groups. For binaural IMU temporal-envelope features, the default layout is
    ``27 x 48``: left/right/difference blocks, each with 9 IMU channels.
    """

    def __init__(
        self,
        in_features: int = 1296,
        num_tokens: int = 27,
        token_dim: int = 48,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        embedding_dim: int = 256,
        num_classes: int = 36,
        dropout: float = 0.3,
    ):
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}")
        if token_dim <= 0:
            raise ValueError(f"token_dim must be positive, got {token_dim}")
        if in_features != num_tokens * token_dim:
            raise ValueError(
                "in_features must equal num_tokens * token_dim, got "
                f"{in_features} != {num_tokens} * {token_dim}"
            )
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by num_heads, got "
                f"{hidden_dim} and {num_heads}"
            )
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")

        self.in_features = int(in_features)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self._embedding_dim = int(embedding_dim)

        self.token_projection = nn.Linear(self.token_dim, self.hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens + 1, self.hidden_dim)
        )
        self.input_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(num_heads),
            dim_feedforward=int(round(self.hidden_dim * mlp_ratio)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(num_layers),
        )
        self.embedding_projection = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self._embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(self._embedding_dim, self.num_classes)

        self._reset_parameters()

    @property
    def embedding_dim(self) -> int:
        """Dimension of the pre-classifier representation."""
        return self._embedding_dim

    def extract_features(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the CLS-based feature embedding."""
        del lengths
        x = self._prepare_input(x)
        batch_size = x.shape[0]
        tokens = x.view(batch_size, self.num_tokens, self.token_dim)
        tokens = self.token_projection(tokens)

        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.position_embedding
        tokens = self.input_dropout(tokens)

        encoded = self.encoder(tokens)
        return self.embedding_projection(encoded[:, 0])

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return class logits."""
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

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)


class IMUBinauralFeatureTokenTransformer(FeatureTokenTransformer):
    """Feature-token Transformer for binaural IMU left/right/difference features."""

    def __init__(
        self,
        in_features: int = 1296,
        num_tokens: int = 27,
        token_dim: int = 48,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        embedding_dim: int = 256,
        num_classes: int = 36,
        dropout: float = 0.3,
    ):
        super().__init__(
            in_features=in_features,
            num_tokens=num_tokens,
            token_dim=token_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=dropout,
        )
