"""Tests for feature-token Transformer models."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from silentspeechoe.models.build import build_model
from silentspeechoe.models.feature_token_transformer import (
    FeatureTokenTransformer,
    IMUBinauralFeatureTokenTransformer,
)


def test_binaural_feature_token_transformer_forward_shape() -> None:
    model = IMUBinauralFeatureTokenTransformer()
    x = torch.randn(4, 1296)

    out = model(x)

    assert out.shape == (4, 36)
    assert torch.isfinite(out).all()


def test_feature_token_transformer_extract_features_shape() -> None:
    model = FeatureTokenTransformer(
        in_features=12,
        num_tokens=3,
        token_dim=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        embedding_dim=6,
        num_classes=3,
    )
    x = torch.randn(5, 12)

    features = model.extract_features(x)

    assert features.shape == (5, 6)
    assert model.embedding_dim == 6


def test_feature_token_transformer_rejects_bad_layout() -> None:
    with pytest.raises(ValueError, match="num_tokens \\* token_dim"):
        FeatureTokenTransformer(in_features=10, num_tokens=3, token_dim=4)


def test_feature_token_transformer_rejects_wrong_feature_dim() -> None:
    model = FeatureTokenTransformer(
        in_features=12,
        num_tokens=3,
        token_dim=4,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
    )

    with pytest.raises(ValueError, match="Expected 12 input features"):
        model(torch.randn(2, 11))


def test_factory_builds_binaural_feature_token_transformer() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "imu_feature_token_transformer_binaural",
                "in_features": 1296,
                "num_tokens": 27,
                "token_dim": 48,
                "hidden_dim": 128,
                "num_layers": 2,
                "num_heads": 4,
                "embedding_dim": 256,
                "num_classes": 36,
            }
        }
    )

    model = build_model(cfg)

    assert isinstance(model, IMUBinauralFeatureTokenTransformer)
    assert model.embedding_dim == 256
