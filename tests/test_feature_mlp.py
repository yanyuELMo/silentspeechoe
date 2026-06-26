"""Tests for fixed-length feature MLP models."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from silentspeechoe.models.build import build_model
from silentspeechoe.models.mlp import (
    FeatureMLP,
    IMUBinauralFeatureMLP,
    IMUFeatureMLP,
)


def test_imu_feature_mlp_forward_shape() -> None:
    model = IMUFeatureMLP()
    x = torch.randn(4, 432)
    lengths = torch.full((4,), 432, dtype=torch.long)

    out = model(x, lengths=lengths)

    assert out.shape == (4, 36)
    assert torch.isfinite(out).all()


def test_imu_binaural_feature_mlp_forward_shape() -> None:
    model = IMUBinauralFeatureMLP()
    x = torch.randn(3, 1296)

    out = model(x)

    assert out.shape == (3, 36)
    assert torch.isfinite(out).all()


def test_feature_mlp_extract_features_shape() -> None:
    model = FeatureMLP(in_features=10, hidden_features=(8, 6), num_classes=3)
    x = torch.randn(5, 10)

    features = model.extract_features(x)

    assert features.shape == (5, 6)
    assert model.embedding_dim == 6


def test_feature_mlp_rejects_wrong_feature_dim() -> None:
    model = FeatureMLP(in_features=10, hidden_features=(8,), num_classes=3)

    with pytest.raises(ValueError, match="Expected 10 input features"):
        model(torch.randn(2, 9))


def test_factory_builds_imu_feature_mlp() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "imu_feature_mlp",
                "in_features": 432,
                "hidden_features": [128, 64],
                "num_classes": 36,
            }
        }
    )

    model = build_model(cfg)

    assert isinstance(model, IMUFeatureMLP)
    assert model.embedding_dim == 64


def test_factory_builds_binaural_feature_mlp() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "imu_feature_mlp_binaural",
                "in_features": 1296,
                "hidden_features": [256, 128],
                "num_classes": 36,
            }
        }
    )

    model = build_model(cfg)

    assert isinstance(model, IMUBinauralFeatureMLP)
    assert model.embedding_dim == 128
