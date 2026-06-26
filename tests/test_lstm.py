"""Tests for LSTM sensor encoders."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from silentspeechoe.models.build import build_model
from silentspeechoe.models.lstm import IMUBinauralBiLSTM


def test_imu_binaural_bilstm_forward_shape() -> None:
    model = IMUBinauralBiLSTM(
        in_channels=18,
        hidden_size=32,
        num_layers=1,
        num_classes=36,
    )
    x = torch.randn(4, 18, 120)
    lengths = torch.tensor([120, 90, 60, 30], dtype=torch.long)

    logits = model(x, lengths=lengths)

    assert logits.shape == (4, 36)
    assert torch.isfinite(logits).all()


def test_imu_binaural_bilstm_extract_features_shape() -> None:
    model = IMUBinauralBiLSTM(
        in_channels=18,
        hidden_size=32,
        num_layers=1,
        num_classes=36,
    )
    x = torch.randn(3, 18, 80)
    lengths = torch.tensor([80, 50, 20], dtype=torch.long)

    features = model.extract_features(x, lengths=lengths)

    assert features.shape == (3, model.embedding_dim)
    assert model.embedding_dim == 128
    assert torch.isfinite(features).all()


def test_factory_builds_imu_binaural_bilstm() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "imu_binaural_bilstm",
                "in_channels": 18,
                "hidden_size": 32,
                "num_layers": 1,
                "num_classes": 36,
            }
        }
    )

    model = build_model(cfg)

    assert isinstance(model, IMUBinauralBiLSTM)
