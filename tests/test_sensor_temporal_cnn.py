"""Tests for raw sensor temporal CNN entry points."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from silentspeechoe.models.build import build_model
from silentspeechoe.models.tcn import (
    BoneAccTemporalCNN,
    BoneRawTCN,
    IMUBinauralLRDiffTemporalCNN,
    IMUTemporalCNN,
)


def test_bone_acc_temporal_cnn_forward_shape() -> None:
    """Single-ear bone_acc windows use the 3-channel entry point."""
    model = BoneAccTemporalCNN()
    x = torch.randn(4, 3, 1000)
    lengths = torch.tensor([1000, 800, 500, 120], dtype=torch.long)

    out = model(x, lengths=lengths)

    assert out.shape == (4, 36)
    assert torch.isfinite(out).all()


def test_imu_temporal_cnn_forward_shape() -> None:
    """IMU windows use the 9-channel entry point."""
    model = IMUTemporalCNN()
    x = torch.randn(4, 9, 189)
    lengths = torch.tensor([189, 150, 100, 50], dtype=torch.long)

    out = model(x, lengths=lengths)

    assert out.shape == (4, 36)
    assert torch.isfinite(out).all()


def test_bone_raw_tcn_keeps_binaural_default() -> None:
    """The backward-compatible raw bone TCN still defaults to 6 channels."""
    model = BoneRawTCN()
    x = torch.randn(2, 6, 1000)
    lengths = torch.tensor([1000, 900], dtype=torch.long)

    out = model(x, lengths=lengths)

    assert out.shape == (2, 36)


def test_imu_binaural_lrdiff_temporal_cnn_forward_shape() -> None:
    """Binaural left/right/difference IMU windows use 27 channels."""
    model = IMUBinauralLRDiffTemporalCNN()
    x = torch.randn(3, 27, 189)
    lengths = torch.tensor([189, 150, 80], dtype=torch.long)

    out = model(x, lengths=lengths)

    assert out.shape == (3, 36)
    assert torch.isfinite(out).all()


def test_factory_builds_bone_acc_temporal_cnn() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "bone_acc_temporal_cnn",
                "in_channels": 3,
                "num_classes": 36,
            }
        }
    )

    model = build_model(cfg)

    assert isinstance(model, BoneAccTemporalCNN)


def test_factory_builds_imu_temporal_cnn() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "imu_temporal_cnn",
                "in_channels": 9,
                "num_classes": 36,
            }
        }
    )

    model = build_model(cfg)

    assert isinstance(model, IMUTemporalCNN)


def test_factory_builds_imu_binaural_lrdiff_temporal_cnn() -> None:
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "imu_binaural_lrdiff_temporal_cnn",
                "in_channels": 27,
                "num_classes": 36,
            }
        }
    )

    model = build_model(cfg)

    assert isinstance(model, IMUBinauralLRDiffTemporalCNN)
