"""Tests for IMU window augmentation utilities."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from silentspeechoe.data.imu_augmentation import (
    IMUAugmentationConfig,
    IMUWindowAugmenter,
    apply_gaussian_noise,
    apply_rotation,
    apply_scaling,
    apply_time_warp,
    augment_imu_window,
    augment_processed_imu_sample,
)
from silentspeechoe.data.imu_preprocessing import PrecomputedIMUDataset


def _synthetic_imu_window(channels: int = 9, length: int = 64) -> torch.Tensor:
    """Create a deterministic ``[C, T]`` IMU window."""
    time = torch.linspace(0.0, 1.0, length, dtype=torch.float32)
    rows = []
    for channel in range(channels):
        rows.append(torch.sin((channel + 1) * torch.pi * time) + channel)
    return torch.stack(rows, dim=0)


def test_rotation_is_identity_for_zero_degrees() -> None:
    x = _synthetic_imu_window()
    out = apply_rotation(x, max_degrees=0.0, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(out, x)


def test_time_warp_is_identity_for_unit_scale() -> None:
    x = _synthetic_imu_window()
    out = apply_time_warp(
        x,
        min_scale=1.0,
        max_scale=1.0,
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.allclose(out, x)


def test_scaling_applies_constant_factor() -> None:
    x = _synthetic_imu_window()
    out = apply_scaling(
        x,
        min_scale=2.0,
        max_scale=2.0,
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.allclose(out, x * 2.0)


def test_gaussian_noise_changes_signal() -> None:
    x = _synthetic_imu_window()
    out = apply_gaussian_noise(
        x,
        min_ratio=0.1,
        max_ratio=0.1,
        generator=torch.Generator().manual_seed(1),
    )
    assert out.shape == x.shape
    assert not torch.allclose(out, x)


def test_full_augmenter_respects_config() -> None:
    x = _synthetic_imu_window()
    config = IMUAugmentationConfig(
        enabled=True,
        sample_prob=1.0,
        rotation_prob=1.0,
        rotation_max_degrees=0.0,
        time_warp_prob=1.0,
        time_warp_min_scale=1.0,
        time_warp_max_scale=1.0,
        scaling_prob=1.0,
        scaling_min_scale=2.0,
        scaling_max_scale=2.0,
        gaussian_noise_prob=1.0,
        gaussian_noise_min_ratio=0.0,
        gaussian_noise_max_ratio=0.0,
    )
    out = augment_imu_window(x, config, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(out, x * 2.0)


def test_sample_probability_zero_keeps_window_unchanged() -> None:
    x = _synthetic_imu_window()
    config = IMUAugmentationConfig(
        enabled=True,
        sample_prob=0.0,
        rotation_prob=1.0,
        rotation_max_degrees=10.0,
        time_warp_prob=1.0,
        time_warp_min_scale=0.9,
        time_warp_max_scale=1.1,
        scaling_prob=1.0,
        scaling_min_scale=2.0,
        scaling_max_scale=2.0,
        gaussian_noise_prob=1.0,
        gaussian_noise_min_ratio=0.1,
        gaussian_noise_max_ratio=0.1,
    )
    out = augment_imu_window(x, config, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(out, x)


def test_processed_sample_keeps_original_window() -> None:
    x = _synthetic_imu_window()
    sample = {
        "x": x,
        "y": 3,
        "length": 64,
        "domain": "normal",
        "subject_id": "sub_00",
        "session_id": "000_123",
        "sentence_id": "nonsem_001",
        "repeat_id": 1,
        "side": "left",
    }
    config = IMUAugmentationConfig(
        enabled=True,
        sample_prob=1.0,
        rotation_prob=0.0,
        time_warp_prob=0.0,
        scaling_prob=1.0,
        scaling_min_scale=2.0,
        scaling_max_scale=2.0,
        gaussian_noise_prob=0.0,
    )
    out = augment_processed_imu_sample(
        sample,
        config,
        generator=torch.Generator().manual_seed(1),
    )
    assert "x_original" in out
    assert out["x_original"] is not sample["x"]
    assert torch.allclose(out["x_original"], x)
    assert torch.allclose(out["x"], x * 2.0)


def test_precomputed_dataset_applies_augmenter(tmp_path: Path) -> None:
    feat_dir = tmp_path / "imu_windows"
    feat_dir.mkdir()

    sample = {
        "x": _synthetic_imu_window(),
        "y": 3,
        "length": 64,
        "domain": "normal",
        "subject_id": "sub_00",
        "session_id": "000_123",
        "sentence_id": "nonsem_001",
        "repeat_id": 1,
        "side": "left",
    }
    torch.save(sample, feat_dir / "sample.pt")

    manifest = {
        "name": "test",
        "num_samples": 1,
        "sides": ["left"],
        "channels": [
            "acc.x",
            "acc.y",
            "acc.z",
            "gyro.x",
            "gyro.y",
            "gyro.z",
            "mag.x",
            "mag.y",
            "mag.z",
        ],
        "target_sample_rate": 200.0,
        "records": [
            {
                "idx": 0,
                "file": "sample.pt",
                "subject_id": "sub_00",
                "session_id": "000_123",
                "event_id": 0,
                "sentence_id": "nonsem_001",
                "sentence_type": "non_semantic",
                "label_id": 3,
                "domain": "normal",
                "repeat_id": 1,
                "side": "left",
                "start_time": 0.0,
                "end_time": 1.0,
                "length": 64,
            }
        ],
    }
    with (feat_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    config = IMUAugmentationConfig(
        enabled=True,
        rotation_prob=0.0,
        time_warp_prob=0.0,
        scaling_prob=1.0,
        scaling_min_scale=2.0,
        scaling_max_scale=2.0,
        gaussian_noise_prob=0.0,
    )
    dataset = PrecomputedIMUDataset(
        feat_dir / "manifest.json",
        feat_dir,
        augmenter=IMUWindowAugmenter(config),
    )

    item = dataset[0]
    assert torch.allclose(item["x"], sample["x"] * 2.0)
