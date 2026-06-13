"""Tests for IMU temporal envelope features (data‑free, synthetic only)."""

from __future__ import annotations

import numpy as np

from silentspeechoe.features.imu_temporal_envelope import (
    _PER_SIGNAL_DIM,
    FEATURE_DIM,
    extract_imu_temporal_envelope_features,
    extract_signal_features,
)


class TestExtractSignalFeatures:
    def test_output_dim(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 2000).astype(np.float32)
        feats = extract_signal_features(signal)
        assert feats.shape == (_PER_SIGNAL_DIM,)
        assert feats.dtype == np.float32

    def test_output_finite(self):
        rng = np.random.default_rng(1)
        signal = rng.normal(0, 1, 3000).astype(np.float32)
        feats = extract_signal_features(signal)
        assert np.all(np.isfinite(feats))

    def test_constant_signal(self):
        signal = np.ones(1000, dtype=np.float32)
        feats = extract_signal_features(signal)
        assert np.all(np.isfinite(feats))

    def test_very_short_signal(self):
        signal = np.array([0.1, 0.2], dtype=np.float32)
        feats = extract_signal_features(signal)
        assert feats.shape == (_PER_SIGNAL_DIM,)
        assert np.all(np.isfinite(feats))

    def test_single_sample(self):
        signal = np.array([3.0], dtype=np.float32)
        feats = extract_signal_features(signal)
        assert feats.shape == (_PER_SIGNAL_DIM,)
        assert np.all(np.isfinite(feats))


class TestExtractIMUTemporalEnvelope:
    def test_output_dim(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0, 1, (9, 2000)).astype(np.float32)
        feats = extract_imu_temporal_envelope_features(x)
        assert feats.shape == (FEATURE_DIM,)
        assert feats.dtype == np.float32

    def test_output_finite(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, (9, 3000)).astype(np.float32)
        feats = extract_imu_temporal_envelope_features(x)
        assert np.all(np.isfinite(feats))

    def test_all_zeros(self):
        x = np.zeros((9, 500), dtype=np.float32)
        feats = extract_imu_temporal_envelope_features(x)
        assert np.all(np.isfinite(feats))

    def test_short_signal(self):
        x = np.ones((9, 10), dtype=np.float32)
        feats = extract_imu_temporal_envelope_features(x)
        assert feats.shape == (FEATURE_DIM,)
        assert np.all(np.isfinite(feats))

    def test_feature_dim_constant(self):
        assert FEATURE_DIM == 432
        assert _PER_SIGNAL_DIM == 36

    def test_magnitude_signals_added(self):
        """Verify the 3 magnitude channels are present in output dimension."""
        assert FEATURE_DIM == 12 * _PER_SIGNAL_DIM  # 9 raw + 3 mag = 12
