"""Tests for IMU MFCC feature extraction (data‑free, synthetic only)."""

from __future__ import annotations

import numpy as np

from silentspeechoe.features.imu_mfcc import (
    _mel_filterbank,
    compute_mfcc,
    extract_imu_mfcc_features,
    extract_imu_mfcc_sequence,
    feature_dim,
    hz_to_mel,
    mel_to_hz,
)


class TestMelScale:
    def test_hz_to_mel_zero(self):
        assert hz_to_mel(np.array(0.0)) == 0.0

    def test_mel_roundtrip(self):
        hz = np.array([1.0, 10.0, 50.0, 90.0])
        mel = hz_to_mel(hz)
        hz_back = mel_to_hz(mel)
        assert np.allclose(hz, hz_back, atol=1e-4)


class TestMelFilterbank:
    def test_shape(self):
        fb = _mel_filterbank(
            n_fft=128,
            sample_rate=200.0,
            n_mels=20,
            fmin=0.5,
            fmax=90.0,
        )
        assert fb.shape == (20, 65)  # n_mels x (n_fft//2 + 1)

    def test_non_negative(self):
        fb = _mel_filterbank(
            n_fft=256,
            sample_rate=200.0,
            n_mels=20,
            fmin=1.0,
            fmax=80.0,
        )
        assert np.all(fb >= 0.0)


class TestComputeMFCC:
    def test_output_shape(self):
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 1, 2000).astype(np.float32)
        mfcc = compute_mfcc(signal, sample_rate=200.0)
        assert mfcc.shape[0] == 13  # n_mfcc
        assert mfcc.shape[1] > 0  # some frames

    def test_output_finite(self):
        rng = np.random.default_rng(1)
        signal = rng.normal(0, 1, 3000).astype(np.float32)
        mfcc = compute_mfcc(signal, sample_rate=200.0)
        assert np.all(np.isfinite(mfcc))

    def test_too_short_signal(self):
        signal = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mfcc = compute_mfcc(signal, sample_rate=200.0, frame_length=50)
        assert mfcc.shape == (13, 0)

    def test_constant_signal(self):
        """Constant signal should not produce NaN."""
        signal = np.ones(2000, dtype=np.float32)
        mfcc = compute_mfcc(signal, sample_rate=200.0)
        assert np.all(np.isfinite(mfcc))

    def test_num_frames(self):
        """8-second signal at 200Hz = 1600 samples.
        frame_length=50, hop_length=10 → (1600-50)//10 + 1 = 156 frames.
        """
        rng = np.random.default_rng(3)
        signal = rng.normal(0, 1, 1600).astype(np.float32)
        mfcc = compute_mfcc(signal, sample_rate=200.0, frame_length=50, hop_length=10)
        expected_frames = (1600 - 50) // 10 + 1  # 156
        assert mfcc.shape[1] == expected_frames


class TestExtractIMUMFCC:
    def test_output_dim(self):
        rng = np.random.default_rng(7)
        x = rng.normal(0, 1, (9, 2000)).astype(np.float32)
        feats = extract_imu_mfcc_features(x, sample_rate=200.0)
        expected = feature_dim(num_channels=9, n_mfcc=13)  # 234
        assert feats.shape == (expected,)
        assert feats.dtype == np.float32

    def test_output_finite(self):
        rng = np.random.default_rng(2)
        x = rng.normal(0, 1, (9, 3000)).astype(np.float32)
        feats = extract_imu_mfcc_features(x, sample_rate=200.0)
        assert np.all(np.isfinite(feats))

    def test_fewer_channels(self):
        """6‑channel (acc+gyro) should give 156 dims."""
        rng = np.random.default_rng(5)
        x = rng.normal(0, 1, (6, 2000)).astype(np.float32)
        feats = extract_imu_mfcc_features(x, sample_rate=200.0)
        expected = feature_dim(num_channels=6, n_mfcc=13)  # 156
        assert feats.shape == (expected,)

    def test_short_signal_returns_zeros(self):
        """Very short signal should return zero vector, not crash."""
        x = np.ones((9, 10), dtype=np.float32)
        feats = extract_imu_mfcc_features(x, sample_rate=200.0, frame_length=50)
        assert feats.shape == (234,)
        assert np.all(np.isfinite(feats))


class TestExtractIMUMFCCSequence:
    def test_preserves_mfcc_time_axis(self):
        rng = np.random.default_rng(11)
        x = rng.normal(0, 1, (18, 1890)).astype(np.float32)
        seq = extract_imu_mfcc_sequence(
            x,
            sample_rate=189.0,
            frame_length=47,
            hop_length=9,
        )
        expected_frames = (1890 - 47) // 9 + 1
        assert seq.shape == (18 * 13, expected_frames)
        assert seq.dtype == np.float32
        assert np.all(np.isfinite(seq))

    def test_short_signal_returns_one_zero_frame(self):
        x = np.ones((18, 10), dtype=np.float32)
        seq = extract_imu_mfcc_sequence(x, sample_rate=189.0, frame_length=47)
        assert seq.shape == (18 * 13, 1)
        assert np.all(seq == 0.0)


class TestFeatureDim:
    def test_default(self):
        assert feature_dim(num_channels=9, n_mfcc=13) == 234

    def test_six_channels(self):
        assert feature_dim(num_channels=6, n_mfcc=13) == 156

    def test_fewer_mfcc(self):
        assert feature_dim(num_channels=9, n_mfcc=8) == 144
