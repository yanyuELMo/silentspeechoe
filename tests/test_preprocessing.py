"""Tests for bone‑acc preprocessing functions (data‑free, synthetic only)."""

from __future__ import annotations

import numpy as np
import pytest

from silentspeechoe.data.preprocessing import (
    center_bone_acc_axes,
    compute_dynamic_magnitude,
    preprocess_bone_acc_window,
    stack_binaural_bone_acc,
    zscore_signal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_xyz(t: int = 200, seed: int = 42) -> np.ndarray:
    """Create a plausible ``[T, 3]`` bone‑acc window."""
    rng = np.random.default_rng(seed)
    # Simulate constant gravity offset + vibration
    offsets = np.array([0.05, -0.15, 1.02], dtype=np.float32)
    signal = 0.3 * rng.standard_normal((t, 3), dtype=np.float32)
    return signal + offsets


# ---------------------------------------------------------------------------
# center_bone_acc_axes
# ---------------------------------------------------------------------------


class TestCenterBoneAccAxes:
    def test_median_centering(self):
        xyz = _synthetic_xyz(300)
        centered = center_bone_acc_axes(xyz, method="median")
        assert centered.shape == xyz.shape
        assert centered.dtype == np.float32
        # Per‑axis median should be near zero.
        assert np.allclose(np.median(centered, axis=0), 0.0, atol=1e-5)

    def test_mean_centering(self):
        xyz = _synthetic_xyz(200)
        centered = center_bone_acc_axes(xyz, method="mean")
        assert np.allclose(np.mean(centered, axis=0), 0.0, atol=1e-5)

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown centering method"):
            center_bone_acc_axes(_synthetic_xyz(10), method="mode")

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="Expected 2‑D"):
            center_bone_acc_axes(np.zeros(10))

    def test_wrong_columns_raises(self):
        with pytest.raises(ValueError, match="exactly 3 columns"):
            center_bone_acc_axes(np.zeros((10, 4)))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non‑empty"):
            center_bone_acc_axes(np.zeros((0, 3)))


# ---------------------------------------------------------------------------
# compute_dynamic_magnitude
# ---------------------------------------------------------------------------


class TestComputeDynamicMagnitude:
    def test_output_shape(self):
        xyz = _synthetic_xyz(150)
        mag = compute_dynamic_magnitude(xyz)
        assert mag.shape == (150,)
        assert mag.dtype == np.float32

    def test_non_negative(self):
        mag = compute_dynamic_magnitude(_synthetic_xyz(100))
        assert np.all(mag >= 0.0)

    def test_zero_input(self):
        xyz = np.zeros((10, 3), dtype=np.float32)
        mag = compute_dynamic_magnitude(xyz)
        assert np.allclose(mag, 0.0)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="Expected 2‑D"):
            compute_dynamic_magnitude(np.zeros(10))


# ---------------------------------------------------------------------------
# zscore_signal
# ---------------------------------------------------------------------------


class TestZscoreSignal:
    def test_near_zero_mean(self):
        signal = _synthetic_xyz(200)[:, 0]  # just x channel as 1‑D
        out = zscore_signal(signal)
        # Mean around simulation noise level (should be close to 0)
        assert abs(float(np.mean(out))) < 0.2

    def test_near_unit_std(self):
        signal = np.sin(np.linspace(0, 4 * np.pi, 300)).astype(np.float32)
        out = zscore_signal(signal)
        assert 0.9 < float(np.std(out)) < 1.1

    def test_constant_signal_returns_zeros(self):
        out = zscore_signal(np.ones(50, dtype=np.float32) * 3.0)
        assert np.allclose(out, 0.0)

    def test_zero_std_safe(self):
        """Input with negligible variance returns zeros, not NaN."""
        out = zscore_signal(np.full(20, 7.0, dtype=np.float32), eps=1e-6)
        assert np.allclose(out, 0.0)
        assert not np.any(np.isnan(out))

    def test_non_1d_raises(self):
        with pytest.raises(ValueError, match="Expected 1‑D"):
            zscore_signal(np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# preprocess_bone_acc_window
# ---------------------------------------------------------------------------


class TestPreprocessBoneAccWindow:
    def test_output_shape_and_dtype(self):
        xyz = _synthetic_xyz(180)
        out = preprocess_bone_acc_window(xyz)
        assert out.shape == (180,)
        assert out.dtype == np.float32

    def test_end_to_end_finite(self):
        xyz = _synthetic_xyz(256)
        out = preprocess_bone_acc_window(xyz)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# stack_binaural_bone_acc
# ---------------------------------------------------------------------------


class TestStackBinauralBoneAcc:
    def test_same_length(self):
        left = _synthetic_xyz(100, seed=1)
        right = _synthetic_xyz(100, seed=2)
        x, meta = stack_binaural_bone_acc(left, right)
        assert x.shape == (2, 100)
        assert x.dtype == np.float32
        assert meta["left_length"] == 100
        assert meta["right_length"] == 100
        assert meta["length"] == 100

    def test_pads_shorter_left(self):
        left = _synthetic_xyz(50, seed=1)
        right = _synthetic_xyz(100, seed=2)
        x, meta = stack_binaural_bone_acc(left, right)
        assert x.shape == (2, 100)
        assert meta["left_length"] == 50
        assert meta["right_length"] == 100
        assert meta["length"] == 100
        # Left ear padded region (beyond 50) should be zero.
        assert np.allclose(x[0, 50:], 0.0)

    def test_pads_shorter_right(self):
        left = _synthetic_xyz(120, seed=1)
        right = _synthetic_xyz(80, seed=2)
        x, meta = stack_binaural_bone_acc(left, right)
        assert x.shape == (2, 120)
        assert meta["left_length"] == 120
        assert meta["right_length"] == 80
        assert meta["length"] == 120
        # Right ear padded region should be zero.
        assert np.allclose(x[1, 80:], 0.0)

    def test_left_right_independent(self):
        """Left and right channels should differ (different seeds)."""
        left = _synthetic_xyz(60, seed=10)
        right = _synthetic_xyz(60, seed=20)
        x, _ = stack_binaural_bone_acc(left, right)
        assert not np.allclose(x[0], x[1])
