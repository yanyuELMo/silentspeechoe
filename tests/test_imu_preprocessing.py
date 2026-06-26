"""Tests for IMU preprocessing functions (data-free, synthetic only)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from silentspeechoe.data.imu_augmentation import (
    IMUAugmentationConfig,
    IMUWindowAugmenter,
)
from silentspeechoe.data.imu_preprocessing import (
    IMU_CHANNELS,
    IMUDataset,
    PrecomputedIMUDataset,
    build_imu_records,
    clean_imu_timestamps,
    condition_resampled_imu_window,
    find_imu_path,
    imu_pad_collate,
    load_imu,
    median_mad_despike_imu_window,
    preprocess_imu_window,
    resample_imu_window,
    slice_imu_window,
    slice_imu_window_with_time,
)
from silentspeechoe.data.subject_filtering import filter_subject_dataframe

# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _synthetic_imu_df(
    t: int = 500,
    seed: int = 42,
    sample_rate: float = 189.0,
) -> pd.DataFrame:
    """Create a plausible IMU DataFrame with 9 channels."""
    rng = np.random.default_rng(seed)
    timestamps = np.cumsum(rng.uniform(0.004, 0.006, size=t))
    timestamps[0] = 0.0

    data: dict[str, np.ndarray] = {"timestamp": timestamps}
    # Add plausible sensor ranges.
    data["acc.x"] = rng.normal(0, 0.3, t).astype(np.float32)
    data["acc.y"] = rng.normal(0, 0.3, t).astype(np.float32)
    data["acc.z"] = rng.normal(1, 0.3, t).astype(np.float32)  # gravity
    data["gyro.x"] = rng.normal(0, 0.1, t).astype(np.float32)
    data["gyro.y"] = rng.normal(0, 0.1, t).astype(np.float32)
    data["gyro.z"] = rng.normal(0, 0.1, t).astype(np.float32)
    data["mag.x"] = rng.normal(30, 5, t).astype(np.float32)
    data["mag.y"] = rng.normal(0, 5, t).astype(np.float32)
    data["mag.z"] = rng.normal(-20, 5, t).astype(np.float32)

    return pd.DataFrame(data)


def _synthetic_imu_ndarray(
    t: int = 200,
    seed: int = 42,
    sample_rate: float = 189.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(timestamps, values)`` where values is ``[T, 9]``."""
    rng = np.random.default_rng(seed)
    timestamps = np.cumsum(rng.uniform(0.004, 0.006, size=t)).astype(np.float64)
    timestamps[0] = 0.0
    values = np.column_stack(
        [
            rng.normal(0, 0.3, t),  # acc.x
            rng.normal(0, 0.3, t),  # acc.y
            rng.normal(1, 0.3, t),  # acc.z
            rng.normal(0, 0.1, t),  # gyro.x
            rng.normal(0, 0.1, t),  # gyro.y
            rng.normal(0, 0.1, t),  # gyro.z
            rng.normal(30, 5, t),  # mag.x
            rng.normal(0, 5, t),  # mag.y
            rng.normal(-20, 5, t),  # mag.z
        ]
    ).astype(np.float32)
    return timestamps, values


# ---------------------------------------------------------------------------
# load_imu
# ---------------------------------------------------------------------------


class TestLoadIMU:
    def test_load_synthetic(self, tmp_path):
        path = tmp_path / "test__imu.csv"
        df_in = _synthetic_imu_df(100)
        df_in.to_csv(path, index=False)
        df_out = load_imu(path)
        assert set(IMU_CHANNELS).issubset(set(df_out.columns))
        assert "timestamp" in df_out.columns
        assert len(df_out) == 100

    def test_missing_columns_raises(self, tmp_path):
        path = tmp_path / "bad__imu.csv"
        pd.DataFrame({"timestamp": [0.0, 1.0], "acc.x": [0.1, 0.2]}).to_csv(
            path, index=False
        )
        with pytest.raises(ValueError, match="Missing columns"):
            load_imu(path)


# ---------------------------------------------------------------------------
# slice_imu_window
# ---------------------------------------------------------------------------


class TestSliceIMUWindow:
    def test_basic_window(self):
        df = _synthetic_imu_df(500)
        values = slice_imu_window(df, 0.5, 1.5)
        assert values.ndim == 2
        assert values.shape[1] == 9
        assert values.dtype == np.float32

    def test_empty_window(self):
        df = _synthetic_imu_df(100)
        values = slice_imu_window(df, 9999.0, 9999.1)
        assert values.shape == (0, 9)

    def test_with_padding(self):
        df = _synthetic_imu_df(500)
        v_no_pad = slice_imu_window(df, 0.5, 1.0, padding_sec=0.0)
        v_pad = slice_imu_window(df, 0.5, 1.0, padding_sec=0.2)
        assert v_pad.shape[0] >= v_no_pad.shape[0]

    def test_with_time_returns_timestamps(self):
        df = _synthetic_imu_df(500)
        ts, vals = slice_imu_window_with_time(df, 0.5, 1.5)
        assert ts.ndim == 1
        assert vals.ndim == 2
        assert ts.shape[0] == vals.shape[0]
        assert ts.dtype == np.float64


# ---------------------------------------------------------------------------
# clean_imu_timestamps
# ---------------------------------------------------------------------------


class TestCleanIMUTimestamps:
    def test_no_nan_passthrough(self):
        ts, vals = _synthetic_imu_ndarray(200)
        ts_c, vals_c = clean_imu_timestamps(ts, vals)
        # Sorted input with no duplicates should pass through mostly intact.
        assert ts_c.shape[0] <= ts.shape[0]
        assert vals_c.shape[0] <= vals.shape[0]
        assert np.all(np.diff(ts_c) > 0)  # strictly monotonic

    def test_removes_nan(self):
        ts, vals = _synthetic_imu_ndarray(100)
        vals[10, 0] = np.nan
        vals[20, 3] = np.inf
        ts_c, vals_c = clean_imu_timestamps(ts, vals)
        assert ts_c.shape[0] <= ts.shape[0] - 2  # at least 2 rows removed
        assert np.all(np.isfinite(vals_c))

    def test_removes_duplicate_timestamps(self):
        ts = np.array([0.0, 0.1, 0.1, 0.2, 0.3], dtype=np.float64)
        vals = np.random.default_rng(0).normal(size=(5, 9)).astype(np.float32)
        ts_c, vals_c = clean_imu_timestamps(ts, vals)
        assert ts_c.shape[0] <= 4  # at least one duplicate removed
        assert len(np.unique(ts_c)) == ts_c.shape[0]

    def test_empty_input(self):
        ts = np.array([], dtype=np.float64)
        vals = np.empty((0, 9), dtype=np.float32)
        ts_c, vals_c = clean_imu_timestamps(ts, vals)
        assert ts_c.shape[0] == 0
        assert vals_c.shape[0] == 0

    def test_strictly_monotonic_output(self):
        ts, vals = _synthetic_imu_ndarray(300)
        # Shuffle to test sort + dedup.
        rng = np.random.default_rng(7)
        perm = rng.permutation(300)
        ts_shuffled = ts[perm]
        vals_shuffled = vals[perm]
        ts_c, vals_c = clean_imu_timestamps(ts_shuffled, vals_shuffled)
        assert np.all(np.diff(ts_c) > 0.0)

    def test_all_nan_returns_empty(self):
        ts = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        vals = np.full((3, 9), np.nan, dtype=np.float32)
        ts_c, vals_c = clean_imu_timestamps(ts, vals)
        assert ts_c.shape[0] == 0
        assert vals_c.shape[0] == 0


# ---------------------------------------------------------------------------
# resample_imu_window
# ---------------------------------------------------------------------------


class TestResampleIMUWindow:
    def test_output_shape(self):
        ts, vals = _synthetic_imu_ndarray(600)
        duration = ts[-1] - ts[0]
        out = resample_imu_window(ts, vals, ts[0], ts[-1], target_sample_rate=200.0)
        expected_t = max(1, int(round(duration * 200.0)))
        assert out.shape == (9, expected_t)
        assert out.dtype == np.float32

    def test_exact_duration(self):
        """For a 1-second window, 200 Hz should give ~200 samples."""
        ts = np.linspace(0, 1, 190, dtype=np.float64)  # ~190 Hz original
        vals = np.random.default_rng(1).normal(size=(190, 9)).astype(np.float32)
        out = resample_imu_window(ts, vals, 0.0, 1.0, target_sample_rate=200.0)
        assert out.shape[0] == 9
        assert abs(out.shape[1] - 200) <= 1  # near 200

    def test_short_window(self):
        """Very short window should still return at least 1 sample."""
        ts = np.array([0.0, 0.005, 0.01], dtype=np.float64)
        vals = np.random.default_rng(2).normal(size=(3, 9)).astype(np.float32)
        out = resample_imu_window(ts, vals, 0.0, 0.01, target_sample_rate=200.0)
        assert out.shape[0] == 9
        assert out.shape[1] >= 1

    def test_zero_duration_returns_empty(self):
        ts = np.array([0.0], dtype=np.float64)
        vals = np.random.default_rng(3).normal(size=(1, 9)).astype(np.float32)
        out = resample_imu_window(ts, vals, 0.5, 0.5, target_sample_rate=200.0)
        assert out.shape == (9, 0)

    def test_single_sample_repeats(self):
        ts = np.array([0.0], dtype=np.float64)
        vals = np.array([[1.0] * 9], dtype=np.float32)
        out = resample_imu_window(ts, vals, 0.0, 0.1, target_sample_rate=100.0)
        assert out.shape == (9, 10)  # 0.1s * 100Hz = 10
        # All values should equal the single input.
        for c in range(9):
            assert np.allclose(out[c], vals[0, c])

    def test_no_samples_returns_zeros(self):
        ts = np.array([], dtype=np.float64)
        vals = np.empty((0, 9), dtype=np.float32)
        out = resample_imu_window(ts, vals, 0.0, 1.0, target_sample_rate=200.0)
        assert out.shape == (9, 200)

    def test_finite_output(self):
        ts, vals = _synthetic_imu_ndarray(500)
        out = resample_imu_window(ts, vals, ts[0], ts[-1], target_sample_rate=200.0)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# post-resampling conditioning
# ---------------------------------------------------------------------------


class TestIMUConditioning:
    def test_median_mad_despike_clips_only_large_spike(self):
        x = np.zeros((9, 101), dtype=np.float32)
        x[0] = np.linspace(-1.0, 1.0, 101, dtype=np.float32)
        x[0, 50] = 100.0

        out = median_mad_despike_imu_window(x, threshold=8.0)

        assert out.shape == x.shape
        assert out.dtype == np.float32
        assert out[0, 50] < 100.0
        assert np.allclose(out[1:], x[1:])

    def test_conditioning_outputs_zero_mean_unit_std(self):
        rng = np.random.default_rng(7)
        x = rng.normal(loc=10.0, scale=2.0, size=(9, 300)).astype(np.float32)

        out = condition_resampled_imu_window(x)

        assert out.shape == x.shape
        assert np.all(np.isfinite(out))
        assert np.allclose(out.mean(axis=1), 0.0, atol=1e-5)
        assert np.allclose(out.std(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# preprocess_imu_window (end-to-end with temp files)
# ---------------------------------------------------------------------------


class TestPreprocessIMUWindow:
    def test_end_to_end(self, tmp_path):
        """Full pipeline on a synthetic CSV file."""
        path = tmp_path / "test__imu.csv"
        df = _synthetic_imu_df(800)
        df.to_csv(path, index=False)

        x, meta = preprocess_imu_window(
            path,
            start_sec=0.5,
            end_sec=2.5,
            target_sample_rate=200.0,
            padding_sec=0.0,
        )

        assert x.shape[0] == 9
        assert x.shape[1] > 0
        assert x.dtype == np.float32
        assert meta["length"] == x.shape[1]
        assert meta["num_finite"] > 0
        assert np.all(np.isfinite(x))

    def test_empty_window(self, tmp_path):
        path = tmp_path / "test__imu.csv"
        df = _synthetic_imu_df(100)
        df.to_csv(path, index=False)

        x, meta = preprocess_imu_window(
            path,
            start_sec=9999.0,
            end_sec=9999.1,
            target_sample_rate=200.0,
        )
        assert x.shape == (9, 0)
        assert meta["length"] == 0
        assert meta["num_finite"] == 0

    def test_with_padding(self, tmp_path):
        path = tmp_path / "test__imu.csv"
        df = _synthetic_imu_df(800)
        df.to_csv(path, index=False)

        x_no_pad, m_no = preprocess_imu_window(
            path, 0.5, 2.0, target_sample_rate=200.0, padding_sec=0.0
        )
        x_pad, m_pad = preprocess_imu_window(
            path, 0.5, 2.0, target_sample_rate=200.0, padding_sec=0.3
        )
        # Padded should have more samples.
        assert m_pad["length"] > m_no["length"]
        assert x_pad.shape[1] > x_no_pad.shape[1]

    def test_normalize_option(self, tmp_path):
        path = tmp_path / "test__imu.csv"
        df = _synthetic_imu_df(800)
        df.to_csv(path, index=False)

        x_norm, _ = preprocess_imu_window(
            path, 0.5, 2.5, target_sample_rate=200.0, normalize=True
        )
        # Each channel should be roughly zero-mean, unit-std.
        for c in range(9):
            assert abs(float(x_norm[c].mean())) < 0.3
            assert 0.5 < float(x_norm[c].std()) < 1.5

    def test_full_conditioning_options(self, tmp_path):
        path = tmp_path / "test__imu.csv"
        df = _synthetic_imu_df(800)
        df.to_csv(path, index=False)

        x, meta = preprocess_imu_window(
            path,
            0.5,
            2.5,
            target_sample_rate=200.0,
            despike=True,
            remove_dc=True,
            normalize=True,
        )

        assert meta["despike"] is True
        assert meta["remove_dc"] is True
        assert meta["normalize"] is True
        assert np.allclose(x.mean(axis=1), 0.0, atol=1e-5)
        assert np.allclose(x.std(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# IMUDataset
# ---------------------------------------------------------------------------


class TestIMUDataset:
    @staticmethod
    def _make_records(tmp_path, n: int = 5) -> list[dict]:
        """Create synthetic records pointing to real temp CSV files."""
        records: list[dict] = []
        for i in range(n):
            path = tmp_path / f"sub_0{i}__imu.csv"
            df = _synthetic_imu_df(600, seed=i)
            df.to_csv(path, index=False)
            records.append(
                {
                    "subject_id": f"sub_0{i}",
                    "session_id": f"00{i}_1234567890",
                    "event_id": i,
                    "sentence_id": f"nonsem_{i + 1:03d}",
                    "label_id": i,
                    "domain": "normal",
                    "repeat_id": 1,
                    "sentence_type": "non_semantic",
                    "side": "left",
                    "path": path,
                    "start_time": 0.3,
                    "end_time": 2.3,
                }
            )
        return records

    def test_len(self, tmp_path):
        records = self._make_records(tmp_path, 5)
        ds = IMUDataset(records)
        assert len(ds) == 5

    def test_getitem_shape(self, tmp_path):
        records = self._make_records(tmp_path, 3)
        ds = IMUDataset(records, target_sample_rate=200.0)
        item = ds[0]
        assert item["x"].shape[0] == 9
        assert item["x"].shape[1] > 0
        assert item["x"].dtype == torch.float32
        assert isinstance(item["y"], int)
        assert isinstance(item["length"], int)
        assert item["length"] == item["x"].shape[1]

    def test_getitem_metadata(self, tmp_path):
        records = self._make_records(tmp_path, 1)
        ds = IMUDataset(records)
        item = ds[0]
        assert item["domain"] == "normal"
        assert item["subject_id"] == "sub_00"
        assert item["side"] == "left"
        assert item["sentence_id"] == "nonsem_001"
        assert item["repeat_id"] == 1

    def test_getitem_preserves_original_when_augmented(self, tmp_path):
        records = self._make_records(tmp_path, 1)
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
        ds = IMUDataset(records, augmenter=IMUWindowAugmenter(config))
        item = ds[0]
        assert "x_original" in item
        assert torch.allclose(item["x_original"] * 2.0, item["x"])

    def test_empty_window_record(self, tmp_path):
        """Record with window outside data range should return [9, 0]."""
        records = self._make_records(tmp_path, 1)
        records[0]["start_time"] = 9999.0
        records[0]["end_time"] = 9999.1
        ds = IMUDataset(records)
        item = ds[0]
        assert item["x"].shape == (9, 0)
        assert item["length"] == 0

    def test_df_cache_reuse(self, tmp_path):
        """Multiple samples from the same CSV should reuse the cached DF."""
        path = tmp_path / "shared__imu.csv"
        df = _synthetic_imu_df(1000)
        df.to_csv(path, index=False)

        records = []
        for i in range(3):
            records.append(
                {
                    "subject_id": "sub_00",
                    "session_id": "000_123",
                    "event_id": i,
                    "sentence_id": f"nonsem_{i + 1:03d}",
                    "label_id": i,
                    "domain": "normal",
                    "repeat_id": 1,
                    "sentence_type": "non_semantic",
                    "side": "left",
                    "path": path,
                    "start_time": float(i) * 0.5,
                    "end_time": float(i) * 0.5 + 2.0,
                }
            )
        ds = IMUDataset(records)
        for i in range(3):
            item = ds[i]
            assert item["x"].shape[0] == 9
        # Cache should have exactly one entry.
        assert len(ds._df_cache) == 1


# ---------------------------------------------------------------------------
# imu_pad_collate
# ---------------------------------------------------------------------------


class TestIMUPadCollate:
    def test_basic_collate(self):
        batch = [
            {
                "x": torch.randn(9, 100),
                "y": 0,
                "length": 100,
                "domain": "normal",
                "subject_id": "sub_00",
                "session_id": "s1",
                "sentence_id": "nonsem_001",
                "repeat_id": 1,
                "side": "left",
            },
            {
                "x": torch.randn(9, 150),
                "y": 1,
                "length": 150,
                "domain": "whisper",
                "subject_id": "sub_01",
                "session_id": "s2",
                "sentence_id": "nonsem_002",
                "repeat_id": 2,
                "side": "left",
            },
        ]
        out = imu_pad_collate(batch)
        assert out["x"].shape == (2, 9, 150)
        assert out["y"].tolist() == [0, 1]
        assert out["lengths"].tolist() == [100, 150]
        assert out["domain"] == ["normal", "whisper"]
        assert out["side"] == ["left", "left"]
        # Second sample should be untouched, first zero-padded beyond 100.
        assert torch.allclose(out["x"][0, :, 100:], torch.zeros(9, 50))
        assert torch.allclose(out["x"][1], batch[1]["x"])

    def test_single_sample(self):
        batch = [
            {
                "x": torch.randn(9, 80),
                "y": 5,
                "length": 80,
                "domain": "silent",
                "subject_id": "sub_02",
                "session_id": "s3",
                "sentence_id": "sem_001",
                "repeat_id": 1,
                "side": "right",
            }
        ]
        out = imu_pad_collate(batch)
        assert out["x"].shape == (1, 9, 80)
        assert out["lengths"].tolist() == [80]

    def test_empty_batch(self):
        out = imu_pad_collate([])
        assert out["x"].shape == (0, 9, 0)
        assert len(out["y"]) == 0

    def test_wrong_dim_raises(self):
        with pytest.raises(ValueError, match="Expected x of shape"):
            imu_pad_collate([{"x": torch.randn(100), "y": 0}])

    def test_missing_optional_fields(self):
        """Collate should work even when optional keys are absent."""
        batch = [
            {
                "x": torch.randn(9, 50),
                "y": 0,
                "length": 50,
                "subject_id": "sub_00",
            }
        ]
        out = imu_pad_collate(batch)
        assert out["domain"] == [""]
        assert out["session_id"] == [""]


# ---------------------------------------------------------------------------
# find_imu_path
# ---------------------------------------------------------------------------


class TestFindIMUPath:
    def test_finds_file(self, tmp_path):
        """Simulate the raw data directory structure."""
        raw = tmp_path / "data" / "raw"
        imu_dir = raw / "left" / "00" / "semantic"
        imu_dir.mkdir(parents=True)
        imu_file = imu_dir / "sensor_003_123__imu.csv"
        imu_file.write_text(
            "timestamp,acc.x,acc.y,acc.z,gyro.x,gyro.y,gyro.z,mag.x,mag.y,mag.z\n"
        )

        path = find_imu_path(
            "00",
            "left",
            "semantic",
            base_dir=tmp_path,
            raw_root="data/raw",
        )
        assert path is not None
        assert path.name == "sensor_003_123__imu.csv"

    def test_returns_none_when_missing(self, tmp_path):
        path = find_imu_path(
            "99",
            "left",
            "semantic",
            base_dir=tmp_path,
            raw_root="data/raw",
        )
        assert path is None

    def test_missing_directory(self, tmp_path):
        path = find_imu_path(
            "00",
            "left",
            "nonexistent",
            base_dir=tmp_path,
            raw_root="data/raw",
        )
        assert path is None


# ---------------------------------------------------------------------------
# build_imu_records (requires real events.csv — skip if missing)
# ---------------------------------------------------------------------------


class TestBuildIMURecords:
    def test_requires_events_csv(self):
        """build_imu_records raises if events.csv is missing."""
        with pytest.raises(FileNotFoundError):
            build_imu_records(
                events_path="/nonexistent/events.csv",
                raw_dir="/nonexistent/raw",
                sides=["left"],
            )

    def test_dataframe_filter_excludes_subjects_26_and_51(self):
        """Global subject filtering removes excluded users from event tables."""
        df = pd.DataFrame(
            {
                "subject_id": ["sub_25", "26", "sub_51", "sub_52"],
                "event_id": [1, 2, 3, 4],
            }
        )
        filtered = filter_subject_dataframe(df)
        assert filtered["subject_id"].tolist() == ["sub_25", "sub_52"]


# ---------------------------------------------------------------------------
# PrecomputedIMUDataset
# ---------------------------------------------------------------------------


class TestPrecomputedIMUDataset:
    @staticmethod
    def _make_manifest_and_files(tmp_path, n: int = 5) -> tuple[Path, Path]:
        """Create a synthetic manifest + .pt files, return (manifest_path, dir)."""
        import json

        feat_dir = tmp_path / "imu_windows"
        feat_dir.mkdir()

        records: list[dict] = []
        for i in range(n):
            fname = f"sub_0{i}_left_00{i}_{i:05d}.pt"
            sample = {
                "x": torch.randn(9, 500 + i * 100),
                "y": i % 36,
                "length": 500 + i * 100,
                "domain": ["normal", "whisper", "silent"][i % 3],
                "subject_id": f"sub_0{i}",
                "session_id": f"00{i}_123",
                "sentence_id": f"nonsem_{i + 1:03d}",
                "repeat_id": (i % 2) + 1,
                "side": "left",
            }
            torch.save(sample, feat_dir / fname)
            records.append(
                {
                    "idx": i,
                    "file": fname,
                    "subject_id": f"sub_0{i}",
                    "session_id": f"00{i}_123",
                    "event_id": i,
                    "sentence_id": f"nonsem_{i + 1:03d}",
                    "sentence_type": "non_semantic",
                    "label_id": i % 36,
                    "domain": ["normal", "whisper", "silent"][i % 3],
                    "repeat_id": (i % 2) + 1,
                    "side": "left",
                    "start_time": float(i),
                    "end_time": float(i) + 2.5,
                    "length": 500 + i * 100,
                    "num_raw_samples": 450 + i * 90,
                    "raw_path": f"data/raw/left/0{i}/non-semantic/sensor__imu.csv",
                }
            )

        manifest = {
            "name": "test_imu",
            "num_samples": n,
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
            "padding_sec": 0.0,
            "normalize": False,
            "length_stats": {"min": 20, "max": 1000},
            "records": records,
            "skipped": [],
        }
        manifest_path = feat_dir / "manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f)

        return manifest_path, feat_dir

    def test_len(self, tmp_path):
        manifest_path, feat_dir = self._make_manifest_and_files(tmp_path, 5)
        ds = PrecomputedIMUDataset(manifest_path, feat_dir)
        assert len(ds) == 5

    def test_getitem_shape(self, tmp_path):
        manifest_path, feat_dir = self._make_manifest_and_files(tmp_path, 3)
        ds = PrecomputedIMUDataset(manifest_path, feat_dir)
        item = ds[0]
        assert item["x"].shape[0] == 9
        assert item["x"].shape[1] == 500
        assert item["x"].dtype == torch.float32
        assert isinstance(item["y"], int)
        assert isinstance(item["length"], int)

    def test_getitem_metadata(self, tmp_path):
        manifest_path, feat_dir = self._make_manifest_and_files(tmp_path, 1)
        ds = PrecomputedIMUDataset(manifest_path, feat_dir)
        item = ds[0]
        assert item["domain"] == "normal"
        assert item["subject_id"] == "sub_00"
        assert item["side"] == "left"
        assert item["sentence_id"] == "nonsem_001"

    def test_collate_with_padding(self, tmp_path):
        manifest_path, feat_dir = self._make_manifest_and_files(tmp_path, 4)
        ds = PrecomputedIMUDataset(manifest_path, feat_dir)
        batch = [ds[i] for i in range(4)]
        out = imu_pad_collate(batch)
        max_len = max(item["length"] for item in batch)
        assert out["x"].shape == (4, 9, max_len)
        assert out["lengths"].tolist() == [500, 600, 700, 800]

    def test_domain_distribution(self, tmp_path):
        manifest_path, feat_dir = self._make_manifest_and_files(tmp_path, 6)
        ds = PrecomputedIMUDataset(manifest_path, feat_dir)
        domains = [ds[i]["domain"] for i in range(len(ds))]
        assert "normal" in domains
        assert "whisper" in domains
        assert "silent" in domains

    def test_subset_works(self, tmp_path):
        manifest_path, feat_dir = self._make_manifest_and_files(tmp_path, 5)
        ds = PrecomputedIMUDataset(manifest_path, feat_dir)
        sub = torch.utils.data.Subset(ds, [0, 2, 4])
        assert len(sub) == 3
        batch = [sub[i] for i in range(3)]
        out = imu_pad_collate(batch)
        assert out["x"].shape[0] == 3

    def test_channel_indices_accgyro(self, tmp_path):
        """Selecting indices [0..5] returns [6, T] tensors."""
        manifest_path, feat_dir = self._make_manifest_and_files(tmp_path, 3)
        ds = PrecomputedIMUDataset(
            manifest_path, feat_dir, channel_indices=[0, 1, 2, 3, 4, 5]
        )
        assert ds.num_channels == 6
        item = ds[0]
        assert item["x"].shape[0] == 6
        assert item["x"].shape[1] == 500
        assert item["length"] == 500  # unchanged

    def test_channel_indices_none(self, tmp_path):
        """None (default) returns all 9 channels."""
        manifest_path, feat_dir = self._make_manifest_and_files(tmp_path, 2)
        ds = PrecomputedIMUDataset(manifest_path, feat_dir, channel_indices=None)
        assert ds.num_channels == 9
        item = ds[0]
        assert item["x"].shape[0] == 9

    def test_getitem_preserves_original_when_augmented(self, tmp_path):
        manifest_path, feat_dir = self._make_manifest_and_files(tmp_path, 1)
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
        ds = PrecomputedIMUDataset(
            manifest_path,
            feat_dir,
            augmenter=IMUWindowAugmenter(config),
        )
        item = ds[0]
        assert "x_original" in item
        assert torch.allclose(item["x_original"] * 2.0, item["x"])

    def test_channel_indices_collate(self, tmp_path):
        """Padding should work with reduced channels."""
        manifest_path, feat_dir = self._make_manifest_and_files(tmp_path, 4)
        ds = PrecomputedIMUDataset(
            manifest_path, feat_dir, channel_indices=[0, 1, 2, 3, 4, 5]
        )
        batch = [ds[i] for i in range(4)]
        out = imu_pad_collate(batch)
        max_len = max(item["length"] for item in batch)
        assert out["x"].shape == (4, 6, max_len)

    def test_channel_indices_single(self, tmp_path):
        """Single channel subset works."""
        manifest_path, feat_dir = self._make_manifest_and_files(tmp_path, 2)
        ds = PrecomputedIMUDataset(manifest_path, feat_dir, channel_indices=[0])
        assert ds.num_channels == 1
        item = ds[0]
        assert item["x"].shape[0] == 1

    def test_excluded_subjects_are_filtered_from_manifest(self, tmp_path):
        """Excluded subjects should never appear in precomputed datasets."""
        import json

        manifest_path, feat_dir = self._make_manifest_and_files(tmp_path, 4)
        with manifest_path.open("r") as f:
            manifest = json.load(f)

        manifest["records"][0]["subject_id"] = "sub_26"
        manifest["records"][1]["subject_id"] = "sub_51"
        with manifest_path.open("w") as f:
            json.dump(manifest, f)

        ds = PrecomputedIMUDataset(manifest_path, feat_dir)
        assert len(ds) == 2
        assert all(
            item["subject_id"] not in {"sub_26", "sub_51"} for item in ds.records
        )
