"""Temporal envelope handcrafted features for IMU sensor sequences.

Extracts a fixed‑length 432‑dim feature vector from a ``[9, T]`` IMU
window by computing global, differential, envelope, and segmented
statistics over 12 derived signals (9 raw channels + 3 magnitudes).

All computation is pure NumPy — no audio or ML dependencies.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RAW_CHANNELS: list[str] = [
    "acc.x",
    "acc.y",
    "acc.z",
    "gyro.x",
    "gyro.y",
    "gyro.z",
    "mag.x",
    "mag.y",
    "mag.z",
]

_MAG_CHANNELS: list[str] = ["acc_mag", "gyro_mag", "mag_mag"]

ALL_SIGNAL_NAMES: list[str] = _RAW_CHANNELS + _MAG_CHANNELS

_NUM_SEGMENTS: int = 4

# Per‑signal feature groups.
_GLOBAL_STAT_NAMES: list[str] = [
    "mean",
    "std",
    "min",
    "max",
    "range",
    "rms",
    "mean_abs",
    "median",
    "iqr",
]
_DIFF_STAT_NAMES: list[str] = [
    "mean_abs_diff",
    "std_diff",
    "max_abs_diff",
    "zero_crossing_rate_diff",
]
_ENVELOPE_STAT_NAMES: list[str] = [
    "env_mean",
    "env_std",
    "env_max",
    "env_energy",
    "env_time_to_peak",
    "env_mean_slope",
    "env_std_slope",
]
_SEGMENT_STAT_NAMES: list[str] = ["seg_mean", "seg_std", "seg_rms", "seg_max"]

FEATURE_DIM: int = len(ALL_SIGNAL_NAMES) * (
    len(_GLOBAL_STAT_NAMES)
    + len(_DIFF_STAT_NAMES)
    + len(_ENVELOPE_STAT_NAMES)
    + _NUM_SEGMENTS * len(_SEGMENT_STAT_NAMES)
)  # 432

_PER_SIGNAL_DIM: int = FEATURE_DIM // len(ALL_SIGNAL_NAMES)  # 36


def derived_signal_names_for_num_channels(num_channels: int) -> list[str]:
    """Return magnitude signal names available for a channel count.

    The standard channel order is expected: acc.xyz, gyro.xyz, then mag.xyz.
    Six-channel acc+gyro input therefore yields acc_mag and gyro_mag only.
    """
    if num_channels < 3:
        raise ValueError(
            f"IMU temporal-envelope features need at least 3 channels, "
            f"got {num_channels}"
        )
    if num_channels < 6:
        return ["acc_mag"]
    if num_channels < 9:
        return ["acc_mag", "gyro_mag"]
    return ["acc_mag", "gyro_mag", "mag_mag"]


def feature_dim_for_num_channels(num_channels: int) -> int:
    """Return the temporal-envelope feature dimension for C input channels."""
    num_signals = num_channels + len(
        derived_signal_names_for_num_channels(num_channels)
    )
    return num_signals * _PER_SIGNAL_DIM


def _moving_average(signal: np.ndarray, window_size: int = 11) -> np.ndarray:
    """Boxcar moving average with reflective padding at boundaries."""
    if window_size <= 1 or signal.shape[0] < window_size:
        return signal.copy()
    kernel = np.ones(window_size, dtype=np.float64) / window_size
    return np.convolve(signal, kernel, mode="same")


# ---------------------------------------------------------------------------
# Per‑signal feature extraction
# ---------------------------------------------------------------------------


def _compute_global_stats(signal: np.ndarray) -> np.ndarray:
    """A. Global temporal statistics — 9 features."""
    return np.array(
        [
            float(np.mean(signal)),
            float(np.std(signal, ddof=0)),
            float(np.min(signal)),
            float(np.max(signal)),
            float(np.max(signal) - np.min(signal)),
            float(np.sqrt(np.mean(signal**2))),
            float(np.mean(np.abs(signal))),
            float(np.median(signal)),
            float(np.subtract(*np.percentile(signal, [75, 25]))),
        ],
        dtype=np.float32,
    )


def _compute_diff_stats(signal: np.ndarray) -> np.ndarray:
    """B. First‑difference / dynamics statistics — 4 features."""
    T = signal.shape[0]
    if T < 2:
        return np.zeros(len(_DIFF_STAT_NAMES), dtype=np.float32)

    dx = np.diff(signal)
    zcr = float(np.sum(np.abs(np.diff(np.signbit(dx)))) / max(T - 1, 1))
    return np.array(
        [
            float(np.mean(np.abs(dx))),
            float(np.std(dx, ddof=0)),
            float(np.max(np.abs(dx))),
            zcr,
        ],
        dtype=np.float32,
    )


def _compute_envelope_features(
    signal: np.ndarray,
    window_size: int = 11,
) -> np.ndarray:
    """C. Envelope features — 7 features."""
    T = signal.shape[0]
    abs_sig = np.abs(signal)
    envelope = _moving_average(abs_sig, window_size=window_size)

    env_energy = float(np.mean(envelope**2))
    env_time_to_peak = float(np.argmax(envelope)) / max(T - 1, 1) if T > 1 else 0.0

    # Slope of envelope.
    if T >= 2:
        d_env = np.diff(envelope)
        env_mean_slope = float(np.mean(d_env))
        env_std_slope = float(np.std(d_env, ddof=0))
    else:
        env_mean_slope = 0.0
        env_std_slope = 0.0

    return np.array(
        [
            float(np.mean(envelope)),
            float(np.std(envelope, ddof=0)),
            float(np.max(envelope)),
            env_energy,
            env_time_to_peak,
            env_mean_slope,
            env_std_slope,
        ],
        dtype=np.float32,
    )


def _compute_segmented_stats(signal: np.ndarray) -> np.ndarray:
    """D. 4‑segment temporal statistics — 16 features."""
    T = signal.shape[0]
    seg_len = max(1, T // _NUM_SEGMENTS)
    feats: list[float] = []

    for s in range(_NUM_SEGMENTS):
        start = s * seg_len
        end = start + seg_len if s < _NUM_SEGMENTS - 1 else T
        segment = signal[start:end]

        if segment.shape[0] == 0:
            feats.extend([0.0, 0.0, 0.0, 0.0])
        else:
            feats.extend(
                [
                    float(np.mean(segment)),
                    float(np.std(segment, ddof=0)),
                    float(np.sqrt(np.mean(segment**2))),
                    float(np.max(segment)),
                ]
            )

    return np.array(feats, dtype=np.float32)


def extract_signal_features(
    signal: np.ndarray,
    *,
    envelope_window: int = 11,
) -> np.ndarray:
    """Extract the full 36‑dim feature vector for a single 1‑D signal.

    Args:
        signal: 1‑D float array of shape ``[T]``.
        envelope_window: Window size for moving‑average envelope.

    Returns:
        Float32 array of shape ``[36]``.
    """
    g = _compute_global_stats(signal)
    d = _compute_diff_stats(signal)
    e = _compute_envelope_features(signal, window_size=envelope_window)
    s = _compute_segmented_stats(signal)
    return np.concatenate([g, d, e, s]).astype(np.float32)


# ---------------------------------------------------------------------------
# Multi‑channel extraction (public API)
# ---------------------------------------------------------------------------


def extract_imu_temporal_envelope_features(
    x: np.ndarray,
    *,
    envelope_window: int = 11,
) -> np.ndarray:
    """Extract temporal-envelope features from an IMU window.

    Steps:
        1. Compute available magnitude signals from acc, gyro, and mag.
        2. For each signal, extract 36 features (global + diff + envelope
           + segmented).
        3. Concatenate into a flat vector.

    Args:
        x: Float32 array of shape ``[C, T]``. The standard channel order is
            acc.xyz, gyro.xyz, then mag.xyz. C=6 keeps acc+gyro only; C=9
            keeps acc+gyro+mag.
        envelope_window: Moving‑average window size for envelope.

    Returns:
        Float32 feature vector. C=9 gives 432 dims; C=6 gives 288 dims.
    """
    C = x.shape[0]
    if C < 3:
        raise ValueError(f"Expected at least 3 IMU channels, got {C}")

    signals: list[np.ndarray] = [x[c] for c in range(C)]
    acc_mag = np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
    signals.append(acc_mag)
    if C >= 6:
        gyro_mag = np.sqrt(x[3] ** 2 + x[4] ** 2 + x[5] ** 2)
        signals.append(gyro_mag)
    if C >= 9:
        mag_mag = np.sqrt(x[6] ** 2 + x[7] ** 2 + x[8] ** 2)
        signals.append(mag_mag)

    features: list[np.ndarray] = []
    for sig in signals:
        features.append(extract_signal_features(sig, envelope_window=envelope_window))

    result = np.concatenate(features).astype(np.float32)

    # Safety: replace NaN/inf with 0.
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result
