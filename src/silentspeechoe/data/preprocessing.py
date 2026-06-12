"""Preprocessing primitives for bone‑acceleration sensor streams.

Each function operates on a single utterance window — the raw ``[T, 3]``
xyz bone‑acc array sliced from a CSV file.  The pipeline is:

1. **Center** each axis by subtracting its median (robust to outliers).
2. **Dynamic magnitude** — compute the instantaneous vector magnitude.
3. **Per‑window z‑score** — standardise each window independently.

All functions are pure NumPy.  No file I/O, no torch, no resampling.
"""

from __future__ import annotations

import numpy as np


def center_bone_acc_axes(
    xyz: np.ndarray,
    method: str = "median",
) -> np.ndarray:
    """Center each axis of a bone‑acceleration window.

    Args:
        xyz: Float array of shape ``[T, 3]`` (x, y, z columns).
        method: Centering method — ``"median"`` (default) or ``"mean"``.

    Returns:
        Centered ``float32`` array of shape ``[T, 3]``.

    Raises:
        ValueError: If the input is not 2‑D with exactly 3 columns.
    """
    _validate_xyz(xyz)

    arr = xyz.astype(np.float32, copy=False)

    if method == "median":
        offsets = np.median(arr, axis=0).astype(np.float32)
    elif method == "mean":
        offsets = np.mean(arr, axis=0, dtype=np.float64).astype(np.float32)
    else:
        raise ValueError(f"Unknown centering method: {method!r}")

    return arr - offsets


def compute_dynamic_magnitude(xyz: np.ndarray) -> np.ndarray:
    """Compute per‑sample vector magnitude from centered xyz data.

    Args:
        xyz: Float array of shape ``[T, 3]`` (x, y, z columns).

    Returns:
        ``float32`` array of shape ``[T]`` — ``sqrt(x² + y² + z²)``.

    Raises:
        ValueError: If the input is not 2‑D with exactly 3 columns.
    """
    _validate_xyz(xyz)

    sq = xyz.astype(np.float64) ** 2
    mag = np.sqrt(np.sum(sq, axis=1))
    return mag.astype(np.float32)


def zscore_signal(
    signal: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Apply per‑window z‑score standardisation.

    Args:
        signal: 1‑D float array of shape ``[T]``.
        eps: Small constant to avoid division by zero when the standard
            deviation is negligible.

    Returns:
        ``float32`` array of shape ``[T]`` with near‑zero mean and
        near‑unit standard deviation.  If the input is constant (std ≤
        *eps*) the output is all zeros.
    """
    if signal.ndim != 1:
        raise ValueError(f"Expected 1‑D signal, got shape {signal.shape}")

    arr = signal.astype(np.float32, copy=False)
    std = float(np.std(arr))

    if std <= eps:
        return np.zeros_like(arr, dtype=np.float32)

    mean = float(np.mean(arr))
    return (arr - mean) / np.float32(std)


def preprocess_bone_acc_window(xyz: np.ndarray) -> np.ndarray:
    """Run the full single‑ear preprocessing chain.

    Steps:

    1. Center axes (median subtraction).
    2. Compute dynamic magnitude.
    3. Per‑window z‑score standardisation.

    Args:
        xyz: Float array of shape ``[T, 3]``.

    Returns:
        ``float32`` array of shape ``[T]``.
    """
    centered = center_bone_acc_axes(xyz, method="median")
    magnitude = compute_dynamic_magnitude(centered)
    return zscore_signal(magnitude)


def stack_binaural_bone_acc(
    left_xyz: np.ndarray,
    right_xyz: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Preprocess left and right ear windows and stack into one sample.

    Each ear is preprocessed independently via
    :func:`preprocess_bone_acc_window`.  The shorter ear is zero‑padded
    to match the longer ear so the result is always rectangular.

    Args:
        left_xyz:  Raw left‑ear ``[T_l, 3]`` float array.
        right_xyz: Raw right‑ear ``[T_r, 3]`` float array.

    Returns:
        ``(x, meta)`` where:

        * ``x`` — ``float32`` array of shape ``[2, T_max]``
          (channel 0 = left, channel 1 = right).
        * ``meta`` — dict with keys ``left_length``, ``right_length``,
          ``length`` (all Python ``int``).
    """
    left_proc = preprocess_bone_acc_window(left_xyz)  # [T_l]
    right_proc = preprocess_bone_acc_window(right_xyz)  # [T_r]

    left_len = int(left_proc.shape[0])
    right_len = int(right_proc.shape[0])
    max_len = max(left_len, right_len)

    if left_len < max_len:
        left_proc = np.pad(left_proc, (0, max_len - left_len))
    if right_len < max_len:
        right_proc = np.pad(right_proc, (0, max_len - right_len))

    x = np.stack([left_proc, right_proc], axis=0).astype(np.float32)

    meta = {
        "left_length": left_len,
        "right_length": right_len,
        "length": max_len,
    }
    return x, meta


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_xyz(xyz: np.ndarray) -> None:
    """Raise ``ValueError`` if *xyz* is not ``[T, 3]``."""
    if xyz.ndim != 2:
        raise ValueError(
            f"Expected 2‑D array of shape [T, 3], got ndim={xyz.ndim} "
            f"with shape {xyz.shape}"
        )
    if xyz.shape[1] != 3:
        raise ValueError(f"Expected exactly 3 columns (x, y, z), got shape {xyz.shape}")
    if xyz.shape[0] == 0:
        raise ValueError("Input array must be non‑empty (T > 0)")
