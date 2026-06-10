"""Low‑level sensor-data I/O helpers.

Load raw OpenEarable CSV streams and slice utterance windows from them.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Column names in bone_acc CSV files
_BONE_COLS = ["bone_acc.x", "bone_acc.y", "bone_acc.z"]


def find_bone_acc_path(
    subject_id: str,
    side: str,
    subset: str,
    base_dir: str | Path = ".",
) -> Path | None:
    """Locate the bone‑acceleration CSV for a given subject / side / subset.

    Args:
        subject_id: e.g. ``"00"``.
        side: ``"left"`` or ``"right"``.
        subset: ``"non-semantic"`` or ``"semantic"``.
        base_dir: Project root.

    Returns:
        Path to the CSV, or ``None`` if not found.
    """
    pattern = Path(base_dir) / "data" / "raw" / side / subject_id / subset
    if not pattern.exists():
        logger.debug("Directory missing: %s", pattern)
        return None
    candidates = sorted(pattern.glob("*__bone_acc.csv"))
    if not candidates:
        logger.debug("No bone_acc CSV in %s", pattern)
        return None
    if len(candidates) > 1:
        logger.warning(
            "Multiple bone_acc CSVs in %s — using %s", pattern, candidates[0]
        )
    return candidates[0]


def load_bone_acc(path: str | Path) -> pd.DataFrame:
    """Read a bone‑acceleration CSV into a DataFrame.

    Expected columns: ``timestamp, bone_acc.x, bone_acc.y, bone_acc.z``.
    """
    df = pd.read_csv(path)
    required = {"timestamp", "bone_acc.x", "bone_acc.y", "bone_acc.z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df


def slice_bone_acc_window(
    df: pd.DataFrame,
    start_sec: float,
    end_sec: float,
    *,
    padding_sec: float = 0.0,
) -> np.ndarray:
    """Extract a time window from a bone‑acceleration DataFrame.

    Args:
        df: DataFrame loaded via :func:`load_bone_acc`.
        start_sec: Window start in seconds.
        end_sec: Window end in seconds.
        padding_sec: Optional padding added *before* and *after*
            the window (default ``0.0``).

    Returns:
        Float32 array of shape ``[time, 3]`` (``x, y, z`` columns).
        Returns an empty array ``(0, 3)`` if no samples fall inside
        the (padded) window.
    """
    t0 = start_sec - padding_sec
    t1 = end_sec + padding_sec

    mask = (df["timestamp"] >= t0) & (df["timestamp"] <= t1)
    values = df.loc[mask, _BONE_COLS].to_numpy(dtype=np.float32)

    if values.shape[0] == 0:
        logger.debug(
            "Empty window [%.3f, %.3f] (padded [%.3f, %.3f])",
            start_sec,
            end_sec,
            t0,
            t1,
        )
    return values
