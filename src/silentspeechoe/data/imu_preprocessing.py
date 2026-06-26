"""IMU sensor preprocessing for OpenEarable 2.0 data.

The pipeline reads raw IMU CSV files, slices utterance windows according to
``events.csv``, resamples to a configurable target rate (default 200 Hz), and
returns clean ``[9, T]`` float32 tensors.

IMU channels (9 total)::

    acc.x  acc.y  acc.z
    gyro.x gyro.y gyro.z
    mag.x  mag.y  mag.z

Side selection is configurable via ``sides``: ``left``, ``right``, or both.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .imu_augmentation import IMUWindowAugmenter
from .labels import EVENT_FIELDS
from .subject_filtering import filter_subject_dataframe, filter_subject_records

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMU_CHANNELS: list[str] = [
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

_NUM_IMU_CHANNELS: int = len(IMU_CHANNELS)  # 9

_IMU_REQUIRED_COLS = {"timestamp"} | set(IMU_CHANNELS)
_MAD_TO_SIGMA = 1.4826
_DEFAULT_EPS = 1e-6

# Columns used to pair left and right event rows (same as bone-acc pairing).
_PAIRING_KEY: list[str] = [
    "subject_id",
    "event_id",
    "sentence_id",
    "label_id",
    "domain",
    "repeat_id",
]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _raw_subject_id(subject_id: str) -> str:
    """Convert ``sub_00`` → ``00``."""
    return str(subject_id).removeprefix("sub_")


def _raw_subset(sentence_type: str) -> str:
    """Map event ``sentence_type`` to raw-data subset folder."""
    if sentence_type == "non_semantic":
        return "non-semantic"
    if sentence_type == "semantic":
        return "semantic"
    raise ValueError(f"Unknown sentence_type: {sentence_type!r}")


def _imu_path_from_event(
    raw_dir: Path,
    *,
    ear: str,
    subject_id: str,
    sentence_type: str,
    session_id: str,
) -> Path:
    """Build the exact raw IMU path for one event row.

    Example result::

        data/raw/left/00/semantic/sensor_003_2718698242__imu.csv
    """
    return (
        raw_dir
        / ear
        / _raw_subject_id(subject_id)
        / _raw_subset(sentence_type)
        / f"sensor_{session_id}__imu.csv"
    )


def find_imu_path(
    subject_id: str,
    side: str,
    subset: str,
    base_dir: str | Path = ".",
    *,
    raw_root: str = "data/raw",
) -> Path | None:
    """Locate the IMU CSV for a given subject / side / subset.

    Args:
        subject_id: e.g. ``"00"``.
        side: ``"left"`` or ``"right"``.
        subset: ``"non-semantic"`` or ``"semantic"``.
        base_dir: Project root (or parent of *raw_root*).
        raw_root: Sub-path under *base_dir* where raw data lives.

    Returns:
        Path to the CSV, or ``None`` if not found.
    """
    raw_root_path = Path(raw_root) if raw_root else Path()
    pattern = Path(base_dir) / raw_root_path / side / subject_id / subset
    if not pattern.exists():
        logger.debug("Directory missing: %s", pattern)
        return None
    candidates = sorted(pattern.glob("*__imu.csv"))
    if not candidates:
        logger.debug("No IMU CSV in %s", pattern)
        return None
    if len(candidates) > 1:
        logger.warning("Multiple IMU CSVs in %s — using %s", pattern, candidates[0])
    return candidates[0]


# ---------------------------------------------------------------------------
# Low-level I/O
# ---------------------------------------------------------------------------


def load_imu(path: str | Path) -> pd.DataFrame:
    """Read an IMU CSV into a DataFrame.

    Expected columns: ``timestamp``, ``acc.x``, ``acc.y``, ``acc.z``,
    ``gyro.x``, ``gyro.y``, ``gyro.z``, ``mag.x``, ``mag.y``, ``mag.z``.

    Args:
        path: Path to the IMU CSV file.

    Returns:
        DataFrame with all expected columns.

    Raises:
        ValueError: If required columns are missing.
    """
    df = pd.read_csv(path)
    missing = _IMU_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def slice_imu_window(
    df: pd.DataFrame,
    start_sec: float,
    end_sec: float,
    *,
    padding_sec: float = 0.0,
) -> np.ndarray:
    """Extract a time window from an IMU DataFrame.

    Args:
        df: DataFrame loaded via :func:`load_imu`.
        start_sec: Window start in seconds.
        end_sec: Window end in seconds.
        padding_sec: Optional padding added *before* and *after*
            the window (default ``0.0``).

    Returns:
        Float32 array of shape ``[time, 9]`` (IMU channels in order).
        Returns an empty array ``(0, 9)`` if no samples fall inside
        the padded window.
    """
    t0 = start_sec - padding_sec
    t1 = end_sec + padding_sec

    mask = (df["timestamp"] >= t0) & (df["timestamp"] <= t1)
    values = df.loc[mask, IMU_CHANNELS].to_numpy(dtype=np.float32)

    if values.shape[0] == 0:
        logger.debug(
            "Empty IMU window [%.3f, %.3f] (padded [%.3f, %.3f])",
            start_sec,
            end_sec,
            t0,
            t1,
        )
    return values


def slice_imu_window_with_time(
    df: pd.DataFrame,
    start_sec: float,
    end_sec: float,
    *,
    padding_sec: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a time window with timestamps from an IMU DataFrame.

    Args:
        df: DataFrame loaded via :func:`load_imu`.
        start_sec: Window start in seconds.
        end_sec: Window end in seconds.
        padding_sec: Optional padding (default ``0.0``).

    Returns:
        ``(timestamps, values)`` where *timestamps* is ``float64`` of
        shape ``[T]`` and *values* is ``float32`` of shape ``[T, 9]``.
        Both are empty if no samples fall inside the padded window.
    """
    t0 = start_sec - padding_sec
    t1 = end_sec + padding_sec

    mask = (df["timestamp"] >= t0) & (df["timestamp"] <= t1)
    ts = df.loc[mask, "timestamp"].to_numpy(dtype=np.float64)
    values = df.loc[mask, IMU_CHANNELS].to_numpy(dtype=np.float32)

    return ts, values


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------


def clean_imu_timestamps(
    timestamps: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean IMU data by removing NaN/inf and deduplicating timestamps.

    Steps:

    1. Drop rows where any IMU channel is NaN or inf.
    2. Sort by timestamp.
    3. For duplicate timestamps keep the *first* occurrence.
    4. Ensure timestamps are strictly monotonic (required for interpolation).

    Args:
        timestamps: Float array of shape ``[T]``.
        values: Float32 array of shape ``[T, C]``.

    Returns:
        ``(timestamps_clean, values_clean)`` — both arrays may be shorter
        than the inputs.
    """
    if timestamps.shape[0] == 0:
        return timestamps, values

    # 1. Drop NaN / inf rows.
    finite_mask = np.isfinite(values).all(axis=1) & np.isfinite(timestamps)
    ts = timestamps[finite_mask]
    vals = values[finite_mask]

    if ts.shape[0] == 0:
        return ts, vals

    # 2. Sort by timestamp.
    sort_idx = np.argsort(ts)
    ts = ts[sort_idx]
    vals = vals[sort_idx]

    # 3. Deduplicate timestamps — keep first occurrence.
    _, unique_idx = np.unique(ts, return_index=True)
    ts = ts[unique_idx]
    vals = vals[unique_idx]

    # 4. Ensure strictly monotonic (remove any remaining non-increasing steps).
    mono_mask = np.diff(ts) > 0.0
    mono_mask = np.concatenate([[True], mono_mask])
    ts = ts[mono_mask]
    vals = vals[mono_mask]

    return ts, vals


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def resample_imu_window(
    timestamps: np.ndarray,
    values: np.ndarray,
    start_time: float,
    end_time: float,
    target_sample_rate: float = 200.0,
) -> np.ndarray:
    """Resample IMU data to a regular time grid via per-channel linear
    interpolation.

    Args:
        timestamps: Original timestamps in seconds, shape ``[T]``.
        values: Original IMU values, shape ``[T, C]`` (C=9).
        start_time: Start of the regular grid in seconds.
        end_time: End of the regular grid in seconds.
        target_sample_rate: Output sample rate in Hz (default 200).

    Returns:
        Float32 array of shape ``[C, T_out]`` where
        ``T_out = int((end_time - start_time) * target_sample_rate)``.
        Returns ``[C, 0]`` if the window duration is non-positive or no
        valid samples are available.
    """
    duration = end_time - start_time
    if duration <= 0:
        logger.debug("Non-positive window duration: %.3f", duration)
        return np.empty((values.shape[1], 0), dtype=np.float32)

    num_out = max(1, int(round(duration * target_sample_rate)))
    target_t = np.linspace(start_time, end_time, num_out, dtype=np.float64)

    if timestamps.shape[0] < 2:
        # Not enough points to interpolate — repeat the single value or zeros.
        if timestamps.shape[0] == 1:
            out = np.tile(values[0:1, :].T, (1, num_out)).astype(np.float32)
        else:
            out = np.zeros((values.shape[1], num_out), dtype=np.float32)
        return out

    # Per-channel linear interpolation.
    C = values.shape[1]
    out = np.empty((C, num_out), dtype=np.float32)
    for c in range(C):
        out[c] = np.interp(target_t, timestamps, values[:, c]).astype(np.float32)

    return out


# ---------------------------------------------------------------------------
# Post-resampling signal conditioning
# ---------------------------------------------------------------------------


def median_mad_despike_imu_window(
    x: np.ndarray,
    *,
    threshold: float = 8.0,
    eps: float = _DEFAULT_EPS,
) -> np.ndarray:
    """Clip strong IMU spikes using a per-window Median/MAD rule.

    The operation is intentionally mild: values are clipped to
    ``median +/- threshold * 1.4826 * MAD`` per channel. Typical non-spike
    samples are left unchanged, and flat channels are passed through.
    """
    _validate_preprocessed_window(x)
    if x.shape[1] == 0 or threshold <= 0.0:
        return x.astype(np.float32, copy=True)

    median = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - median), axis=1, keepdims=True)
    robust_std = _MAD_TO_SIGMA * mad

    out = x.astype(np.float32, copy=True)
    valid = (robust_std > eps).reshape(-1)
    if not np.any(valid):
        return out

    lower = median - threshold * robust_std
    upper = median + threshold * robust_std
    out[valid] = np.clip(out[valid], lower[valid], upper[valid])
    return out.astype(np.float32, copy=False)


def remove_imu_dc(x: np.ndarray) -> np.ndarray:
    """Remove per-window, per-channel DC offsets from an IMU window."""
    _validate_preprocessed_window(x)
    if x.shape[1] == 0:
        return x.astype(np.float32, copy=True)
    mean = x.mean(axis=1, keepdims=True)
    return (x - mean).astype(np.float32)


def zscore_imu_window(x: np.ndarray, *, eps: float = _DEFAULT_EPS) -> np.ndarray:
    """Apply per-window, per-channel z-score normalization."""
    _validate_preprocessed_window(x)
    if x.shape[1] == 0:
        return x.astype(np.float32, copy=True)
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std <= eps, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def condition_resampled_imu_window(
    x: np.ndarray,
    *,
    despike: bool = True,
    despike_threshold: float = 8.0,
    remove_dc: bool = True,
    normalize: bool = True,
    eps: float = _DEFAULT_EPS,
) -> np.ndarray:
    """Apply the standard post-resampling IMU conditioning chain."""
    _validate_preprocessed_window(x)
    out = x.astype(np.float32, copy=True)
    if despike:
        out = median_mad_despike_imu_window(
            out,
            threshold=despike_threshold,
            eps=eps,
        )
    if remove_dc:
        out = remove_imu_dc(out)
    if normalize:
        out = zscore_imu_window(out, eps=eps)
    return out.astype(np.float32, copy=False)


def _validate_preprocessed_window(x: np.ndarray) -> None:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(x)!r}")
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape [C, T], got {x.shape}")


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_imu_window(
    path: str | Path,
    start_sec: float,
    end_sec: float,
    *,
    target_sample_rate: float = 200.0,
    padding_sec: float = 0.0,
    normalize: bool = False,
    remove_dc: bool = False,
    despike: bool = False,
    despike_threshold: float = 8.0,
) -> tuple[np.ndarray, dict]:
    """Run the full IMU preprocessing pipeline for one utterance window.

    Steps:

    1. Load the IMU CSV.
    2. Slice the utterance window ``[start_sec, end_sec]`` with optional
       padding.
    3. Clean NaN/inf values and deduplicate timestamps.
    4. Resample to a regular ``target_sample_rate`` Hz grid.
    5. Optionally apply mild Median/MAD despiking, DC removal, and per-channel
       z-score normalization.

    Args:
        path: Path to the IMU CSV file.
        start_sec: Window start in seconds.
        end_sec: Window end in seconds.
        target_sample_rate: Output sample rate in Hz.
        padding_sec: Optional padding around the window.
        normalize: If ``True``, apply per-channel z-score normalization.
        remove_dc: If ``True``, subtract each channel's window mean.
        despike: If ``True``, clip extreme channel values with Median/MAD.
        despike_threshold: Robust-sigma clipping threshold for despiking.

    Returns:
        ``(x, meta)`` where:

        * ``x`` — ``float32`` array of shape ``[9, T]``.
        * ``meta`` — dict with keys ``length`` (int), ``num_finite`` (int,
          number of raw samples before resampling).
    """
    df = load_imu(path)

    # Slice window with timestamps.
    timestamps, values = slice_imu_window_with_time(
        df, start_sec, end_sec, padding_sec=padding_sec
    )

    num_raw = timestamps.shape[0]

    if num_raw == 0:
        # Empty window — return zero-length tensor.
        x = np.empty((_NUM_IMU_CHANNELS, 0), dtype=np.float32)
        return x, {"length": 0, "num_finite": 0}

    # Clean.
    timestamps, values = clean_imu_timestamps(timestamps, values)

    # Resample.
    t_start = start_sec - padding_sec
    t_end = end_sec + padding_sec
    x = resample_imu_window(
        timestamps,
        values,
        start_time=t_start,
        end_time=t_end,
        target_sample_rate=target_sample_rate,
    )

    x = condition_resampled_imu_window(
        x,
        despike=despike,
        despike_threshold=despike_threshold,
        remove_dc=remove_dc,
        normalize=normalize,
    )

    meta = {
        "length": int(x.shape[1]),
        "num_finite": num_raw,
        "despike": bool(despike),
        "despike_threshold": float(despike_threshold),
        "remove_dc": bool(remove_dc),
        "normalize": bool(normalize),
    }
    return x, meta


def _per_channel_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply per-channel z-score normalization.

    Args:
        x: Float32 array of shape ``[C, T]``.
        eps: Small constant to avoid division by zero.

    Returns:
        Normalized ``float32`` array of shape ``[C, T]``.
    """
    return zscore_imu_window(x, eps=eps)


# ---------------------------------------------------------------------------
# Record builder (events.csv → IMU records)
# ---------------------------------------------------------------------------


def build_imu_records(
    events_path: str | Path = "data/metadata/events.csv",
    raw_dir: str | Path = "data/raw",
    *,
    sides: list[str] | None = None,
) -> list[dict]:
    """Build IMU records from ``events.csv`` for the requested ear side(s).

    Each returned record represents one IMU utterance window.  When
    ``sides`` includes both ``"left"`` and ``"right"``, left and right
    events for the same utterance are returned as separate records (they
    can be fused later by a dataset or model).

    Args:
        events_path: Path to ``events.csv``.
        raw_dir: Path to the raw data directory (``data/raw``).
        sides: Ear sides to include — ``["left"]``, ``["right"]``, or
            ``["left", "right"]``.  Defaults to ``["left"]``.

    Returns:
        List of record dicts, each with keys:

        * ``subject_id`` — e.g. ``"sub_00"``
        * ``session_id`` — recording session ID string
        * ``event_id`` — slot index (int)
        * ``sentence_id`` — e.g. ``"nonsem_001"``
        * ``label_id`` — 0‑based class label (int, 0‑35)
        * ``domain`` — ``"normal"``, ``"whisper"``, or ``"silent"``
        * ``repeat_id`` — 1 or 2
        * ``sentence_type`` — ``"non_semantic"`` or ``"semantic"``
        * ``side`` — ``"left"`` or ``"right"``
        * ``path`` — ``Path`` to the raw IMU CSV
        * ``start_time`` — window start in seconds (float)
        * ``end_time`` — window end in seconds (float)

        Records are skipped when the raw IMU file is missing.
    """
    if sides is None:
        sides = ["left"]

    events_path = Path(events_path)
    raw_dir = Path(raw_dir)
    df = pd.read_csv(events_path)
    _validate_event_columns(df)
    df = filter_subject_dataframe(df)

    records: list[dict] = []

    for side in sides:
        side_df = df[df["ear"] == side]
        for _, row in side_df.iterrows():
            subject_id = str(row["subject_id"])
            sentence_type = str(row["sentence_type"])
            session_id = str(row["session_id"])

            imu_path = _imu_path_from_event(
                raw_dir,
                ear=side,
                subject_id=subject_id,
                sentence_type=sentence_type,
                session_id=session_id,
            )

            if not imu_path.exists():
                logger.debug(
                    "Skipping IMU record: subject=%s side=%s event=%s — "
                    "file missing: %s",
                    subject_id,
                    side,
                    row["event_id"],
                    imu_path,
                )
                continue

            records.append(
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "event_id": int(row["event_id"]),
                    "sentence_id": str(row["sentence_id"]),
                    "label_id": int(row["label_id"]),
                    "domain": str(row["domain"]),
                    "repeat_id": int(row["repeat_id"]),
                    "sentence_type": sentence_type,
                    "side": side,
                    "path": imu_path,
                    "start_time": float(row["start_time"]),
                    "end_time": float(row["end_time"]),
                }
            )

    logger.info(
        "Built %d IMU records (sides=%s) from %s",
        len(records),
        sides,
        events_path,
    )
    return records


def _validate_event_columns(df: pd.DataFrame) -> None:
    """Check that *df* has the required ``events.csv`` columns."""
    required = set(EVENT_FIELDS)
    actual = set(df.columns)
    missing = required - actual
    if missing:
        raise ValueError(f"events.csv is missing required columns: {sorted(missing)}")


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class IMUDataset(Dataset):
    """Torch Dataset for single-side IMU utterance classification.

    Each item is a dict::

        {
            "x":          FloatTensor [9, T],
            "y":          int (0‑35),
            "length":     int,
            "domain":     str,
            "subject_id": str,
            "session_id": str,
            "sentence_id": str,
            "repeat_id":  int,
            "side":       str,
        }

    Preprocessing (slice → clean → resample) is applied per-sample in
    ``__getitem__``.  No fixed-length coercion happens here — use
    :func:`imu_pad_collate` to pad within a batch.

    If ``augmenter`` is provided, it is applied to the preprocessed
    ``[9, T]`` tensor before the sample is returned. The original tensor
    is kept in ``x_original`` so downstream code can inspect it.
    """

    def __init__(
        self,
        records: list[dict],
        *,
        target_sample_rate: float = 200.0,
        padding_sec: float = 0.0,
        normalize: bool = False,
        augmenter: IMUWindowAugmenter | None = None,
    ):
        self.records = filter_subject_records(records)
        self.target_sample_rate = target_sample_rate
        self.padding_sec = padding_sec
        self.normalize = normalize
        self.augmenter = augmenter

        # In-memory cache for loaded DataFrames, keyed by path.
        self._df_cache: dict[str, pd.DataFrame] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # Load DataFrame (cached).
        cache_key = str(rec["path"])
        if cache_key not in self._df_cache:
            self._df_cache[cache_key] = load_imu(rec["path"])
        df = self._df_cache[cache_key]

        # Slice window with timestamps.
        timestamps, values = slice_imu_window_with_time(
            df,
            rec["start_time"],
            rec["end_time"],
            padding_sec=self.padding_sec,
        )

        if timestamps.shape[0] == 0:
            x = np.empty((_NUM_IMU_CHANNELS, 0), dtype=np.float32)
            length = 0
        else:
            timestamps, values = clean_imu_timestamps(timestamps, values)
            t_start = rec["start_time"] - self.padding_sec
            t_end = rec["end_time"] + self.padding_sec
            x = resample_imu_window(
                timestamps,
                values,
                start_time=t_start,
                end_time=t_end,
                target_sample_rate=self.target_sample_rate,
            )
            if self.normalize:
                x = _per_channel_zscore(x)

        x_tensor = torch.from_numpy(x)
        x_original = x_tensor.clone() if self.augmenter is not None else None
        if self.augmenter is not None:
            x_tensor = self.augmenter(x_tensor)

        length = int(x_tensor.shape[1])

        item = {
            "x": x_tensor,
            "y": int(rec["label_id"]),
            "length": length,
            "domain": rec["domain"],
            "subject_id": rec["subject_id"],
            "session_id": rec["session_id"],
            "sentence_id": str(rec["sentence_id"]),
            "repeat_id": int(rec["repeat_id"]),
            "side": rec["side"],
        }
        if x_original is not None:
            item["x_original"] = x_original
        return item


# ---------------------------------------------------------------------------
# Batch collation
# ---------------------------------------------------------------------------


def imu_pad_collate(batch: list[dict]) -> dict:
    """Collate a list of IMU dataset items into a padded batch.

    Each dataset item is expected to be a dict with at least::

        {
            "x":          FloatTensor [C, T_i],
            "y":          int,
            "length":     int,
            "domain":     str,
            "subject_id": str,
            "session_id": str,
            "sentence_id": str,
            "repeat_id":  int,
            "side":       str,
        }

    Returns a dict with:

    * ``x`` — ``FloatTensor [B, C, max_T]`` (zero-padded)
    * ``y`` — ``LongTensor [B]``
    * ``lengths`` — ``LongTensor [B]``
    * ``domain`` — list of str
    * ``subject_id`` — list of str
    * ``session_id`` — list of str
    * ``sentence_id`` — list of str
    * ``repeat_id`` — list of int
    * ``side`` — list of str
    """
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    lengths: list[int] = []
    domains: list[str] = []
    subjects: list[str] = []
    sessions: list[str] = []
    sentence_ids: list[str] = []
    repeat_ids: list[int] = []
    sides: list[str] = []

    for item in batch:
        x = item["x"]
        if x.dim() != 2:
            raise ValueError(f"Expected x of shape [C, T], got {x.shape}")
        xs.append(x)
        ys.append(int(item["y"]))
        lengths.append(int(item.get("length", x.shape[1])))
        domains.append(item.get("domain", ""))
        subjects.append(item["subject_id"])
        sessions.append(item.get("session_id", ""))
        sentence_ids.append(str(item.get("sentence_id", "")))
        repeat_ids.append(int(item.get("repeat_id", -1)))
        sides.append(item.get("side", ""))

    max_len = max(lengths) if lengths else 0
    C = xs[0].shape[0] if xs else _NUM_IMU_CHANNELS

    padded = torch.zeros(len(xs), C, max_len, dtype=torch.float32)
    for i, x in enumerate(xs):
        T = x.shape[1]
        if T > 0:
            padded[i, :, :T] = x

    return {
        "x": padded,
        "y": torch.tensor(ys, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "domain": domains,
        "subject_id": subjects,
        "session_id": sessions,
        "sentence_id": sentence_ids,
        "repeat_id": repeat_ids,
        "side": sides,
    }


# ---------------------------------------------------------------------------
# Pre‑computed IMU Dataset
# ---------------------------------------------------------------------------


class PrecomputedIMUDataset(Dataset):
    """Torch Dataset that loads pre-computed IMU windows from ``.pt`` files.

    Assumes windows were pre‑computed by a preprocessing script.  Each
    ``.pt`` file contains a dict with keys ``x`` (``[9, T]`` tensor),
    ``y``, ``domain``, ``subject_id``, ``session_id``, ``sentence_id``,
    ``repeat_id``, ``side``, ``length``.

    ``__getitem__`` only does a ``torch.load`` call — near-zero CPU cost.

    If ``augmenter`` is provided, it is applied to the loaded ``[C, T]``
    tensor before the sample is returned. The original tensor is kept in
    ``x_original`` so downstream code can inspect it.

    Args:
        manifest_path: Path to the ``manifest.json`` file.
        features_dir: Directory containing the ``.pt`` files.
        channel_indices: Optional list of channel indices to select from
            the 9‑channel tensor.  Default ``None`` keeps all 9 channels.
            Use ``[0,1,2,3,4,5]`` for acc+gyro only (6 channels).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        features_dir: str | Path,
        *,
        channel_indices: list[int] | None = None,
        augmenter: IMUWindowAugmenter | None = None,
    ):
        import json

        manifest_path = Path(manifest_path)
        features_dir = Path(features_dir)

        with manifest_path.open("r") as f:
            manifest_data = json.load(f)

        self.features_dir = features_dir
        self.records: list[dict] = filter_subject_records(manifest_data["records"])
        self.channel_indices = channel_indices
        self.augmenter = augmenter

        # Store preprocessing params for reference.
        self.target_sample_rate = float(manifest_data.get("target_sample_rate", 200.0))
        self.sides = manifest_data.get("sides", ["left"])
        self.channels = manifest_data.get("channels", IMU_CHANNELS)

    @property
    def num_channels(self) -> int:
        """Effective number of channels after selection."""
        if self.channel_indices is not None:
            return len(self.channel_indices)
        return len(self.channels)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        file_path = self.features_dir / rec["file"]
        data = torch.load(file_path, weights_only=True)
        x = data["x"]  # [C, T]
        if self.channel_indices is not None:
            x = x[self.channel_indices, :]
        x_original = x.clone() if self.augmenter is not None else None
        if self.augmenter is not None:
            x = self.augmenter(x)
        item = {
            "x": x,  # [C', T] — already contiguous
            "y": int(data["y"]),
            "length": int(data.get("length", data["x"].shape[1])),
            "domain": data["domain"],
            "subject_id": data["subject_id"],
            "session_id": data.get("session_id", ""),
            "sentence_id": str(data.get("sentence_id", "")),
            "repeat_id": int(data.get("repeat_id", -1)),
            "side": data.get("side", data.get("ear", "")),
        }
        if x_original is not None:
            item["x_original"] = x_original
        return item


# ---------------------------------------------------------------------------
# MFCC collate (fixed‑length 1‑D vectors → stacked tensor)
# ---------------------------------------------------------------------------


def mfcc_collate(batch: list[dict]) -> dict:
    """Collate MFCC feature samples into a batch.

    Each sample has ``x`` of shape ``[D]`` (1‑D).  Stacks them into
    ``[B, D]`` and provides dummy ``lengths`` for trainer compatibility.
    """
    xs = torch.stack([item["x"] for item in batch])
    ys = torch.tensor([int(item["y"]) for item in batch], dtype=torch.long)
    return {
        "x": xs,
        "y": ys,
        "lengths": torch.full((len(batch),), xs.shape[1], dtype=torch.long),
        "domain": [item.get("domain", "") for item in batch],
        "subject_id": [item["subject_id"] for item in batch],
        "session_id": [item.get("session_id", "") for item in batch],
        "sentence_id": [str(item.get("sentence_id", "")) for item in batch],
        "repeat_id": [int(item.get("repeat_id", -1)) for item in batch],
        "side": [item.get("side", "") for item in batch],
    }


# ---------------------------------------------------------------------------
# MFCC Feature Dataset (pre‑computed fixed‑length vectors)
# ---------------------------------------------------------------------------


class MFCCFeatureDataset(Dataset):
    """Torch Dataset for pre‑computed MFCC feature vectors.

    Each ``.pt`` file contains a dict with ``x`` (``[D]`` fixed‑length
    vector), ``y``, ``domain``, ``subject_id``, etc.

    Args:
        manifest_path: Path to ``manifest.json``.
        features_dir: Directory containing ``.pt`` files.
        reshape_2d: If ``True``, reshape ``[D]`` → ``[C, D//C]`` for 1‑D
            CNN consumption (9 channels × 26 features by default).
            Uses ``imu_pad_collate`` as collate function.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        features_dir: str | Path,
        *,
        reshape_2d: bool = False,
    ):
        import json

        manifest_path = Path(manifest_path)
        features_dir = Path(features_dir)

        with manifest_path.open("r") as f:
            manifest_data = json.load(f)

        self.features_dir = features_dir
        self.records: list[dict] = filter_subject_records(manifest_data["records"])
        self.feature_dim = int(manifest_data.get("feature_dim", 234))
        self.num_channels = len(manifest_data.get("channels", IMU_CHANNELS))
        self.channels = manifest_data.get("channels", IMU_CHANNELS)
        self.reshape_2d = reshape_2d

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        file_path = self.features_dir / rec["file"]
        data = torch.load(file_path, weights_only=True)
        x = data["x"]  # [D]
        if self.reshape_2d:
            x = x.reshape(self.num_channels, -1)  # [C, D//C]
        return {
            "x": x,
            "y": int(data["y"]),
            "length": int(x.shape[-1]) if self.reshape_2d else int(x.shape[0]),
            "domain": data["domain"],
            "subject_id": data["subject_id"],
            "session_id": data.get("session_id", ""),
            "sentence_id": str(data.get("sentence_id", "")),
            "repeat_id": int(data.get("repeat_id", -1)),
            "side": data.get("side", ""),
        }
