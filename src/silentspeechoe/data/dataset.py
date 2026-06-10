"""PyTorch Dataset for OpenEarable bone‑acceleration utterance windows.

Supports single‑ear and binaural (left + right) configurations.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .labels import EVENT_FIELDS, parse_all_labels
from .preprocessing import stack_binaural_bone_acc
from .sensor_io import find_bone_acc_path, load_bone_acc, slice_bone_acc_window

logger = logging.getLogger(__name__)

# Validation subjects for subject‑holdout split.
_VAL_SUBJECTS = frozenset({"07", "10", "13", "17"})

# Columns used to pair left and right events from events.csv.
_PAIRING_KEY = [
    "subject_id",
    "event_id",
    "sentence_id",
    "label_id",
    "domain",
    "repeat_id",
]


def _subject_has_raw(subject_id: str, side: str, subset: str, base_dir: Path) -> bool:
    """Check whether raw bone‑acc data exists for a subject / side / subset."""
    return find_bone_acc_path(subject_id, side, subset, base_dir=base_dir) is not None


def _compute_norm(xyz: np.ndarray) -> np.ndarray:
    """Compute per‑sample vector magnitude.

    Args:
        xyz: shape ``[T, 3]``.

    Returns:
        shape ``[T]``, ``sqrt(x² + y² + z²)``.
    """
    return np.sqrt(np.sum(xyz.astype(np.float64) ** 2, axis=1)).astype(np.float32)


# ---------------------------------------------------------------------------
# Legacy record builder (Excel labels → paired records)
# ---------------------------------------------------------------------------


def build_binaural_records(
    base_dir: str | Path = ".",
    val_subjects: frozenset[str] | set[str] = _VAL_SUBJECTS,
) -> tuple[list[dict], list[dict]]:
    """Parse labels and pair left + right windows for binaural samples.

    Only subjects that have *both* left and right raw bone‑acc data are
    included.  Windows where either side returns an empty slice are
    dropped.

    Args:
        base_dir: Project root.
        val_subjects: Subject IDs to hold out for validation.

    Returns:
        ``(train_records, val_records)`` — each record is a dict with
        keys ``subject_id``, ``sentence_id``, ``speech_mode``,
        ``repeat_id``, ``subset``, ``left_start_sec``, ``left_end_sec``,
        ``right_start_sec``, ``right_end_sec``, ``left_path``,
        ``right_path``.
    """
    base = Path(base_dir)
    all_labels = parse_all_labels(base)

    # ---- index labels by subject/event so repeated utterances stay aligned.
    left_index: dict[tuple, dict] = {}
    right_index: dict[tuple, dict] = {}

    for rec in all_labels:
        key = (
            rec["subject_id"],
            rec["event_id"],
        )
        if rec["side"] == "left":
            left_index[key] = rec
        else:
            right_index[key] = rec

    # ---- pair left + right ------------------------------------------------
    paired: list[dict] = []
    common_keys = set(left_index) & set(right_index)

    for key in sorted(common_keys):
        left_rec = left_index[key]
        right_rec = right_index[key]
        subject_id = left_rec["subject_id"]
        subset = left_rec["subset"]

        # Check raw availability for both sides
        if not _subject_has_raw(subject_id, "left", subset, base):
            logger.debug("Missing left raw data for subject %s, skipping", subject_id)
            continue
        if not _subject_has_raw(subject_id, "right", subset, base):
            logger.debug("Missing right raw data for subject %s, skipping", subject_id)
            continue

        left_path = find_bone_acc_path(subject_id, "left", subset, base_dir=base)
        right_path = find_bone_acc_path(subject_id, "right", subset, base_dir=base)

        assert left_path is not None and right_path is not None

        paired.append(
            {
                "subject_id": subject_id,
                "event_id": left_rec["event_id"],
                "left_session_id": left_rec["session_id"],
                "right_session_id": right_rec["session_id"],
                "sentence_id": left_rec["sentence_id"],
                "speech_mode": left_rec["speech_mode"],
                "repeat_id": left_rec["repeat_id"],
                "subset": subset,
                "left_start_sec": left_rec["start_sec"],
                "left_end_sec": left_rec["end_sec"],
                "right_start_sec": right_rec["start_sec"],
                "right_end_sec": right_rec["end_sec"],
                "left_path": left_path,
                "right_path": right_path,
            }
        )

    # ---- train / val split ------------------------------------------------
    train_recs = [r for r in paired if r["subject_id"] not in val_subjects]
    val_recs = [r for r in paired if r["subject_id"] in val_subjects]

    logger.info(
        "Binaural records: %d total (%d train / %d val)",
        len(paired),
        len(train_recs),
        len(val_recs),
    )
    return train_recs, val_recs


# ---------------------------------------------------------------------------
# Event‑CSV record builder (events.csv → paired records)
# ---------------------------------------------------------------------------


def _raw_subject_id(subject_id: str) -> str:
    """Convert ``sub_00`` → ``00``."""
    return str(subject_id).removeprefix("sub_")


def _raw_subset(sentence_type: str) -> str:
    """Map event sentence_type to raw‑data subset folder."""
    if sentence_type == "non_semantic":
        return "non-semantic"
    if sentence_type == "semantic":
        return "semantic"
    raise ValueError(f"Unknown sentence_type: {sentence_type!r}")


def build_binaural_event_records(
    events_path: str | Path = "data/metadata/events.csv",
    raw_dir: str | Path = "data/raw",
) -> list[dict]:
    """Build binaural records by pairing left/right rows from ``events.csv``.

    Args:
        events_path: Path to the event CSV (columns match
            :data:`silentspeechoe.data.labels.EVENT_FIELDS`).
        raw_dir: Path to the raw data directory (typically
            ``data/raw``).

    Returns:
        List of paired record dicts.  Each dict has:

        * ``subject_id``
        * ``event_id``
        * ``sentence_id`` (e.g. ``"nonsem_001"``)
        * ``label_id`` (0‑35)
        * ``domain`` (normal / whisper / silent)
        * ``repeat_id`` (1 or 2)
        * ``sentence_type`` (non_semantic / semantic)
        * ``left_path``, ``right_path`` — ``Path`` to the bone_acc CSV
        * ``left_start_time``, ``left_end_time``
        * ``right_start_time``, ``right_end_time``

        Pairs are skipped when either raw file is missing.
    """
    events_path = Path(events_path)
    raw_dir = Path(raw_dir)

    df = pd.read_csv(events_path)
    _validate_event_columns(df)

    left_df = df[df["ear"] == "left"].copy()
    right_df = df[df["ear"] == "right"].copy()

    merged = pd.merge(
        left_df,
        right_df,
        on=_PAIRING_KEY,
        how="inner",
        suffixes=("_left", "_right"),
    )

    records: list[dict] = []

    for _, row in merged.iterrows():
        subject_id = str(row["subject_id"])
        raw_subj = _raw_subject_id(subject_id)
        sentence_type = str(row["sentence_type_left"])
        subset = _raw_subset(sentence_type)

        left_path = find_bone_acc_path(
            raw_subj, "left", subset, base_dir=raw_dir, raw_root=""
        )
        right_path = find_bone_acc_path(
            raw_subj, "right", subset, base_dir=raw_dir, raw_root=""
        )

        if left_path is None:
            logger.debug(
                "Skipping pair for subject=%s event=%s: missing left raw",
                subject_id,
                row["event_id"],
            )
            continue
        if right_path is None:
            logger.debug(
                "Skipping pair for subject=%s event=%s: missing right raw",
                subject_id,
                row["event_id"],
            )
            continue

        records.append(
            {
                "subject_id": subject_id,
                "event_id": int(row["event_id"]),
                "sentence_id": str(row["sentence_id"]),
                "label_id": int(row["label_id"]),
                "domain": str(row["domain"]),
                "repeat_id": int(row["repeat_id"]),
                "sentence_type": sentence_type,
                "left_path": left_path,
                "right_path": right_path,
                "left_start_time": float(row["start_time_left"]),
                "left_end_time": float(row["end_time_left"]),
                "right_start_time": float(row["start_time_right"]),
                "right_end_time": float(row["end_time_right"]),
            }
        )

    logger.info("Built %d binaural event records from %s", len(records), events_path)
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


class BoneBinauralDataset(Dataset):
    """Torch Dataset for binaural bone‑acceleration utterance classification.

    Each item is a dict::

        {
            "x":             FloatTensor [2, T],   # left & right preprocessed
            "y":             int (0‑35),
            "domain":        str,
            "subject_id":    str,
            "event_id":      int,
            "sentence_id":   str,
            "repeat_id":     int,
            "length":        int,
            "left_length":   int,
            "right_length":  int,
        }

    The two channels are:
    0. left  preprocessed bone‑acc magnitude (centered → magnitude → z‑score)
    1. right same computation.

    Preprocessing is applied per‑sample in ``__getitem__``; no resampling
    or fixed‑length coercion happens here.
    """

    def __init__(
        self,
        records: list[dict],
        padding_sec: float = 0.0,
        base_dir: str | Path = ".",
    ):
        self.records = records
        self.padding_sec = padding_sec
        self.base_dir = Path(base_dir)

        # Cache for loaded dataframes — keyed by path
        self._df_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # Load and slice raw windows.
        left_xyz = self._load_window(
            rec["left_path"], rec["left_start_time"], rec["left_end_time"]
        )
        right_xyz = self._load_window(
            rec["right_path"], rec["right_start_time"], rec["right_end_time"]
        )

        # Preprocess each ear and stack.
        x, meta = stack_binaural_bone_acc(left_xyz, right_xyz)

        return {
            "x": torch.from_numpy(x),
            "y": int(rec["label_id"]),
            "domain": rec["domain"],
            "subject_id": rec["subject_id"],
            "event_id": int(rec["event_id"]),
            "sentence_id": str(rec["sentence_id"]),
            "repeat_id": int(rec["repeat_id"]),
            "length": meta["length"],
            "left_length": meta["left_length"],
            "right_length": meta["right_length"],
        }

    def _load_window(self, path: Path, start_sec: float, end_sec: float) -> np.ndarray:
        """Load a bone‑acc window, using an in‑memory cache for the CSV.

        Returns ``[T, 3]`` float32 array.
        """
        cache_key = str(path)
        if cache_key not in self._df_cache:
            self._df_cache[cache_key] = load_bone_acc(path)
        df = self._df_cache[cache_key]
        return slice_bone_acc_window(
            df, start_sec, end_sec, padding_sec=self.padding_sec
        )


def iter_batch_groups(dataset: BoneBinauralDataset) -> Iterator[str]:
    """Yield the ``domain`` for every sample in order.

    Convenience helper so a trainer can pass ``groups`` directly to
    :func:`silentspeechoe.evaluation.metrics.compute_grouped_classification_metrics`.
    """
    for i in range(len(dataset)):
        yield dataset[i]["domain"]
