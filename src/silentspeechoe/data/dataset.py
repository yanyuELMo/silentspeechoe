"""PyTorch Dataset for OpenEarable bone‑acceleration utterance windows.

Supports single‑ear and binaural (left + right) configurations.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .labels import parse_all_labels
from .sensor_io import find_bone_acc_path, load_bone_acc, slice_bone_acc_window

logger = logging.getLogger(__name__)

# Validation subjects for subject‑holdout split.
_VAL_SUBJECTS = frozenset({"07", "10", "13", "17"})


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


class BoneBinauralDataset(Dataset):
    """Torch Dataset for binaural bone‑acceleration utterance classification.

    Each item is a dict::

        {
            "x":           FloatTensor [2, T],   # left & right norm
            "y":           int (0‑35),
            "speech_mode": str,
            "subject_id":  str,
        }

    The two channels are:
    0. left  ``sqrt(bone_acc.x² + bone_acc.y² + bone_acc.z²)``
    1. right same computation.
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

        # Load (or fetch from cache) the bone_acc arrays for both ears.
        left_arr = self._load_window(
            rec["left_path"], rec["left_start_sec"], rec["left_end_sec"]
        )
        right_arr = self._load_window(
            rec["right_path"], rec["right_start_sec"], rec["right_end_sec"]
        )

        # Compute norm for each ear: [T, 3] → [T]
        left_norm = _compute_norm(left_arr)
        right_norm = _compute_norm(right_arr)

        # Pad the shorter channel so lengths match before stacking.
        max_len = max(left_norm.shape[0], right_norm.shape[0])
        if left_norm.shape[0] < max_len:
            left_norm = np.pad(left_norm, (0, max_len - left_norm.shape[0]))
        if right_norm.shape[0] < max_len:
            right_norm = np.pad(right_norm, (0, max_len - right_norm.shape[0]))

        # Stack → [2, T]
        x = torch.from_numpy(np.stack([left_norm, right_norm], axis=0))

        # Label: sentence_id is 1‑36 → 0‑35
        y = int(rec["sentence_id"]) - 1

        return {
            "x": x,
            "y": y,
            "speech_mode": rec["speech_mode"],
            "subject_id": rec["subject_id"],
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


def iter_batch_groups(dataset: BoneBinauralDataset) -> Iterator[list[str]]:
    """Yield the ``speech_mode`` for every sample in order.

    Convenience helper so a trainer can pass ``groups`` directly to
    :func:`silentspeechoe.evaluation.metrics.compute_grouped_classification_metrics`.
    """
    for i in range(len(dataset)):
        yield dataset[i]["speech_mode"]
