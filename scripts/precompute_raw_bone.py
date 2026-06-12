"""Pre‑compute raw binaural bone‑acc windows and cache them as ``.pt`` files.

Run once before training the raw TCN baseline::

    python scripts/precompute_raw_bone.py

Output structure::

    data/processed/raw_bone_binaural/
    ├── manifest.json
    ├── sub_01_001.pt
    ├── sub_01_002.pt
    └── ...

Each ``.pt`` file contains a dict with:

* ``x`` — ``FloatTensor`` of shape ``[6, T]``
  (left_x, left_y, left_z, right_x, right_y, right_z)
* ``y`` — int label ID (0‑35)
* metadata fields: ``domain``, ``subject_id``, ``event_id``,
  ``sentence_id``, ``repeat_id``, ``length``
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Make the src package importable without installing.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from silentspeechoe.data.dataset import build_binaural_event_records  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Channel list (matches manifest schema)
# ---------------------------------------------------------------------------
_CHANNEL_NAMES = [
    "left_bone_acc_x",
    "left_bone_acc_y",
    "left_bone_acc_z",
    "right_bone_acc_x",
    "right_bone_acc_y",
    "right_bone_acc_z",
]

# Columns in each raw CSV we care about.
_BONE_COLS = ["bone_acc.x", "bone_acc.y", "bone_acc.z"]


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------


def _median_center_per_channel(x: np.ndarray) -> np.ndarray:
    """Subtract the per‑channel median.

    Args:
        x: ``float32`` array of shape ``[C, T]``.

    Returns:
        Centered ``float32`` array of shape ``[C, T]``.
    """
    medians = np.median(x, axis=1, keepdims=True).astype(np.float32)
    return x - medians


def _moving_average_3(x: np.ndarray) -> np.ndarray:
    """Apply a causal 3‑point moving average along the time axis.

    For *t* < 2 the window is truncated (no padding).

    Args:
        x: ``float32`` array of shape ``[C, T]``.

    Returns:
        Smoothed ``float32`` array of shape ``[C, T]``.
    """
    C, T = x.shape
    if T < 2:
        return x.copy()
    out = np.empty_like(x)
    for c in range(C):
        row = x[c]
        smoothed = np.convolve(row, np.ones(3) / 3.0, mode="same")
        # Fix boundary: first sample = itself, second = mean of first two.
        smoothed[0] = row[0]
        if T >= 2:
            smoothed[1] = (row[0] + row[1]) / 2.0
        out[c] = smoothed
    return out.astype(np.float32)


def _per_channel_zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply per‑channel z‑score standardisation.

    Args:
        x: ``float32`` array of shape ``[C, T]``.
        eps: Guard against zero std.

    Returns:
        ``float32`` array of shape ``[C, T]`` with near‑zero mean and
        near‑unit std per channel.
    """
    C, T = x.shape
    out = np.empty_like(x)
    for c in range(C):
        row = x[c]
        std = float(np.std(row))
        if std <= eps:
            out[c] = np.zeros_like(row)
        else:
            mean = float(np.mean(row))
            out[c] = (row - mean) / np.float32(std)
    return out


def preprocess_raw_bone_window(x: np.ndarray) -> np.ndarray:
    """Run the full raw bone‑acc preprocessing chain.

    Steps:
    1. Per‑channel median center.
    2. 3‑point moving average smoothing.
    3. Per‑channel z‑score.

    Args:
        x: ``float32`` array of shape ``[6, T]``.

    Returns:
        Preprocessed ``float32`` array of shape ``[6, T]``.
    """
    x = _median_center_per_channel(x)
    x = _moving_average_3(x)
    x = _per_channel_zscore(x)
    return x


# ---------------------------------------------------------------------------
# Simple LRU cache for CSV DataFrames
# ---------------------------------------------------------------------------


class _CSVCache:
    """Least‑recently‑used cache for ``pd.DataFrame`` keyed by path."""

    def __init__(self, max_size: int = 8) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def get(self, path: str) -> pd.DataFrame:
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        df = pd.read_csv(path)
        self._cache[path] = df
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)  # evict oldest
        return df

    def __len__(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pre‑compute raw binaural bone‑acc windows"
    )
    parser.add_argument(
        "--events",
        default=str(_PROJECT_ROOT / "data" / "metadata" / "events.csv"),
        help="Path to events.csv",
    )
    parser.add_argument(
        "--raw-dir",
        default=str(_PROJECT_ROOT / "data" / "raw"),
        help="Path to raw sensor data",
    )
    parser.add_argument(
        "--out-dir",
        default=str(_PROJECT_ROOT / "data" / "processed" / "raw_bone_binaural"),
        help="Output directory for .pt files and manifest",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=2,
        help="Minimum window length in samples (skip shorter)",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=8,
        help="Max number of CSV DataFrames kept in memory",
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _build_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load all paired records --------------------------------------------
    records = build_binaural_event_records(
        events_path=args.events,
        raw_dir=args.raw_dir,
    )
    logger.info("Total paired records: %d", len(records))

    # ---- pre‑compute --------------------------------------------------------
    csv_cache = _CSVCache(max_size=args.cache_size)
    manifest: list[dict] = []
    skipped: int = 0

    for idx, rec in enumerate(
        tqdm(records, desc="Pre‑computing raw windows", unit="smp")
    ):
        # --- load left window ------------------------------------------------
        left_df = csv_cache.get(str(rec["left_path"]))
        left_mask = (left_df["timestamp"] >= rec["left_start_time"]) & (
            left_df["timestamp"] <= rec["left_end_time"]
        )
        left_xyz = left_df.loc[left_mask, _BONE_COLS].to_numpy(dtype=np.float32).T
        # left_xyz: [3, T_l]

        # --- load right window -----------------------------------------------
        right_df = csv_cache.get(str(rec["right_path"]))
        right_mask = (right_df["timestamp"] >= rec["right_start_time"]) & (
            right_df["timestamp"] <= rec["right_end_time"]
        )
        right_xyz = (
            right_df.loc[right_mask, _BONE_COLS].to_numpy(dtype=np.float32).T
        )  # [3, T_r]

        # --- align: crop to minimum length -----------------------------------
        left_len = left_xyz.shape[1]
        right_len = right_xyz.shape[1]
        T = min(left_len, right_len)

        if T < args.min_length:
            logger.debug(
                "Skipping %s event %d: T=%d < %d (left=%d, right=%d)",
                rec["subject_id"],
                rec["event_id"],
                T,
                args.min_length,
                left_len,
                right_len,
            )
            skipped += 1
            continue

        left_cropped = left_xyz[:, :T]  # [3, T]
        right_cropped = right_xyz[:, :T]  # [3, T]

        # --- stack: [6, T] ---------------------------------------------------
        x_raw = np.concatenate([left_cropped, right_cropped], axis=0)  # [6, T]

        # --- preprocess ------------------------------------------------------
        x_proc = preprocess_raw_bone_window(x_raw)  # [6, T]

        # --- save ------------------------------------------------------------
        file_name = f"{rec['subject_id']}_{rec['event_id']}.pt"
        file_path = out_dir / file_name

        sample = {
            "x": torch.from_numpy(x_proc).contiguous(),  # [6, T]
            "y": int(rec["label_id"]),
            "domain": rec["domain"],
            "subject_id": rec["subject_id"],
            "event_id": int(rec["event_id"]),
            "sentence_id": str(rec["sentence_id"]),
            "repeat_id": int(rec["repeat_id"]),
            "length": int(T),
        }
        torch.save(sample, file_path)

        manifest.append(
            {
                "idx": idx,
                "file": file_name,
                "subject_id": rec["subject_id"],
                "event_id": int(rec["event_id"]),
                "domain": rec["domain"],
                "label_id": int(rec["label_id"]),
                "length": int(T),
            }
        )

    # ---- write manifest -----------------------------------------------------
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(
            {
                "num_samples": len(manifest),
                "channels": _CHANNEL_NAMES,
                "preprocessing": {
                    "align": "crop_to_min_length",
                    "center": "per_channel_median",
                    "smoothing": "moving_average_3",
                    "normalization": "per_channel_zscore",
                },
                "records": manifest,
            },
            f,
            indent=2,
        )

    logger.info("Manifest saved to %s", manifest_path)
    logger.info(
        "Done — %d samples written, %d skipped (T < %d)",
        len(manifest),
        skipped,
        args.min_length,
    )


if __name__ == "__main__":
    main()
