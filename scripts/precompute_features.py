"""Pre‑compute binaural bone‑acc features and cache them as ``.pt`` files.

Run once before training to eliminate per‑epoch feature extraction overhead::

    python scripts/precompute_features.py                  # CPU
    python scripts/precompute_features.py --device cuda     # GPU‑accelerated

Output structure::

    data/processed/features/bone_binaural/
    ├── manifest.json
    ├── sub_01_001.pt
    ├── sub_01_002.pt
    └── ...
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Make the src package importable without installing.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from silentspeechoe.data.dataset import build_binaural_event_records  # noqa: E402
from silentspeechoe.features.bone_acc import (  # noqa: E402
    extract_binaural_bone_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pre‑compute bone‑acc features")
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
        default=str(
            _PROJECT_ROOT / "data" / "processed" / "features" / "bone_binaural"
        ),
        help="Output directory for .pt files and manifest",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for torch spectral feature computation",
    )
    parser.add_argument(
        "--frame-ms",
        type=float,
        default=50.0,
        help="Frame duration in milliseconds",
    )
    parser.add_argument(
        "--hop-ms",
        type=float,
        default=10.0,
        help="Frame stride in milliseconds",
    )
    return parser


def _load_window(path: Path, start: float, end: float) -> tuple:
    """Load one bone‑acc window from a CSV file.

    Returns ``(xyz, timestamps)`` — both NumPy arrays.
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(path)
    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    xyz = df.loc[mask, ["bone_acc.x", "bone_acc.y", "bone_acc.z"]].to_numpy(
        dtype=np.float32
    )
    ts = df.loc[mask, "timestamp"].to_numpy(dtype=np.float64)
    return xyz, ts


def main() -> None:
    args = _build_parser().parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    # ---- load all paired records --------------------------------------------
    records = build_binaural_event_records(
        events_path=args.events,
        raw_dir=args.raw_dir,
    )
    logger.info("Total paired records: %d", len(records))

    # ---- pre‑compute features -----------------------------------------------
    manifest: list[dict] = []

    for idx, rec in enumerate(tqdm(records, desc="Pre‑computing features", unit="smp")):
        # Load left window.
        left_xyz, left_ts = _load_window(
            rec["left_path"], rec["left_start_time"], rec["left_end_time"]
        )
        # Load right window.
        right_xyz, right_ts = _load_window(
            rec["right_path"], rec["right_start_time"], rec["right_end_time"]
        )

        # Feature extraction (NumPy preprocessing on CPU; torch ops on *device*).
        features, meta = extract_binaural_bone_features(
            left_xyz,
            left_ts,
            right_xyz,
            right_ts,
            frame_ms=args.frame_ms,
            hop_ms=args.hop_ms,
            device=device,
        )

        # Save per‑record .pt file (always CPU tensor for portability).
        file_name = f"{rec['subject_id']}_{rec['event_id']}.pt"
        file_path = out_dir / file_name

        save_dict = {
            "x": features.cpu().contiguous().T,  # [C, N] — consistent with dataset
            "y": int(rec["label_id"]),
            "domain": rec["domain"],
            "subject_id": rec["subject_id"],
            "event_id": int(rec["event_id"]),
            "sentence_id": str(rec["sentence_id"]),
            "repeat_id": int(rec["repeat_id"]),
            "num_frames": meta["num_frames"],
            "left_num_frames": meta["left_num_frames"],
            "right_num_frames": meta["right_num_frames"],
        }
        torch.save(save_dict, file_path)

        manifest.append(
            {
                "idx": idx,
                "file": file_name,
                "subject_id": rec["subject_id"],
                "event_id": int(rec["event_id"]),
                "domain": rec["domain"],
                "label_id": int(rec["label_id"]),
            }
        )

    # ---- write manifest -----------------------------------------------------
    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(
            {
                "num_samples": len(manifest),
                "frame_ms": args.frame_ms,
                "hop_ms": args.hop_ms,
                "feature_dim": 60,  # 6 channels × 10 per‑ear features
                "records": manifest,
            },
            f,
            indent=2,
        )
    logger.info("Manifest saved to %s", manifest_path)
    logger.info("Done — %d feature files written to %s", len(manifest), out_dir)


if __name__ == "__main__":
    main()
