"""Build binaural temporal‑envelope features from left + right ear features.

Pairs left and right samples by metadata key, concatenates the three
feature blocks (left, right, left − right), and saves 1296‑dim vectors.

Usage::

    python scripts/precompute_imu_binaural_temporal_envelope.py \\
        --left-dir data/processed/features/imu_temporal_envelope_left_200hz_raw9 \\
        --right-dir data/processed/features/imu_temporal_envelope_right_200hz_raw9 \\
        --out-dir data/processed/features/imu_te_binaural_lrdiff_200hz_raw9
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Columns used to pair left and right samples.
_PAIR_KEY = [
    "subject_id",
    "event_id",
    "sentence_id",
    "label_id",
    "domain",
    "repeat_id",
    "sentence_type",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build binaural temporal-envelope features."
    )
    p.add_argument("--left-dir", required=True, help="Left ear feature directory.")
    p.add_argument("--right-dir", required=True, help="Right ear feature directory.")
    p.add_argument("--out-dir", required=True, help="Output directory.")
    return p.parse_args(argv)


def _pair_key_from_rec(rec: dict) -> tuple:
    return tuple(rec.get(field) for field in _PAIR_KEY)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load manifests.
    left_manifest_path = left_dir / "manifest.json"
    right_manifest_path = right_dir / "manifest.json"
    if not left_manifest_path.exists():
        raise FileNotFoundError(f"Left manifest missing: {left_manifest_path}")
    if not right_manifest_path.exists():
        raise FileNotFoundError(f"Right manifest missing: {right_manifest_path}")

    with left_manifest_path.open("r") as f:
        left_manifest = json.load(f)
    with right_manifest_path.open("r") as f:
        right_manifest = json.load(f)

    left_records = left_manifest["records"]
    right_records = right_manifest["records"]

    logger.info("Left  samples : %d", len(left_records))
    logger.info("Right samples : %d", len(right_records))

    # Index right by pairing key for fast lookup.
    right_index: dict[tuple, dict] = {}
    for rec in right_records:
        key = _pair_key_from_rec(rec)
        right_index[key] = rec

    single_dim = int(left_manifest.get("feature_dim", 432))
    binaural_dim = single_dim * 3  # left + right + (left - right)

    new_records: list[dict] = []
    skipped: list[dict] = []
    num_unmatched_left = 0
    num_unmatched_right = 0

    for left_rec in left_records:
        key = _pair_key_from_rec(left_rec)
        right_rec = right_index.pop(key, None)

        if right_rec is None:
            num_unmatched_left += 1
            skipped.append(
                {
                    "side": "left_only",
                    "subject_id": left_rec["subject_id"],
                    "event_id": left_rec.get("event_id"),
                    "domain": left_rec.get("domain"),
                    "sentence_id": left_rec.get("sentence_id"),
                }
            )
            continue

        # Load features.
        left_data = torch.load(left_dir / left_rec["file"], weights_only=True)
        right_data = torch.load(right_dir / right_rec["file"], weights_only=True)

        left_feat = left_data["x"]  # [432]
        right_feat = right_data["x"]  # [432]
        diff_feat = left_feat - right_feat  # [432]
        binaural_feat = torch.cat([left_feat, right_feat, diff_feat])  # [1296]

        out_file = left_rec["file"]  # reuse left filename
        out_path = out_dir / out_file
        torch.save(
            {
                "x": binaural_feat,
                "y": int(left_data.get("y", left_rec.get("label_id", -1))),
                "label_id": int(left_rec.get("label_id", -1)),
                "domain": left_rec.get("domain", ""),
                "subject_id": left_rec["subject_id"],
                "event_id": int(left_rec.get("event_id", -1)),
                "sentence_id": str(left_rec.get("sentence_id", "")),
                "sentence_type": str(left_rec.get("sentence_type", "")),
                "repeat_id": int(left_data.get("repeat_id", -1)),
                "side": "binaural",
                "left_source_file": str(left_dir / left_rec["file"]),
                "right_source_file": str(right_dir / right_rec["file"]),
                "feature_layout": {
                    "left": [0, single_dim],
                    "right": [single_dim, 2 * single_dim],
                    "left_minus_right": [2 * single_dim, 3 * single_dim],
                },
            },
            out_path,
        )

        new_records.append(
            {
                "idx": len(new_records),
                "file": out_file,
                "subject_id": left_rec["subject_id"],
                "domain": left_rec.get("domain", ""),
                "label_id": int(left_rec.get("label_id", -1)),
                "sentence_id": str(left_rec.get("sentence_id", "")),
                "sentence_type": str(left_rec.get("sentence_type", "")),
                "event_id": int(left_rec.get("event_id", -1)),
                "repeat_id": int(left_data.get("repeat_id", -1)),
                "side": "binaural",
            }
        )

    # Remaining right-only records.
    for _key, right_rec in right_index.items():
        num_unmatched_right += 1
        skipped.append(
            {
                "side": "right_only",
                "subject_id": right_rec["subject_id"],
                "event_id": right_rec.get("event_id"),
                "domain": right_rec.get("domain"),
                "sentence_id": right_rec.get("sentence_id"),
            }
        )

    # Write manifest.
    out_manifest = {
        "name": out_dir.name,
        "num_samples": len(new_records),
        "num_skipped": len(skipped),
        "num_unmatched_left": num_unmatched_left,
        "num_unmatched_right": num_unmatched_right,
        "feature_dim": binaural_dim,
        "single_ear_feature_dim": single_dim,
        "left_dir": str(left_dir.resolve()),
        "right_dir": str(right_dir.resolve()),
        "diff_definition": "left_feature_minus_right_feature",
        "feature_layout": {
            "left": [0, single_dim],
            "right": [single_dim, 2 * single_dim],
            "left_minus_right": [2 * single_dim, 3 * single_dim],
        },
        "channels": left_manifest.get("channels", []),
        "derived_signals": left_manifest.get("derived_signals", []),
        "feature_groups": left_manifest.get("feature_groups", {}),
        "records": new_records,
        "skipped": skipped,
    }
    out_manifest_path = out_dir / "manifest.json"
    with out_manifest_path.open("w") as f:
        json.dump(out_manifest, f, indent=2)

    logger.info(
        "Binaural: %d pairs saved, %d skipped (left-only=%d, right-only=%d)",
        len(new_records),
        len(skipped),
        num_unmatched_left,
        num_unmatched_right,
    )


if __name__ == "__main__":
    main()
