"""Pre‑compute temporal‑envelope features for all processed IMU samples.

Reads every ``.pt`` file referenced by the manifest, extracts a 432‑dim
feature vector per sample, and saves them alongside a new manifest.

Usage::

    python scripts/precompute_imu_temporal_envelope.py \\
        --input-dir data/processed/imu_windows/left_200hz_raw9 \\
        --out-dir data/processed/features/imu_temporal_envelope_left_200hz_raw9
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

# Ensure the package is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from silentspeechoe.features.imu_temporal_envelope import (  # noqa: E402
    _DIFF_STAT_NAMES,
    _ENVELOPE_STAT_NAMES,
    _GLOBAL_STAT_NAMES,
    _NUM_SEGMENTS,
    _SEGMENT_STAT_NAMES,
    FEATURE_DIM,
    extract_imu_temporal_envelope_features,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-compute temporal envelope features for IMU windows."
    )
    p.add_argument("--input-dir", required=True, help="Processed IMU directory.")
    p.add_argument("--out-dir", required=True, help="Output directory for features.")
    p.add_argument("--envelope-window", type=int, default=11, help="MA window size.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r") as f:
        manifest = json.load(f)

    records = manifest["records"]
    channels = manifest.get("channels", [f"ch{i}" for i in range(9)])

    logger.info("Input dir       : %s", input_dir)
    logger.info("Output dir      : %s", out_dir)
    logger.info("Samples         : %d", len(records))
    logger.info("Channels        : %d (%s)", len(channels), ", ".join(channels))
    logger.info("Feature dim     : %d", FEATURE_DIM)
    logger.info("Envelope window : %d", args.envelope_window)

    new_records: list[dict] = []
    num_skipped = 0

    for rec in records:
        src_file = input_dir / rec["file"]
        if not src_file.exists():
            logger.warning("Source file missing: %s — skipping", src_file)
            num_skipped += 1
            continue

        data = torch.load(src_file, weights_only=True)
        x = data["x"].numpy()  # [9, T]

        features = extract_imu_temporal_envelope_features(
            x,
            envelope_window=args.envelope_window,
        )

        out_file = rec["file"]
        out_path = out_dir / out_file
        torch.save(
            {
                "x": torch.from_numpy(features),
                "y": int(data["y"]),
                "domain": data["domain"],
                "subject_id": data["subject_id"],
                "session_id": data.get("session_id", ""),
                "event_id": int(rec.get("event_id", -1)),
                "sentence_id": str(data.get("sentence_id", "")),
                "sentence_type": str(rec.get("sentence_type", "")),
                "repeat_id": int(data.get("repeat_id", -1)),
                "side": data.get("side", ""),
                "source_file": str(src_file),
            },
            out_path,
        )

        new_records.append(
            {
                "idx": len(new_records),
                "file": out_file,
                "subject_id": rec["subject_id"],
                "domain": rec["domain"],
                "label_id": int(rec.get("label_id", data.get("y", -1))),
                "sentence_id": rec.get("sentence_id", ""),
                "sentence_type": rec.get("sentence_type", ""),
                "repeat_id": int(data.get("repeat_id", -1)),
                "side": data.get("side", ""),
                "source_file": str(src_file),
            }
        )

    # Write output manifest.
    out_manifest = {
        "name": out_dir.name,
        "num_samples": len(new_records),
        "num_skipped": num_skipped,
        "feature_dim": FEATURE_DIM,
        "source_dir": str(input_dir.resolve()),
        "sample_rate": manifest.get("target_sample_rate", 200.0),
        "channels": channels,
        "derived_signals": ["acc_mag", "gyro_mag", "mag_mag"],
        "num_segments": _NUM_SEGMENTS,
        "envelope": {
            "method": "moving_average_abs",
            "window_size": args.envelope_window,
        },
        "feature_groups": {
            "global_stats": _GLOBAL_STAT_NAMES,
            "diff_stats": _DIFF_STAT_NAMES,
            "envelope_stats": _ENVELOPE_STAT_NAMES,
            "segmented_stats": _SEGMENT_STAT_NAMES,
        },
        "records": new_records,
    }
    out_manifest_path = out_dir / "manifest.json"
    with out_manifest_path.open("w") as f:
        json.dump(out_manifest, f, indent=2)

    logger.info(
        "Saved %d features + manifest to %s (%d skipped)",
        len(new_records),
        out_dir,
        num_skipped,
    )


if __name__ == "__main__":
    main()
