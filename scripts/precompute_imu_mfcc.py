"""Pre‑compute MFCC features for all processed IMU samples.

Reads every ``.pt`` file referenced by the manifest in the processed IMU
directory, extracts fixed‑length MFCC features (234‑dim by default), and
saves them alongside a new manifest into the output directory.

Usage::

    python scripts/precompute_imu_mfcc.py \\
        --input-dir data/processed/imu_windows/left_200hz_raw9 \\
        --out-dir data/processed/features/imu_mfcc_left_200hz_raw9

    python scripts/precompute_imu_mfcc.py \\
        --input-dir data/processed/imu_windows/left_200hz_raw9 \\
        --out-dir data/processed/features/imu_mfcc_left_200hz_raw9 \\
        --n-mfcc 13 --n-mels 20 --frame-length 50 --hop-length 10
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

from silentspeechoe.features.imu_mfcc import (  # noqa: E402
    extract_imu_mfcc_features,
    feature_dim,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-compute MFCC features for processed IMU windows."
    )
    p.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing manifest.json and .pt files "
        "(e.g. data/processed/imu_windows/left_200hz_raw9).",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for features and new manifest.",
    )
    p.add_argument(
        "--n-mfcc", type=int, default=13, help="Number of MFCC coefficients."
    )
    p.add_argument("--n-mels", type=int, default=20, help="Mel filterbank channels.")
    p.add_argument(
        "--frame-length", type=int, default=50, help="Frame length in samples."
    )
    p.add_argument("--hop-length", type=int, default=10, help="Hop length in samples.")
    p.add_argument("--fmin", type=float, default=0.5, help="Lowest mel frequency (Hz).")
    p.add_argument(
        "--fmax", type=float, default=90.0, help="Highest mel frequency (Hz)."
    )
    p.add_argument(
        "--sample-rate", type=float, default=200.0, help="Sample rate in Hz."
    )
    p.add_argument(
        "--n-fft", type=int, default=None, help="FFT size (auto if omitted)."
    )
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
    num_channels = len(channels)
    dim = feature_dim(num_channels=num_channels, n_mfcc=args.n_mfcc)

    logger.info("Input dir    : %s", input_dir)
    logger.info("Output dir   : %s", out_dir)
    logger.info("Samples      : %d", len(records))
    logger.info("Channels     : %d (%s)", num_channels, ", ".join(channels))
    logger.info("Feature dim  : %d", dim)
    logger.info(
        "MFCC params  : n_mfcc=%d n_mels=%d frame=%d hop=%d fmin=%.1f fmax=%.1f",
        args.n_mfcc,
        args.n_mels,
        args.frame_length,
        args.hop_length,
        args.fmin,
        args.fmax,
    )

    new_records: list[dict] = []
    num_skipped = 0

    for rec in records:
        src_file = input_dir / rec["file"]
        if not src_file.exists():
            logger.warning("Source file missing: %s — skipping", src_file)
            num_skipped += 1
            continue

        data = torch.load(src_file, weights_only=True)
        x = data["x"].numpy()  # [C, T]

        features = extract_imu_mfcc_features(
            x,
            sample_rate=args.sample_rate,
            n_mfcc=args.n_mfcc,
            n_mels=args.n_mels,
            frame_length=args.frame_length,
            hop_length=args.hop_length,
            fmin=args.fmin,
            fmax=args.fmax,
            n_fft=args.n_fft,
        )

        out_file = rec["file"]  # reuse same filename
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
                "idx": int(rec.get("idx", len(new_records))),
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
        "num_samples": len(new_records),
        "num_skipped": num_skipped,
        "feature_dim": dim,
        "source_dir": str(input_dir.resolve()),
        "sample_rate": args.sample_rate,
        "channels": channels,
        "mfcc": {
            "n_mfcc": args.n_mfcc,
            "n_mels": args.n_mels,
            "frame_length": args.frame_length,
            "hop_length": args.hop_length,
            "fmin": args.fmin,
            "fmax": args.fmax,
            "use_delta": False,
            "pooling": ["mean", "std"],
            "n_fft": args.n_fft,
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
