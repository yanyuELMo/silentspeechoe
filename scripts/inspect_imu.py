"""Quick inspection script for IMU preprocessing.

Prints record counts, sample shapes, and a few example windows so you can
sanity-check the pipeline before wiring it into training.

Usage::

    python scripts/inspect_imu.py
    python scripts/inspect_imu.py imu.sides=[left]
    python scripts/inspect_imu.py imu.sides=[right]
    python scripts/inspect_imu.py imu.sides=[left,right]
    python scripts/inspect_imu.py imu.target_sample_rate=100
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the package is importable when the script is run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from silentspeechoe.data.imu_preprocessing import (  # noqa: E402
    IMU_CHANNELS,
    IMUDataset,
    build_imu_records,
    imu_pad_collate,
)

logger = logging.getLogger(__name__)


def _summarize_records(records: list[dict]) -> None:
    """Print summary statistics about a list of IMU records."""
    print(f"\n{'=' * 60}")
    print(f"Total records: {len(records)}")

    if not records:
        print("(no records — check sides / raw data availability)")
        return

    # Per-side counts.
    sides: dict[str, int] = {}
    for r in records:
        sides[r["side"]] = sides.get(r["side"], 0) + 1
    print(f"Per side: {sides}")

    # Per-domain counts.
    domains: dict[str, int] = {}
    for r in records:
        domains[r["domain"]] = domains.get(r["domain"], 0) + 1
    print(f"Per domain: {domains}")

    # Unique subjects.
    subjects = sorted({r["subject_id"] for r in records})
    print(f"Unique subjects ({len(subjects)}): {subjects}")

    # Label range.
    labels = sorted({r["label_id"] for r in records})
    print(f"Label range: {min(labels)} – {max(labels)} ({len(labels)} unique)")

    # Sentence types.
    stypes = {r["sentence_type"] for r in records}
    print(f"Sentence types: {stypes}")


def _inspect_samples(
    records: list[dict],
    cfg: DictConfig,
    *,
    num_samples: int = 3,
) -> None:
    """Load and print a few sample windows."""
    if not records:
        return

    ds = IMUDataset(
        records,
        target_sample_rate=float(cfg.imu.target_sample_rate),
        padding_sec=float(cfg.imu.padding_sec),
        normalize=bool(cfg.imu.normalize),
    )

    print(f"\n{'=' * 60}")
    print(f"Dataset length: {len(ds)}")
    print(f"Target sample rate: {cfg.imu.target_sample_rate} Hz")
    print(f"Normalize: {cfg.imu.normalize}")

    rng = np.random.default_rng(42)
    indices = rng.choice(len(ds), size=min(num_samples, len(ds)), replace=False)

    samples = [ds[int(i)] for i in indices]
    batch = imu_pad_collate(samples)

    print(f"\nBatch x shape: {batch['x'].shape}  (B={len(samples)}, C=9, max_T)")
    print(f"Batch lengths: {batch['lengths'].tolist()}")
    print(f"Batch y: {batch['y'].tolist()}")
    print(f"Batch domains: {batch['domain']}")

    for i in range(len(samples)):
        print(
            f"\n  Sample {i}: "
            f"x={list(samples[i]['x'].shape)}, "
            f"y={samples[i]['y']}, "
            f"length={samples[i]['length']}, "
            f"domain={samples[i]['domain']}, "
            f"subject={samples[i]['subject_id']}, "
            f"side={samples[i]['side']}, "
            f"sent={samples[i]['sentence_id']}, "
            f"rep={samples[i]['repeat_id']}"
        )
        # Per-channel value ranges.
        x_np = samples[i]["x"].numpy()
        if x_np.shape[1] > 0:
            for c in range(min(9, x_np.shape[0])):
                ch_name = IMU_CHANNELS[c]
                print(
                    f"    {ch_name:>8s}: "
                    f"min={x_np[c].min():.3f}, "
                    f"max={x_np[c].max():.3f}, "
                    f"mean={x_np[c].mean():.3f}, "
                    f"std={x_np[c].std():.3f}"
                )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Inspect IMU preprocessing pipeline."""
    print(f"Sides: {cfg.imu.sides}")
    print(f"Channels: {cfg.imu.channels}")
    print(f"Target sample rate: {cfg.imu.target_sample_rate} Hz")

    # Build records from events.csv.
    records = build_imu_records(
        events_path=cfg.imu.metadata_file,
        raw_dir=cfg.imu.raw_root,
        sides=list(cfg.imu.sides),
    )

    _summarize_records(records)
    _inspect_samples(records, cfg)

    print(f"\n{'=' * 60}")
    print("IMU inspection complete.")


if __name__ == "__main__":
    main()
