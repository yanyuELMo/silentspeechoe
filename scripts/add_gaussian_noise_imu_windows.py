"""Add per-sample per-channel Gaussian noise to processed IMU window .pt files.

Reads raw IMU windows from an input directory, adds Gaussian noise scaled by
each channel's own standard deviation, and writes a new noisy dataset.

Noise definition:
    channel_std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    noise = torch.randn_like(x) * channel_std * noise_ratio
    x_noisy = x + noise

Usage:
    python scripts/add_gaussian_noise_imu_windows.py \
        --input-dir data/processed/imu_windows/left_200hz_raw9 \
        --out-dir data/processed/imu_windows/left_200hz_raw9_gaussian_noise_005 \
        --noise-ratio 0.05 \
        --seed 42
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add per-sample per-channel Gaussian noise to IMU windows.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing raw IMU .pt windows and manifest.json.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for noisy .pt windows and manifest.json.",
    )
    parser.add_argument(
        "--noise-ratio",
        type=float,
        required=True,
        help="Noise strength as a fraction of per-channel std (e.g. 0.05 = 5%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the noise generator (default: 42).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, remove and recreate an existing output directory.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate command-line arguments and exit on problems."""
    if not args.input_dir.is_dir():
        sys.exit(
            f"Error: input-dir does not exist or is not a directory: {args.input_dir}"
        )

    manifest_path = args.input_dir / "manifest.json"
    if not manifest_path.is_file():
        sys.exit(f"Error: manifest.json not found in input-dir: {manifest_path}")

    if args.out_dir.exists():
        if args.overwrite:
            print(f"Removing existing output directory (--overwrite): {args.out_dir}")
            shutil.rmtree(args.out_dir)
        else:
            sys.exit(
                f"Error: output directory already exists: {args.out_dir}\n"
                f"Use --overwrite to replace it."
            )

    if args.noise_ratio < 0:
        sys.exit(f"Error: noise-ratio must be >= 0, got {args.noise_ratio}")


def load_manifest(input_dir: Path) -> dict:
    """Load and return the manifest.json from input_dir."""
    with (input_dir / "manifest.json").open("r") as f:
        return json.load(f)


def process_sample(
    sample: dict,
    noise_ratio: float,
    generator: torch.Generator,
) -> dict:
    """Add per-sample per-channel Gaussian noise to a single IMU window.

    The noise for each channel is:
        noise = N(0, 1) * channel_std * noise_ratio

    Where channel_std is the (unbiased=False) standard deviation of that
    channel in the original signal, clamped to a minimum of 1e-6.

    Args:
        sample: dict with key "x" mapped to a [C, T] float tensor.
        noise_ratio: fraction of per-channel std to use as noise sigma.
        generator: seeded torch.Generator for reproducibility.

    Returns:
        A new dict (shallow copy) with "x" replaced by x_noisy and a new
        "noise" metadata entry.  All other keys are preserved.
    """
    x = sample["x"]  # [C, T]
    channel_std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
    noise = (
        torch.randn(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )
        * channel_std
        * noise_ratio
    )
    x_noisy = x + noise

    noisy_sample = {**sample}
    noisy_sample["x"] = x_noisy
    noisy_sample["noise"] = {
        "type": "gaussian",
        "noise_ratio": noise_ratio,
        "std_reference": "per_sample_per_channel_std",
        "seed": generator.initial_seed(),
    }
    return noisy_sample


def process_directory(
    input_dir: Path,
    out_dir: Path,
    noise_ratio: float,
    seed: int,
) -> None:
    """Process all .pt files in input_dir and write noisy versions to out_dir.

    Uses a temporary directory (out_dir.tmp) to avoid leaving half-finished
    output on failure.
    """
    manifest = load_manifest(input_dir)
    records = manifest.get("records", [])
    num_samples = len(records)

    if num_samples == 0:
        sys.exit(f"Error: no records found in manifest for {input_dir}")

    tmp_dir = out_dir.with_suffix(out_dir.suffix + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator()
    generator.manual_seed(seed)

    print(f"Processing {num_samples} samples from {input_dir}")
    print(f"Noise ratio: {noise_ratio}, seed: {seed}")
    print(f"Temp output: {tmp_dir}")

    processed = 0
    try:
        for record in records:
            src_path = input_dir / record["file"]
            dst_path = tmp_dir / record["file"]

            sample = torch.load(src_path, map_location="cpu", weights_only=False)
            noisy_sample = process_sample(sample, noise_ratio, generator)
            torch.save(noisy_sample, dst_path)

            processed += 1
            if processed % 500 == 0:
                print(f"  {processed}/{num_samples} samples processed")

        print(f"  {processed}/{num_samples} samples processed")

        # Build and write the new manifest.
        noisy_manifest = build_noisy_manifest(manifest, input_dir, noise_ratio, seed)
        with (tmp_dir / "manifest.json").open("w") as f:
            json.dump(noisy_manifest, f, indent=2, ensure_ascii=False)

        # Atomically rename tmp -> final.
        tmp_dir.rename(out_dir)
        print(f"Done. Noisy dataset written to {out_dir}")

    except Exception:
        # Clean up temp directory on failure.
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise


def build_noisy_manifest(
    manifest: dict,
    input_dir: Path,
    noise_ratio: float,
    seed: int,
) -> dict:
    """Create the output manifest by updating the original manifest in place.

    Preserves all existing fields and adds noise-related metadata.
    """
    noisy = {**manifest}
    noisy["source_dir"] = str(input_dir)
    noisy["noise"] = {
        "type": "gaussian",
        "noise_ratio": noise_ratio,
        "std_reference": "per_sample_per_channel_std",
        "seed": seed,
    }

    # Append noise info to preprocessing if the key exists.
    if "preprocessing" in noisy and isinstance(noisy["preprocessing"], dict):
        noisy["preprocessing"]["noise"] = "gaussian per-sample per-channel std"
    elif "preprocessing" not in noisy:
        noisy["preprocessing"] = {
            "noise": "gaussian per-sample per-channel std",
        }

    return noisy


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    validate_args(args)

    process_directory(
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        noise_ratio=args.noise_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
