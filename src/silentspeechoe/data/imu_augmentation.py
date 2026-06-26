"""IMU window augmentation utilities for preprocessed OpenEarable data.

The module provides lightweight augmentations for ``[C, T]`` IMU windows:

* rotation: small 3D rotations applied per sensor triad
* time warping: monotonic time distortion with interpolation
* scaling: amplitude scaling per triad, or per channel as a fallback
* gaussian noise: per-channel standard-deviation-scaled noise

Augmentation is applied at the sample level first and then each operation
is sampled independently. This makes it easy to keep most windows unchanged
while still regularizing a small subset of samples.
"""

from __future__ import annotations

import json
import math
import shutil
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

_TRIAD_SIZE = 3
_STD_FLOOR = 1e-6


@dataclass(slots=True)
class IMUAugmentationConfig:
    """Configuration for IMU window augmentation."""

    enabled: bool = True
    sample_prob: float = 1.0
    rotation_prob: float = 0.05
    rotation_max_degrees: float = 5.0
    time_warp_prob: float = 0.05
    time_warp_min_scale: float = 0.95
    time_warp_max_scale: float = 1.05
    scaling_prob: float = 0.05
    scaling_min_scale: float = 0.95
    scaling_max_scale: float = 1.05
    gaussian_noise_prob: float = 0.05
    gaussian_noise_min_ratio: float = 0.01
    gaussian_noise_max_ratio: float = 0.03

    def __post_init__(self) -> None:
        _validate_probability("sample_prob", self.sample_prob)
        _validate_probability("rotation_prob", self.rotation_prob)
        _validate_probability("time_warp_prob", self.time_warp_prob)
        _validate_probability("scaling_prob", self.scaling_prob)
        _validate_probability("gaussian_noise_prob", self.gaussian_noise_prob)
        _validate_non_negative("rotation_max_degrees", self.rotation_max_degrees)
        _validate_bounds(
            "time_warp",
            self.time_warp_min_scale,
            self.time_warp_max_scale,
            lower_bound=0.0,
        )
        _validate_bounds(
            "scaling",
            self.scaling_min_scale,
            self.scaling_max_scale,
            lower_bound=0.0,
        )
        _validate_bounds(
            "gaussian_noise",
            self.gaussian_noise_min_ratio,
            self.gaussian_noise_max_ratio,
            lower_bound=0.0,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def build_imu_augmenter(
    config: Mapping[str, object] | None,
) -> IMUWindowAugmenter | None:
    """Create an augmenter from a mapping of Hydra-style config values."""
    if config is None:
        return None

    enabled = bool(config.get("enabled", True))
    if not enabled:
        return None

    imu_config = IMUAugmentationConfig(
        enabled=enabled,
        sample_prob=float(config.get("sample_prob", 1.0)),
        rotation_prob=float(config.get("rotation_prob", 0.05)),
        rotation_max_degrees=float(config.get("rotation_max_degrees", 5.0)),
        time_warp_prob=float(config.get("time_warp_prob", 0.05)),
        time_warp_min_scale=float(config.get("time_warp_min_scale", 0.95)),
        time_warp_max_scale=float(config.get("time_warp_max_scale", 1.05)),
        scaling_prob=float(config.get("scaling_prob", 0.05)),
        scaling_min_scale=float(config.get("scaling_min_scale", 0.95)),
        scaling_max_scale=float(config.get("scaling_max_scale", 1.05)),
        gaussian_noise_prob=float(config.get("gaussian_noise_prob", 0.05)),
        gaussian_noise_min_ratio=float(config.get("gaussian_noise_min_ratio", 0.01)),
        gaussian_noise_max_ratio=float(config.get("gaussian_noise_max_ratio", 0.03)),
    )
    return IMUWindowAugmenter(imu_config)


def augment_imu_window(
    x: torch.Tensor,
    config: IMUAugmentationConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply random IMU augmentations to a single ``[C, T]`` window."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(x)!r}")
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape [C, T], got {tuple(x.shape)}")
    if x.numel() == 0:
        return x.clone()
    if not config.enabled:
        return x.clone()

    out = x.clone()
    rng = generator
    if config.sample_prob <= 0.0:
        return out
    if config.sample_prob < 1.0 and _draw_probability(rng) >= config.sample_prob:
        return out

    applied = False
    if config.rotation_prob > 0 and _draw_probability(rng) < config.rotation_prob:
        next_out = apply_rotation(out, config.rotation_max_degrees, rng)
        if not torch.equal(next_out, out):
            applied = True
        out = next_out
    if (
        config.time_warp_prob > 0
        and out.shape[1] >= 2
        and _draw_probability(rng) < config.time_warp_prob
    ):
        next_out = apply_time_warp(
            out,
            min_scale=config.time_warp_min_scale,
            max_scale=config.time_warp_max_scale,
            generator=rng,
        )
        if not torch.equal(next_out, out):
            applied = True
        out = next_out
    if config.scaling_prob > 0 and _draw_probability(rng) < config.scaling_prob:
        next_out = apply_scaling(
            out,
            min_scale=config.scaling_min_scale,
            max_scale=config.scaling_max_scale,
            generator=rng,
        )
        if not torch.equal(next_out, out):
            applied = True
        out = next_out
    if (
        config.gaussian_noise_prob > 0
        and _draw_probability(rng) < config.gaussian_noise_prob
    ):
        next_out = apply_gaussian_noise(
            out,
            min_ratio=config.gaussian_noise_min_ratio,
            max_ratio=config.gaussian_noise_max_ratio,
            generator=rng,
        )
        if not torch.equal(next_out, out):
            applied = True
        out = next_out

    if not applied:
        fallback = _apply_fallback_augmentation(out, config, rng)
        if fallback is not None:
            out = fallback

    return out


def apply_rotation(
    x: torch.Tensor,
    max_degrees: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply a small random 3D rotation per triad of channels."""
    _validate_window_tensor(x)
    if x.numel() == 0 or x.shape[0] < _TRIAD_SIZE or max_degrees <= 0:
        return x.clone()
    if x.shape[0] % _TRIAD_SIZE != 0:
        return x.clone()

    matrix = _rotation_matrix(max_degrees, x.device, x.dtype, generator)
    out = x.clone()
    for start in range(0, out.shape[0], _TRIAD_SIZE):
        out[start : start + _TRIAD_SIZE] = matrix @ out[start : start + _TRIAD_SIZE]
    return out


def apply_time_warp(
    x: torch.Tensor,
    *,
    min_scale: float,
    max_scale: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply temporal speed scaling and resample back to the original length."""
    _validate_window_tensor(x)
    if x.numel() == 0 or x.shape[1] < 2:
        return x.clone()
    if math.isclose(min_scale, 1.0) and math.isclose(max_scale, 1.0):
        return x.clone()

    if min_scale <= 0 or max_scale <= 0:
        raise ValueError("time-warp scales must be positive")
    if min_scale > max_scale:
        raise ValueError(
            f"time-warp min_scale must be <= max_scale, got {min_scale} > {max_scale}"
        )

    length = int(x.shape[1])
    scale = float(
        torch.empty((), dtype=torch.float64).uniform_(
            min_scale,
            max_scale,
            generator=generator,
        )
    )
    warped_length = max(2, int(round(length * scale)))
    if warped_length == length:
        return x.clone()

    work = x.unsqueeze(0).to(dtype=torch.float32)
    warped = F.interpolate(
        work,
        size=warped_length,
        mode="linear",
        align_corners=True,
    )
    out = F.interpolate(
        warped,
        size=length,
        mode="linear",
        align_corners=True,
    )
    return out.squeeze(0).to(device=x.device, dtype=x.dtype)


def apply_scaling(
    x: torch.Tensor,
    *,
    min_scale: float,
    max_scale: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply random amplitude scaling to the window."""
    _validate_window_tensor(x)
    if x.numel() == 0:
        return x.clone()
    if math.isclose(min_scale, 1.0) and math.isclose(max_scale, 1.0):
        return x.clone()

    if min_scale <= 0 or max_scale <= 0:
        raise ValueError("scaling factors must be positive")
    if min_scale > max_scale:
        raise ValueError(
            f"scaling min_scale must be <= max_scale, got {min_scale} > {max_scale}"
        )

    out = x.clone()
    group_size = _TRIAD_SIZE if out.shape[0] % _TRIAD_SIZE == 0 else 1
    num_groups = out.shape[0] // group_size
    scales = torch.empty(num_groups, dtype=torch.float64)
    scales.uniform_(min_scale, max_scale, generator=generator)

    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = start + group_size
        scale = scales[group_idx].to(dtype=out.dtype, device=out.device)
        out[start:end] = out[start:end] * scale

    return out


def apply_gaussian_noise(
    x: torch.Tensor,
    *,
    min_ratio: float,
    max_ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add per-channel Gaussian noise scaled by the channel standard deviation."""
    _validate_window_tensor(x)
    if x.numel() == 0:
        return x.clone()
    if min_ratio < 0 or max_ratio < 0:
        raise ValueError("gaussian-noise ratios must be non-negative")
    if min_ratio > max_ratio:
        raise ValueError(
            "gaussian-noise min_ratio must be <= max_ratio, got "
            f"{min_ratio} > {max_ratio}"
        )
    if math.isclose(min_ratio, 0.0) and math.isclose(max_ratio, 0.0):
        return x.clone()

    noise_ratio = _sample_uniform_scalar(min_ratio, max_ratio, generator)
    channel_std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(_STD_FLOOR)
    noise = torch.randn(x.shape, dtype=torch.float32, generator=generator).to(
        device=x.device,
        dtype=x.dtype,
    )
    return x + noise * channel_std * noise_ratio


def augment_processed_imu_sample(
    sample: dict,
    config: IMUAugmentationConfig,
    generator: torch.Generator | None = None,
) -> dict:
    """Return a shallow copy of *sample* with the ``x`` tensor augmented."""
    x = sample["x"]
    if not isinstance(x, torch.Tensor):
        raise TypeError("sample['x'] must be a torch.Tensor")

    augmented = {**sample}
    augmented["x_original"] = x.clone()
    augmented["x"] = augment_imu_window(x, config, generator=generator)
    augmented["augmentation"] = config.to_dict()
    return augmented


def augment_processed_imu_directory(
    input_dir: str | Path,
    out_dir: str | Path,
    *,
    config: IMUAugmentationConfig,
    seed: int = 42,
    overwrite: bool = False,
) -> Path:
    """Augment a directory of precomputed IMU ``.pt`` windows."""
    input_path = Path(input_dir)
    output_path = Path(out_dir)

    _validate_input_directory(input_path)
    _prepare_output_directory(output_path, overwrite=overwrite)

    manifest = _load_manifest(input_path)
    records = manifest.get("records", [])
    if not records:
        raise ValueError(
            f"No records found in manifest: {input_path / 'manifest.json'}"
        )

    tmp_dir = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator()
    generator.manual_seed(seed)

    try:
        for index, record in enumerate(records, start=1):
            src_path = input_path / str(record["file"])
            dst_path = tmp_dir / str(record["file"])
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            sample = torch.load(src_path, map_location="cpu", weights_only=False)
            augmented = augment_processed_imu_sample(sample, config, generator)
            torch.save(augmented, dst_path)

            if index % 500 == 0:
                print(f"  {index}/{len(records)} samples processed")

        print(f"  {len(records)}/{len(records)} samples processed")

        manifest_out = _build_augmented_manifest(manifest, input_path, config, seed)
        with (tmp_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest_out, handle, indent=2, ensure_ascii=False)

        tmp_dir.rename(output_path)
        return output_path

    except Exception:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        raise


def _build_augmented_manifest(
    manifest: dict,
    input_dir: Path,
    config: IMUAugmentationConfig,
    seed: int,
) -> dict:
    augmented = {**manifest}
    augmented["source_dir"] = str(input_dir)
    augmented["augmentation"] = {
        **config.to_dict(),
        "seed": seed,
        "type": "imu_window",
    }

    preprocessing = augmented.get("preprocessing")
    augmentation_name = "rotation/time_warp/scaling/gaussian_noise"
    if isinstance(preprocessing, dict):
        preprocessing = {**preprocessing}
        preprocessing["augmentation"] = augmentation_name
        augmented["preprocessing"] = preprocessing
    else:
        augmented["preprocessing"] = {"augmentation": augmentation_name}

    return augmented


def _load_manifest(input_dir: Path) -> dict:
    with (input_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_input_directory(input_dir: Path) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory does not exist or is not a directory: {input_dir}"
        )
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"manifest.json not found in input directory: {input_dir}"
        )


def _prepare_output_directory(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use overwrite=True to replace it."
            )


def _validate_window_tensor(x: torch.Tensor) -> None:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)!r}")
    if x.ndim != 2:
        raise ValueError(f"Expected x with shape [C, T], got {tuple(x.shape)}")


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _validate_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _validate_bounds(
    name: str,
    low: float,
    high: float,
    *,
    lower_bound: float,
) -> None:
    if low < lower_bound or high < lower_bound:
        raise ValueError(
            f"{name} bounds must be >= {lower_bound}, got {low} and {high}"
        )
    if low > high:
        raise ValueError(f"{name} min must be <= max, got {low} > {high}")


def _draw_probability(generator: torch.Generator | None) -> float:
    return float(torch.rand((), generator=generator).item())


def _sample_index(high: int, generator: torch.Generator | None) -> int:
    if high <= 1:
        return 0
    if generator is None:
        return int(torch.randint(high, ()).item())
    return int(torch.randint(high, (), generator=generator).item())


def _sample_uniform_scalar(
    low: float,
    high: float,
    generator: torch.Generator | None,
) -> float:
    if math.isclose(low, high):
        return float(low)
    return float(torch.empty(()).uniform_(low, high, generator=generator).item())


def _rotation_matrix(
    max_degrees: float,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> torch.Tensor:
    max_radians = math.radians(max_degrees)
    angles = torch.empty(3, dtype=torch.float64)
    angles.uniform_(-max_radians, max_radians, generator=generator)
    ax, ay, az = (float(angle) for angle in angles)

    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)

    rx = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=dtype,
        device=device,
    )
    ry = torch.tensor(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=dtype,
        device=device,
    )
    rz = torch.tensor(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    return rz @ ry @ rx


def _apply_fallback_augmentation(
    x: torch.Tensor,
    config: IMUAugmentationConfig,
    generator: torch.Generator | None,
) -> torch.Tensor | None:
    """Apply one valid augmentation when no operation was sampled."""
    candidates: list[Callable[[torch.Tensor], torch.Tensor]] = []

    if config.rotation_prob > 0 and config.rotation_max_degrees > 0:
        if x.shape[0] >= _TRIAD_SIZE and x.shape[0] % _TRIAD_SIZE == 0:
            candidates.append(
                lambda tensor: apply_rotation(
                    tensor,
                    config.rotation_max_degrees,
                    generator,
                )
            )

    if config.time_warp_prob > 0 and x.shape[1] >= 2:
        if not (
            math.isclose(config.time_warp_min_scale, 1.0)
            and math.isclose(config.time_warp_max_scale, 1.0)
        ):
            candidates.append(
                lambda tensor: apply_time_warp(
                    tensor,
                    min_scale=config.time_warp_min_scale,
                    max_scale=config.time_warp_max_scale,
                    generator=generator,
                )
            )

    if config.scaling_prob > 0:
        if not (
            math.isclose(config.scaling_min_scale, 1.0)
            and math.isclose(config.scaling_max_scale, 1.0)
        ):
            candidates.append(
                lambda tensor: apply_scaling(
                    tensor,
                    min_scale=config.scaling_min_scale,
                    max_scale=config.scaling_max_scale,
                    generator=generator,
                )
            )

    if config.gaussian_noise_prob > 0:
        if not (
            math.isclose(config.gaussian_noise_min_ratio, 0.0)
            and math.isclose(config.gaussian_noise_max_ratio, 0.0)
        ):
            candidates.append(
                lambda tensor: apply_gaussian_noise(
                    tensor,
                    min_ratio=config.gaussian_noise_min_ratio,
                    max_ratio=config.gaussian_noise_max_ratio,
                    generator=generator,
                )
            )

    if not candidates:
        return None

    return candidates[_sample_index(len(candidates), generator)](x)


class IMUWindowAugmenter:
    """Callable IMU window augmenter built from :class:`IMUAugmentationConfig`."""

    def __init__(self, config: IMUAugmentationConfig | None = None):
        self.config = config or IMUAugmentationConfig()

    def __call__(
        self,
        x: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return augment_imu_window(x, self.config, generator=generator)


__all__ = [
    "IMUAugmentationConfig",
    "IMUWindowAugmenter",
    "apply_gaussian_noise",
    "apply_rotation",
    "apply_scaling",
    "apply_time_warp",
    "augment_imu_window",
    "augment_processed_imu_directory",
    "augment_processed_imu_sample",
    "build_imu_augmenter",
]
