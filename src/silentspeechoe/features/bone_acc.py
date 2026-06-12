"""Bone‑acceleration feature engineering for sentence classification.

Feature extraction converts binaural bone‑acc utterance windows into
normalised short‑time spectral features that preserve temporal structure
while reducing subject, device, and domain shortcuts.

Uses PyTorch for all feature computation so that batched GPU
acceleration is available.  The lightweight pre‑processing chain
(median‑center, dynamic magnitude, z‑score) stays in NumPy because
it runs per‑sample inside DataLoader workers.
"""

from __future__ import annotations

import numpy as np
import torch

from silentspeechoe.data.preprocessing import (
    center_bone_acc_axes,
    compute_dynamic_magnitude,
    zscore_signal,
)

# ---------------------------------------------------------------------------
# Default frequency bands for band‑energy ratio features (Hz)
# ---------------------------------------------------------------------------

DEFAULT_BANDS: tuple[tuple[float, float], ...] = (
    (1.0, 20.0),
    (20.0, 80.0),
    (80.0, 300.0),
    (300.0, 1000.0),
    (1000.0, 2500.0),
)

# Per‑ear per‑frame feature dimension with default bands:
#   4 static (log_energy, rms, spectral_centroid, spectral_bandwidth)
#   + len(DEFAULT_BANDS) band ratios
#   + 1 voicing peakiness
DEFAULT_PER_EAR_FEATURE_DIM = 4 + len(DEFAULT_BANDS) + 1  # 10


def _per_ear_feature_dim(bands: tuple[tuple[float, float], ...]) -> int:
    return 4 + len(bands) + 1


# ---------------------------------------------------------------------------
# Sampling rate estimation (NumPy — input comes from pandas timestamps)
# ---------------------------------------------------------------------------


def estimate_sampling_rate(timestamps: np.ndarray) -> float:
    """Estimate the sampling rate from a timestamp array.

    Uses the median of positive timestamp differences to be robust
    against occasional timestamp jitter or resets.

    Args:
        timestamps: 1‑D float array of mostly increasing
            timestamps (e.g. Unix microseconds or seconds).

    Returns:
        Estimated sampling rate in Hz.

    Raises:
        ValueError: If fewer than 2 samples, >1% of diffs are
            negative, or differences are all zero.
    """
    if timestamps.ndim != 1:
        raise ValueError(f"Expected 1‑D timestamps, got shape {timestamps.shape}")
    if timestamps.shape[0] < 2:
        raise ValueError("Need at least 2 timestamps to estimate sampling rate")

    diffs = np.diff(timestamps.astype(np.float64))

    neg_frac = float(np.mean(diffs < 0))
    if neg_frac > 0.01:
        raise ValueError(
            f"Timestamps are {neg_frac * 100:.1f}% non‑monotonic "
            f"(threshold 1%) — possible timestamp reset"
        )

    positive = diffs[diffs > 0]
    if len(positive) == 0:
        raise ValueError("All timestamp differences are zero — cannot estimate rate")

    median_diff = float(np.median(positive))
    if median_diff <= 0:
        raise ValueError(f"Median timestamp difference is {median_diff}")

    return 1.0 / median_diff


# ---------------------------------------------------------------------------
# Signal framing (torch — supports batched CPU / CUDA)
# ---------------------------------------------------------------------------


def frame_signal(
    signal: torch.Tensor,
    sample_rate: float,
    frame_ms: float = 50.0,
    hop_ms: float = 10.0,
) -> torch.Tensor:
    """Partition 1‑D signals into overlapping frames.

    Args:
        signal: ``[B, T]`` or ``[T]`` float tensor.
        sample_rate: Sampling rate in Hz.
        frame_ms: Frame duration in milliseconds.
        hop_ms: Hop (stride) duration in milliseconds.

    Returns:
        ``[B, num_frames, frame_len]`` (or ``[num_frames, frame_len]``
        for 1‑D input) float tensor on the same device as *signal*.

        * If ``T < frame_len``: 1 zero‑padded frame.
        * If ``T == frame_len``: exactly 1 frame.
        * If ``T == frame_len + hop_len``: exactly 2 full frames.
        * Otherwise ``num_frames = 1 + ceil((T - frame_len) / hop_len)``.
    """
    single = signal.dim() == 1
    if single:
        signal = signal.unsqueeze(0)

    B, total = signal.shape
    frame_len = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    hop_len = max(1, int(round(sample_rate * hop_ms / 1000.0)))

    if total <= frame_len:
        padded = torch.nn.functional.pad(signal, (0, frame_len - total), value=0.0)
        result = padded.unsqueeze(1)  # [B, 1, frame_len]
    else:
        unfolded = signal.unfold(dimension=1, size=frame_len, step=hop_len)

        # Check for partial final frame.
        num_full = unfolded.shape[1]
        last_start = (num_full - 1) * hop_len
        if last_start + frame_len < total and total - last_start > hop_len:
            partial_start = num_full * hop_len
            partial_len = total - partial_start
            if partial_len > 0:
                partial = signal[:, partial_start:]
                partial_padded = torch.nn.functional.pad(
                    partial, (0, frame_len - partial_len), value=0.0
                )
                result = torch.cat([unfolded, partial_padded.unsqueeze(1)], dim=1)
            else:
                result = unfolded
        else:
            result = unfolded

    if single:
        result = result.squeeze(0)
    return result.contiguous()


# ---------------------------------------------------------------------------
# Spectral feature computation (torch — batched CPU / CUDA)
# ---------------------------------------------------------------------------


def compute_frame_spectrum_features(
    frames: torch.Tensor,
    sample_rate: float,
    bands: tuple[tuple[float, float], ...] = DEFAULT_BANDS,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Compute per‑frame spectral features from framed signal windows.

    For each frame the following features are extracted:

    * ``log_energy`` — log10 of full‑frame time‑domain energy.
    * ``rms`` — root‑mean‑square of the unwindowed time‑domain frame.
    * ``spectral_centroid`` (Hz)
    * ``spectral_bandwidth`` (Hz)
    * band energy ratios — one per band in *bands*, normalised by
      total power in the valid spectral range (1–2500 Hz).
    * voicing peakiness in 80–300 Hz

    A symmetric Hann window is applied before the FFT for spectral
    features (centroid, bandwidth, band ratios, peakiness).
    RMS and log_energy are computed in the time domain.

    Best input shape for GPU: ``[batch, num_frames, frame_len]``.

    Args:
        frames: ``[B, N, L]`` or ``[N, L]`` float tensor.
        sample_rate: Sampling rate in Hz.
        bands: Frequency band definitions ``(low_hz, high_hz)``.
        eps: Division‑by‑zero guard.

    Returns:
        ``[B, N, F]`` or ``[N, F]`` float tensor on the same device as
        *frames*, where ``F = 4 + len(bands) + 1``.
    """
    single = frames.dim() == 2
    if single:
        frames = frames.unsqueeze(0)

    B, num_frames, frame_len = frames.shape
    device = frames.device
    nyquist = sample_rate / 2.0

    # Symmetric Hann window (matching np.hanning).
    window = torch.hann_window(
        frame_len, periodic=False, device=device, dtype=frames.dtype
    )

    # FFT frequency bins (single‑sided).
    freqs = torch.fft.rfftfreq(frame_len, d=1.0 / sample_rate, device=device)

    # Valid spectral range mask: 1 Hz to min(2500, nyquist).
    max_valid_hz = min(2500.0, nyquist)
    valid_mask = (freqs >= 1.0) & (freqs <= max_valid_hz)

    # Pre‑compute band masks with half‑open intervals.
    band_masks: list[torch.Tensor] = []
    for idx, (low, high) in enumerate(bands):
        capped_high = min(high, nyquist)
        if idx == len(bands) - 1:
            mask = (freqs >= low) & (freqs <= capped_high)
        else:
            mask = (freqs >= low) & (freqs < capped_high)
        band_masks.append(mask & valid_mask)

    # Voicing band mask (80–300 Hz).
    voice_mask = (freqs >= 80.0) & (freqs <= min(300.0, nyquist))

    # Apply Hann window: [B, N, L].
    windowed = frames * window[None, None, :]

    # Batched rFFT: [B, N, L//2+1].
    spec = torch.abs(torch.fft.rfft(windowed, n=frame_len, dim=-1))
    power = spec**2

    # -- time‑domain features ----------------------------------------------
    # log_energy: [B, N].
    td_energy = torch.sum(frames**2, dim=-1) + eps
    log_energy = torch.log10(td_energy + eps)

    # rms: [B, N].
    rms = torch.sqrt(torch.mean(frames**2, dim=-1) + eps)

    # -- spectral features -------------------------------------------------
    # Valid power sum: [B, N].
    valid_power_sum = torch.sum(power[..., valid_mask], dim=-1) + eps

    # Spectral centroid: [B, N].
    centroid = torch.sum(freqs[None, None, :] * spec, dim=-1) / (
        torch.sum(spec, dim=-1) + eps
    )

    # Spectral bandwidth: [B, N].
    diff = freqs[None, None, :] - centroid.unsqueeze(-1)
    bandwidth = torch.sqrt(
        torch.sum(diff**2 * spec, dim=-1) / (torch.sum(spec, dim=-1) + eps)
    )

    # Band energy ratios: list of [B, N].
    band_ratios: list[torch.Tensor] = []
    for bm in band_masks:
        band_power = torch.sum(power[..., bm], dim=-1)
        band_ratios.append(band_power / valid_power_sum)

    # Voicing peakiness: [B, N].
    voice_power = power[..., voice_mask]
    voice_power_size = voice_power.shape[-1]
    voice_mean = torch.mean(voice_power, dim=-1)
    if voice_power_size > 0:
        voice_max = torch.max(voice_power, dim=-1).values
    else:
        voice_max = torch.zeros_like(voice_mean)
    voice_peakiness = torch.where(
        (voice_mean > eps) & (voice_power_size > 0),
        voice_max / (voice_mean + eps),
        torch.zeros_like(voice_max),
    )

    # Assemble: [B, N, feat_dim].
    pieces = [
        log_energy.unsqueeze(-1),
        rms.unsqueeze(-1),
        centroid.unsqueeze(-1),
        bandwidth.unsqueeze(-1),
    ]
    for br in band_ratios:
        pieces.append(br.unsqueeze(-1))
    pieces.append(voice_peakiness.unsqueeze(-1))

    result = torch.cat(pieces, dim=-1)

    if single:
        result = result.squeeze(0)
    return result


# ---------------------------------------------------------------------------
# Delta features
# ---------------------------------------------------------------------------


def compute_delta(features: torch.Tensor) -> torch.Tensor:
    """Compute first‑order temporal delta features.

    Delta at frame ``t`` is ``features[t] - features[t-1]``.
    The first frame delta is all zeros.

    Args:
        features: ``[num_frames, num_features]`` float tensor.

    Returns:
        Tensor with the same shape and device as *features*.
    """
    if features.dim() != 2:
        raise ValueError(f"Expected 2‑D features, got shape {features.shape}")

    delta = torch.zeros_like(features)
    if features.shape[0] > 1:
        delta[1:] = features[1:] - features[:-1]
    return delta


# ---------------------------------------------------------------------------
# Batched z‑score
# ---------------------------------------------------------------------------


def batched_zscore(
    signals: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per‑window z‑score for a batch of 1‑D signals.

    Args:
        signals: ``[B, T]`` or ``[T]`` float tensor.
        eps: Guard against zero std.

    Returns:
        Z‑scored tensor with same shape and device.
    """
    single = signals.dim() == 1
    if single:
        signals = signals.unsqueeze(0)

    mean = torch.mean(signals, dim=-1, keepdim=True)
    std = torch.std(signals, dim=-1, keepdim=True)

    safe_std = torch.where(std > eps, std, torch.ones_like(std))
    result = (signals - mean) / safe_std
    result = torch.where(std > eps, result, torch.zeros_like(result))

    if single:
        result = result.squeeze(0)
    return result


# ---------------------------------------------------------------------------
# Single‑ear feature extraction
# ---------------------------------------------------------------------------


def extract_single_ear_bone_features(
    xyz: np.ndarray,
    timestamps: np.ndarray,
    frame_ms: float = 50.0,
    hop_ms: float = 10.0,
    bands: tuple[tuple[float, float], ...] = DEFAULT_BANDS,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Extract per‑frame spectral features for one ear.

    Pipeline:
    1. Median‑center axes (NumPy).
    2. Compute dynamic magnitude (NumPy).
    3. Per‑window z‑score (NumPy).
    4. Estimate sampling rate from timestamps (NumPy).
    5. Frame the signal (torch).
    6. Compute per‑frame spectral features (torch).

    Args:
        xyz: Raw bone‑acc array of shape ``[T, 3]``.
        timestamps: 1‑D timestamp array of shape ``[T]``.
        frame_ms: Frame length in milliseconds.
        hop_ms: Frame stride in milliseconds.
        bands: Frequency band definitions.
        device: If given, torch ops (framing, FFT, spectral features) run
            on this device.  NumPy preprocessing always runs on CPU.

    Returns:
        ``float32`` tensor of shape ``[num_frames, N]`` where
        ``N = 4 + len(bands) + 1``.
    """
    if xyz.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"xyz and timestamps must have the same length, "
            f"got {xyz.shape[0]} vs {timestamps.shape[0]}"
        )

    # Preprocessing chain (NumPy — fast, runs in DataLoader workers).
    centered = center_bone_acc_axes(xyz, method="median")
    magnitude = compute_dynamic_magnitude(centered)
    normed = zscore_signal(magnitude)

    # Framing + spectral features (torch — GPU‑accelerated when on CUDA).
    sample_rate = estimate_sampling_rate(timestamps)
    t_signal = torch.from_numpy(normed)
    if device is not None:
        t_signal = t_signal.to(device)
    frames = frame_signal(t_signal, sample_rate, frame_ms=frame_ms, hop_ms=hop_ms)
    return compute_frame_spectrum_features(frames, sample_rate, bands=bands)


# ---------------------------------------------------------------------------
# Binaural feature extraction
# ---------------------------------------------------------------------------


def extract_binaural_bone_features(
    left_xyz: np.ndarray,
    left_timestamps: np.ndarray,
    right_xyz: np.ndarray,
    right_timestamps: np.ndarray,
    frame_ms: float = 50.0,
    hop_ms: float = 10.0,
    bands: tuple[tuple[float, float], ...] = DEFAULT_BANDS,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, dict]:
    """Extract binaural spectral features with cross‑ear combinations.

    Steps:

    1. Extract per‑frame features for each ear independently.
    2. Pad the shorter frame sequence to the longer one (zero rows).
    3. Compute binaural combinations:
       * ``mean`` — element‑wise ``(left + right) / 2``
       * ``abs_diff`` — element‑wise ``|left - right|``
    4. Compute first‑order deltas of ``mean`` and ``abs_diff``.
    5. Concatenate into one feature matrix:
       ``[left, right, mean, abs_diff, delta_mean, delta_abs_diff]``

    Frame masks are included in metadata so downstream code can
    ignore padded‑tail regions:

    * ``left_frame_mask`` — ``True`` where left ear had a real frame.
    * ``right_frame_mask`` — ``True`` where right ear had a real frame.
    * ``joint_frame_mask`` — ``True`` only where *both* ears have
      real frames.

    Args:
        left_xyz: Raw left‑ear ``[T_l, 3]`` array.
        left_timestamps: Left‑ear timestamps ``[T_l]``.
        right_xyz: Raw right‑ear ``[T_r, 3]`` array.
        right_timestamps: Right‑ear timestamps ``[T_r]``.
        frame_ms: Frame length in milliseconds.
        hop_ms: Frame stride in milliseconds.
        bands: Frequency band definitions.
        device: If given, torch ops (framing, FFT, spectral features)
            run on this device.  NumPy preprocessing always runs on CPU.

    Returns:
        ``(features, meta)`` where:

        * ``features`` — ``float32`` tensor of shape
          ``[num_frames, N_total]``
          (``N_total = 6 × (4 + len(bands) + 1)``).
        * ``meta`` — dict with keys ``left_num_frames``,
          ``right_num_frames``, ``num_frames``, ``num_features``,
          ``feature_names``, ``left_frame_mask``,
          ``right_frame_mask``, ``joint_frame_mask``.
    """
    left_feat = extract_single_ear_bone_features(
        left_xyz,
        left_timestamps,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        bands=bands,
        device=device,
    )
    right_feat = extract_single_ear_bone_features(
        right_xyz,
        right_timestamps,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        bands=bands,
        device=device,
    )

    left_nf = left_feat.shape[0]
    right_nf = right_feat.shape[0]
    max_nf = max(left_nf, right_nf)
    feat_dim = left_feat.shape[1]
    device = left_feat.device

    # Build frame masks BEFORE padding.
    left_frame_mask = np.zeros(max_nf, dtype=bool)
    left_frame_mask[:left_nf] = True
    right_frame_mask = np.zeros(max_nf, dtype=bool)
    right_frame_mask[:right_nf] = True
    joint_frame_mask = left_frame_mask & right_frame_mask

    # Pad shorter sequence to match the longer one (zero rows).
    if left_nf < max_nf:
        pad = torch.zeros(max_nf - left_nf, feat_dim, device=device)
        left_feat = torch.cat([left_feat, pad], dim=0)
    if right_nf < max_nf:
        pad = torch.zeros(max_nf - right_nf, feat_dim, device=device)
        right_feat = torch.cat([right_feat, pad], dim=0)

    # Binaural combinations.
    mean_feat = (left_feat + right_feat) / 2.0
    abs_diff_feat = torch.abs(left_feat - right_feat)
    delta_mean_feat = compute_delta(mean_feat)
    delta_abs_diff_feat = compute_delta(abs_diff_feat)

    # Zero out fused feature rows where only one ear contributed.
    tail_mask = torch.from_numpy(~joint_frame_mask).to(device)
    mean_feat[tail_mask] = 0.0
    abs_diff_feat[tail_mask] = 0.0
    delta_mean_feat[tail_mask] = 0.0
    delta_abs_diff_feat[tail_mask] = 0.0

    features = torch.cat(
        [
            left_feat,
            right_feat,
            mean_feat,
            abs_diff_feat,
            delta_mean_feat,
            delta_abs_diff_feat,
        ],
        dim=1,
    )

    meta = {
        "left_num_frames": left_nf,
        "right_num_frames": right_nf,
        "num_frames": max_nf,
        "num_features": features.shape[1],
        "feature_names": default_bone_feature_names(bands=bands),
        "left_frame_mask": left_frame_mask,
        "right_frame_mask": right_frame_mask,
        "joint_frame_mask": joint_frame_mask,
    }
    return features, meta


# ---------------------------------------------------------------------------
# Feature name helpers
# ---------------------------------------------------------------------------

_BAND_NAME_SUFFIXES: dict[tuple[float, float], str] = {
    (1.0, 20.0): "band_1_20hz",
    (20.0, 80.0): "band_20_80hz",
    (80.0, 300.0): "band_80_300hz",
    (300.0, 1000.0): "band_300_1000hz",
    (1000.0, 2500.0): "band_1000_2500hz",
}


def _band_name(low: float, high: float) -> str:
    key = (float(low), float(high))
    if key in _BAND_NAME_SUFFIXES:
        return _BAND_NAME_SUFFIXES[key]
    return f"band_{low:.0f}_{high:.0f}hz"


def _base_feature_names(*, bands: tuple[tuple[float, float], ...]) -> list[str]:
    names = [
        "log_energy",
        "rms",
        "spectral_centroid",
        "spectral_bandwidth",
    ]
    for low, high in bands:
        names.append(_band_name(low, high))
    names.append("voicing_peakiness_80_300hz")
    return names


_CHANNEL_PREFIXES = (
    "left",
    "right",
    "mean",
    "abs_diff",
    "delta_mean",
    "delta_abs_diff",
)


def default_bone_feature_names(
    *,
    bands: tuple[tuple[float, float], ...] = DEFAULT_BANDS,
) -> list[str]:
    """Return ordered feature names for the binaural feature matrix.

    The concat order is:
    ``left, right, mean, abs_diff, delta_mean, delta_abs_diff``.

    Args:
        bands: Frequency band definitions (must match the bands used
            during feature extraction).
    """
    base = _base_feature_names(bands=bands)
    names: list[str] = []
    for prefix in _CHANNEL_PREFIXES:
        for bname in base:
            names.append(f"{prefix}_{bname}")
    return names
