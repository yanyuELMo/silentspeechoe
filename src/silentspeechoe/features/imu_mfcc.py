"""MFCC feature extraction for IMU sensor sequences.

Converts each of the 9 IMU channels into 13 MFCCs, pools them with
mean and standard deviation, and concatenates the result into a single
fixed‑length vector of 234 dimensions (9 × 13 × 2).

Uses only NumPy / SciPy — no audio‑specific dependencies.
"""

from __future__ import annotations

import numpy as np
from scipy.fftpack import dct


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Convert Hz to mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """Convert mel scale back to Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(
    n_fft: int,
    sample_rate: float,
    n_mels: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Build a mel‑spaced triangular filterbank matrix.

    Args:
        n_fft: FFT size (number of frequency bins).
        sample_rate: Sampling rate in Hz.
        n_mels: Number of mel filterbank channels.
        fmin: Lowest frequency in Hz.
        fmax: Highest frequency in Hz (clamped to Nyquist).

    Returns:
        Float32 array of shape ``[n_mels, n_fft // 2 + 1]``.
    """
    nyquist = sample_rate / 2.0
    fmax = min(fmax, nyquist)

    mel_low = hz_to_mel(np.array(fmin))
    mel_high = hz_to_mel(np.array(fmax))
    mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    bin_indices = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_fft // 2)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left = bin_indices[m - 1]
        center = bin_indices[m]
        right = bin_indices[m + 1]

        if center > left:
            filters[m - 1, left:center] = (np.arange(left, center) - left) / (
                center - left
            )
        if right > center:
            filters[m - 1, center:right] = (right - np.arange(center, right)) / (
                right - center
            )

    return filters


def compute_mfcc(
    signal: np.ndarray,
    sample_rate: float = 200.0,
    *,
    n_mfcc: int = 13,
    n_mels: int = 20,
    frame_length: int = 50,
    hop_length: int = 10,
    fmin: float = 0.5,
    fmax: float = 90.0,
    n_fft: int | None = None,
) -> np.ndarray:
    """Compute MFCCs for a 1‑D signal.

    Args:
        signal: 1‑D float array of shape ``[T]``.
        sample_rate: Sample rate in Hz (default 200).
        n_mfcc: Number of MFCC coefficients (default 13, excludes C0).
        n_mels: Number of mel filterbank channels.
        frame_length: Frame length in samples (default 50 → 0.25 s).
        hop_length: Hop length in samples (default 10 → 0.05 s).
        fmin: Lowest frequency in Hz.
        fmax: Highest frequency in Hz.
        n_fft: FFT size (defaults to next power of two >= frame_length).

    Returns:
        Float32 array of shape ``[n_mfcc, num_frames]``, or
        ``[n_mfcc, 0]`` when the signal is too short for one frame.
    """
    T = signal.shape[0]

    if T < frame_length:
        return np.empty((n_mfcc, 0), dtype=np.float32)

    if n_fft is None:
        n_fft = 2 ** int(np.ceil(np.log2(frame_length)))

    # ---- framing ---------------------------------------------------------
    num_frames = max(1, (T - frame_length) // hop_length + 1)
    frames = np.zeros((num_frames, frame_length), dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_length
        frames[i] = signal[start : start + frame_length]

    # ---- windowing -------------------------------------------------------
    window = np.hamming(frame_length).astype(np.float32)
    frames = frames * window[None, :]

    # ---- FFT magnitude ---------------------------------------------------
    spec = np.abs(np.fft.rfft(frames, n=n_fft))  # [num_frames, n_fft//2+1]

    # ---- mel filterbank --------------------------------------------------
    mel_fb = _mel_filterbank(n_fft, sample_rate, n_mels, fmin, fmax)
    mel_spec = spec @ mel_fb.T  # [num_frames, n_mels]

    # ---- log -------------------------------------------------------------
    mel_spec = np.log(np.maximum(mel_spec, 1e-10))

    # ---- DCT → MFCC ------------------------------------------------------
    # scipy DCT type-2 on each row; keep coefficients 1..n_mfcc (skip C0).
    mfcc_full = dct(mel_spec, type=2, norm="ortho")  # [num_frames, n_mels]
    mfcc = mfcc_full[:, 1 : n_mfcc + 1].T.astype(np.float32)  # [n_mfcc, num_frames]

    return mfcc


def extract_imu_mfcc_features(
    x: np.ndarray,
    sample_rate: float = 200.0,
    *,
    n_mfcc: int = 13,
    n_mels: int = 20,
    frame_length: int = 50,
    hop_length: int = 10,
    fmin: float = 0.5,
    fmax: float = 90.0,
    use_delta: bool = False,
    n_fft: int | None = None,
) -> np.ndarray:
    """Extract pooled MFCC features from a 9‑channel IMU window.

    For each channel:
        1. Compute MFCCs → ``[n_mfcc, num_frames]``.
        2. Mean over frames → ``[n_mfcc]``.
        3. Std over frames → ``[n_mfcc]``.
        4. Concatenate → ``[2 * n_mfcc]`` per channel.

    The per‑channel features are concatenated into a flat vector.

    Args:
        x: Float32 array of shape ``[C, T]`` (C=9 for full IMU).
        sample_rate: Sample rate in Hz.
        n_mfcc: Number of MFCC coefficients.
        n_mels: Mel filterbank channels.
        frame_length: Frame length in samples.
        hop_length: Hop length in samples.
        fmin: Lowest mel frequency.
        fmax: Highest mel frequency.
        use_delta: If ``True``, also compute delta and delta‑delta
            (not implemented yet).
        n_fft: FFT size.

    Returns:
        Float32 array of shape ``[feature_dim]`` where
        ``feature_dim = C * 2 * n_mfcc`` (default 234).
    """
    C = x.shape[0]
    per_channel_dim = 2 * n_mfcc  # mean + std
    features = np.empty(C * per_channel_dim, dtype=np.float32)

    for c in range(C):
        mfcc = compute_mfcc(
            x[c],
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            n_mels=n_mels,
            frame_length=frame_length,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            n_fft=n_fft,
        )
        if mfcc.shape[1] > 0:
            feats = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1, ddof=0)])
        else:
            feats = np.zeros(per_channel_dim, dtype=np.float32)
        features[c * per_channel_dim : (c + 1) * per_channel_dim] = feats

    return features


def feature_dim(
    num_channels: int = 9,
    n_mfcc: int = 13,
    use_delta: bool = False,
) -> int:
    """Return the output feature dimension for the given parameters."""
    base = num_channels * 2 * n_mfcc
    if use_delta:
        base *= 3  # static + delta + delta-delta (future)
    return base
