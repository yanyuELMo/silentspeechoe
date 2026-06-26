"""Feature extraction package.

Sub‑modules provide reusable feature transforms for sensor streams
(bone‑acc, IMU, etc.) used across the silent‑speech pipeline.
"""

from __future__ import annotations

from silentspeechoe.features.bone_acc import (
    DEFAULT_BANDS,
    DEFAULT_PER_EAR_FEATURE_DIM,
    batched_zscore,
    compute_delta,
    compute_frame_spectrum_features,
    default_bone_feature_names,
    estimate_sampling_rate,
    extract_binaural_bone_features,
    extract_single_ear_bone_features,
    frame_signal,
)
from silentspeechoe.features.imu_mfcc import (
    compute_mfcc,
    extract_imu_mfcc_features,
    extract_imu_mfcc_sequence,
    feature_dim,
)

__all__ = [
    "DEFAULT_BANDS",
    "DEFAULT_PER_EAR_FEATURE_DIM",
    "batched_zscore",
    "compute_delta",
    "compute_frame_spectrum_features",
    "compute_mfcc",
    "default_bone_feature_names",
    "estimate_sampling_rate",
    "extract_binaural_bone_features",
    "extract_imu_mfcc_features",
    "extract_imu_mfcc_sequence",
    "extract_single_ear_bone_features",
    "feature_dim",
    "frame_signal",
]
