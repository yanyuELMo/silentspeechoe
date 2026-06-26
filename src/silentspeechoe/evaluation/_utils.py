"""Shared helpers for evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy(x: Any) -> np.ndarray:
    """Convert input to a NumPy array, handling PyTorch tensors."""
    try:
        import torch

        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def as_1d_array(x: Any, name: str) -> np.ndarray:
    """Convert input to a one-dimensional NumPy array."""
    array = to_numpy(x)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    return array


def as_2d_array(x: Any, name: str) -> np.ndarray:
    """Convert input to a two-dimensional NumPy array."""
    array = to_numpy(x)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D [N, C], got shape {array.shape}")
    return array


def validate_class_indices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> None:
    """Ensure class indices fit the score matrix class dimension."""
    if num_classes <= 0:
        raise ValueError("y_score must contain at least one class column")

    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        invalid = (values < 0) | (values >= num_classes)
        if invalid.any():
            raise ValueError(f"{name} contains labels outside [0, {num_classes - 1}]")
