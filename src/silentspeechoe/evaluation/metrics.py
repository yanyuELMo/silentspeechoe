"""Evaluation metrics for closed-set sentence classification.

Provides per-sample and grouped (by speech mode) metric computation
for the silentspeechoe baseline experiments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    top_k_accuracy_score,
)


def _to_numpy(x: Any) -> np.ndarray:
    """Convert input to a NumPy array, handling PyTorch tensors.

    Args:
        x: NumPy array, PyTorch tensor, or list.

    Returns:
        NumPy array.
    """
    try:
        import torch

        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _as_1d_array(x: Any, name: str) -> np.ndarray:
    """Convert input to a one-dimensional NumPy array."""

    array = _to_numpy(x)
    if array.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {array.shape}")
    return array


def _as_2d_array(x: Any, name: str) -> np.ndarray:
    """Convert input to a two-dimensional NumPy array."""

    array = _to_numpy(x)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-D [N, C], got shape {array.shape}")
    return array


def _validate_class_indices(
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


def _fixed_label_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
) -> float:
    """Compute mean per-class recall over a fixed label set."""

    recalls = []
    for label in labels:
        positives = y_true == label
        if positives.any():
            recalls.append(float(np.mean(y_pred[positives] == label)))
        else:
            recalls.append(0.0)
    return float(np.mean(recalls))


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any,
    top_k: int = 3,
) -> dict[str, float]:
    """Compute standard classification metrics.

    Args:
        y_true: Ground-truth class indices, shape ``[N]``.
        y_pred: Predicted class indices (argmax), shape ``[N]``.
        y_score: Raw logits or class probabilities, shape ``[N, C]``.
            Softmax is *not* required — ranking logits gives the same
            top‑k result.
        top_k: Value of *k* for top‑k accuracy (default 3).

    Returns:
        Dictionary with keys ``accuracy``, ``macro_f1``,
        ``balanced_accuracy``, and ``top3_accuracy``.

    Raises:
        ValueError: If input shapes are inconsistent or ``top_k`` exceeds
            the number of classes.
    """
    y_true = _as_1d_array(y_true, "y_true")
    y_pred = _as_1d_array(y_pred, "y_pred")
    y_score = _as_2d_array(y_score, "y_score")

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true has {y_true.shape[0]} samples, "
            f"y_pred has {y_pred.shape[0]} samples"
        )
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true has {y_true.shape[0]} samples, "
            f"y_score has {y_score.shape[0]} rows"
        )

    num_classes = y_score.shape[1]
    _validate_class_indices(y_true, y_pred, num_classes)
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if top_k > num_classes:
        raise ValueError(f"top_k ({top_k}) must not exceed num_classes ({num_classes})")

    labels = list(range(num_classes))

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0, labels=labels)
    )
    balanced_acc = _fixed_label_balanced_accuracy(y_true, y_pred, labels)
    top3 = float(top_k_accuracy_score(y_true, y_score, k=top_k, labels=labels))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "top3_accuracy": top3,
    }


def compute_grouped_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any,
    groups: Any,
    top_k: int = 3,
) -> dict[str, dict[str, float]]:
    """Compute metrics overall and per group (e.g. speech mode).

    Args:
        y_true: Ground-truth class indices, shape ``[N]``.
        y_pred: Predicted class indices, shape ``[N]``.
        y_score: Raw logits or class probabilities, shape ``[N, C]``.
        groups: Per-sample group labels, shape ``[N]``.
            Typical values: ``"normal"``, ``"whisper"``, ``"silent"``.
        top_k: Value of *k* for top‑k accuracy (default 3).

    Returns:
        A dictionary::

            {
                "overall": {...},
                "by_group": {
                    "normal":  {...},
                    "whisper": {...},
                    "silent":  {...},
                    ...
                },
            }
    """
    y_true = _as_1d_array(y_true, "y_true")
    y_pred = _as_1d_array(y_pred, "y_pred")
    y_score = _as_2d_array(y_score, "y_score")
    groups = _as_1d_array(groups, "groups")

    if groups.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true has {y_true.shape[0]} samples, "
            f"groups has {groups.shape[0]} samples"
        )

    overall = compute_classification_metrics(y_true, y_pred, y_score, top_k=top_k)

    by_group: dict[str, dict[str, float]] = {}
    unique_groups = sorted(set(groups))
    for group in unique_groups:
        mask = groups == group
        if not mask.any():
            continue  # safety; shouldn't happen after set()
        by_group[str(group)] = compute_classification_metrics(
            y_true[mask],
            y_pred[mask],
            y_score[mask],
            top_k=top_k,
        )

    return {"overall": overall, "by_group": by_group}
