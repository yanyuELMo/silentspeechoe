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
    roc_auc_score,
    roc_curve,
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


def _binary_error_rates_at_threshold(
    y_binary: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute binary authentication error rates at one decision threshold."""

    accept = score >= threshold
    positives = y_binary == 1
    negatives = ~positives

    true_accept = np.sum(accept & positives)
    false_reject = np.sum((~accept) & positives)
    false_accept = np.sum(accept & negatives)
    true_reject = np.sum((~accept) & negatives)

    num_positive = true_accept + false_reject
    num_negative = false_accept + true_reject
    if num_positive == 0 or num_negative == 0:
        raise ValueError("Authentication metrics require positives and negatives")

    far = float(false_accept / num_negative)
    frr = float(false_reject / num_positive)
    accuracy = float((true_accept + true_reject) / y_binary.shape[0])
    return {
        "accuracy": accuracy,
        "far": far,
        "frr": frr,
    }


def _compute_one_vs_all_authentication_metrics(
    y_binary: np.ndarray,
    score: np.ndarray,
    *,
    far_target: float,
) -> dict[str, float]:
    """Compute one-vs-all authentication metrics for one subject."""

    positives = int(np.sum(y_binary == 1))
    negatives = int(np.sum(y_binary == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("Each subject must have at least one positive and negative")

    fpr, tpr, thresholds = roc_curve(y_binary, score)
    frr_curve = 1.0 - tpr

    eer_idx = int(np.argmin(np.abs(fpr - frr_curve)))
    eer = float((fpr[eer_idx] + frr_curve[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])
    eer_rates = _binary_error_rates_at_threshold(y_binary, score, eer_threshold)

    eligible = np.flatnonzero(fpr <= far_target)
    if eligible.size == 0:
        far_idx = 0
    else:
        # Among operating points satisfying FAR <= target, choose the one
        # with the lowest FRR (highest TPR).
        far_idx = int(eligible[np.argmax(tpr[eligible])])
    threshold_at_far = float(thresholds[far_idx])
    far_rates = _binary_error_rates_at_threshold(y_binary, score, threshold_at_far)

    return {
        "roc_auc": float(roc_auc_score(y_binary, score)),
        "eer": eer,
        "threshold": eer_threshold,
        "far": eer_rates["far"],
        "frr": eer_rates["frr"],
        "accuracy": eer_rates["accuracy"],
        "far_at_far_target": far_rates["far"],
        "frr_at_far_target": far_rates["frr"],
        "threshold_at_far_target": threshold_at_far,
        "num_positive": float(positives),
        "num_negative": float(negatives),
    }


def compute_authentication_metrics(
    y_true: Any,
    y_score: Any,
    *,
    top_k: int = 3,
    far_target: float = 0.01,
) -> dict[str, Any]:
    """Compute closed-set identification and one-vs-all authentication metrics.

    This function is intended for authentication experiments where each
    subject is evaluated as the positive class against all other subjects.
    The score matrix can be ArcFace logits or cosine similarities between
    query embeddings and enrolled subject templates.

    Args:
        y_true: Ground-truth subject indices, shape ``[N]``.
        y_score: Per-subject scores, shape ``[N, C]``.
        top_k: Value of *k* for top-k identification accuracy.
        far_target: FAR operating point used for ``frr_at_far_target``.
            The default ``0.01`` corresponds to FRR@FAR=1%.

    Returns:
        Dictionary with identification metrics at the top level and
        authentication metrics averaged across valid one-vs-all subjects.
        Per-subject one-vs-all metrics are available in ``by_subject``.

    Raises:
        ValueError: If shapes are inconsistent, labels are out of range, or
            ``far_target`` is outside ``[0, 1]``.
    """

    y_true = _as_1d_array(y_true, "y_true")
    y_score = _as_2d_array(y_score, "y_score")

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true has {y_true.shape[0]} samples, "
            f"y_score has {y_score.shape[0]} rows"
        )
    if not 0.0 <= far_target <= 1.0:
        raise ValueError(f"far_target must be in [0, 1], got {far_target}")

    num_classes = y_score.shape[1]
    y_pred = np.argmax(y_score, axis=1)
    _validate_class_indices(y_true, y_pred, num_classes)

    identification = compute_classification_metrics(
        y_true,
        y_pred,
        y_score,
        top_k=top_k,
    )

    by_subject: dict[str, dict[str, float]] = {}
    for subject_idx in range(num_classes):
        y_binary = (y_true == subject_idx).astype(np.int64)
        if y_binary.sum() == 0 or y_binary.sum() == y_binary.shape[0]:
            continue
        subject_metrics = _compute_one_vs_all_authentication_metrics(
            y_binary,
            y_score[:, subject_idx],
            far_target=far_target,
        )
        by_subject[str(subject_idx)] = subject_metrics

    if not by_subject:
        raise ValueError("No valid one-vs-all subjects for authentication metrics")

    auth_keys = (
        "roc_auc",
        "eer",
        "far",
        "frr",
        "accuracy",
        "far_at_far_target",
        "frr_at_far_target",
    )
    authentication = {
        key: float(np.mean([metrics[key] for metrics in by_subject.values()]))
        for key in auth_keys
    }
    authentication["far_target"] = float(far_target)
    authentication["num_subjects"] = float(len(by_subject))

    result: dict[str, Any] = {
        **identification,
        "roc_auc": authentication["roc_auc"],
        "eer": authentication["eer"],
        "far": authentication["far"],
        "frr": authentication["frr"],
        "far_at_far_target": authentication["far_at_far_target"],
        "frr_at_far_target": authentication["frr_at_far_target"],
        "authentication": authentication,
        "by_subject": by_subject,
    }
    if np.isclose(far_target, 0.01):
        result["frr_at_far_1pct"] = authentication["frr_at_far_target"]
    return result
