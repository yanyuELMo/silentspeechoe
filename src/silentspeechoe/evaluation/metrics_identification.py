"""Identification metrics for subject classification.

Identification answers the question: which enrolled identity does this sample
belong to? Metrics here evaluate closed-set top-1 prediction and open-set
identification with leave-one-user-out unknown folds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

from ._utils import as_1d_array, as_2d_array, validate_class_indices


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


def _safe_fpir_threshold(unknown_scores: np.ndarray, fpir_target: float) -> float:
    """Return the lowest conservative threshold with FPIR <= target."""
    if unknown_scores.size == 0:
        raise ValueError("Cannot select FPIR threshold without unknown scores")

    allowed_false_positives = int(np.floor(fpir_target * unknown_scores.size))
    if allowed_false_positives >= unknown_scores.size:
        return float("-inf")

    sorted_scores = np.sort(unknown_scores.astype(np.float64))[::-1]
    boundary_score = float(sorted_scores[allowed_false_positives])
    return float(np.nextafter(boundary_score, np.inf))


def compute_dir_at_fpir_leave_one_user_out(
    y_true: Any,
    y_score: Any,
    *,
    fpir_target: float = 0.001,
) -> dict[str, Any]:
    """Compute DIR@FPIR with leave-one-user-out unknown folds.

    For each fold, one subject is removed from the enrolled set and used as the
    unknown class. The threshold is selected from that fold's unknown max-score
    distribution so that actual FPIR is conservatively no larger than
    ``fpir_target``. DIR is then the fraction of enrolled-user probes that are
    both accepted and correctly identified.
    """
    y_true = as_1d_array(y_true, "y_true")
    y_score = as_2d_array(y_score, "y_score")

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true has {y_true.shape[0]} samples, "
            f"y_score has {y_score.shape[0]} rows"
        )
    if not 0.0 <= fpir_target <= 1.0:
        raise ValueError(f"fpir_target must be in [0, 1], got {fpir_target}")

    num_classes = y_score.shape[1]
    present_labels = sorted(int(label) for label in set(y_true.tolist()))
    validate_class_indices(y_true, np.asarray(present_labels), num_classes)

    fold_metrics = []
    for unknown_label in present_labels:
        enrolled_labels = [
            label for label in present_labels if label != int(unknown_label)
        ]
        if not enrolled_labels:
            continue

        enrolled = np.asarray(enrolled_labels, dtype=np.int64)
        fold_scores = y_score[:, enrolled].astype(np.float64)
        max_scores = np.max(fold_scores, axis=1)
        predicted_labels = enrolled[np.argmax(fold_scores, axis=1)]

        unknown_mask = y_true == int(unknown_label)
        known_mask = ~unknown_mask
        if not unknown_mask.any() or not known_mask.any():
            continue

        threshold = _safe_fpir_threshold(max_scores[unknown_mask], fpir_target)
        unknown_accept = max_scores[unknown_mask] >= threshold
        known_accept = max_scores[known_mask] >= threshold
        known_correct = predicted_labels[known_mask] == y_true[known_mask]

        fpir = float(np.mean(unknown_accept))
        dir_value = float(np.mean(known_accept & known_correct))
        fold_metrics.append(
            {
                "unknown_label": int(unknown_label),
                "threshold": float(threshold),
                "fpir": fpir,
                "dir": dir_value,
                "num_unknown": float(np.sum(unknown_mask)),
                "num_known": float(np.sum(known_mask)),
                "num_correct_detected_identified": float(
                    np.sum(known_accept & known_correct)
                ),
            }
        )

    if not fold_metrics:
        return {
            "dir": float("nan"),
            "fpir": float("nan"),
            "fpir_target": float(fpir_target),
            "num_folds": 0.0,
            "folds": [],
        }

    return {
        "dir": float(np.mean([fold["dir"] for fold in fold_metrics])),
        "fpir": float(np.mean([fold["fpir"] for fold in fold_metrics])),
        "fpir_target": float(fpir_target),
        "num_folds": float(len(fold_metrics)),
        "folds": fold_metrics,
    }


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any,
    top_k: int = 3,
    fpir_target: float = 0.001,
) -> dict[str, Any]:
    """Compute identification metrics.

    The compact reporting set is top-1 accuracy, macro-F1, and
    DIR@FPIR=0.1%. Legacy balanced/top-k fields are retained for JSON
    compatibility with older experiment artifacts.
    """
    y_true = as_1d_array(y_true, "y_true")
    y_pred = as_1d_array(y_pred, "y_pred")
    y_score = as_2d_array(y_score, "y_score")

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
    validate_class_indices(y_true, y_pred, num_classes)
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
    open_set = compute_dir_at_fpir_leave_one_user_out(
        y_true,
        y_score,
        fpir_target=fpir_target,
    )
    dir_at_fpir = float(open_set["dir"])

    return {
        "top1_accuracy": accuracy,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "dir_at_fpir_target": dir_at_fpir,
        "dir_at_fpir_0p1pct": dir_at_fpir if np.isclose(fpir_target, 0.001) else np.nan,
        "fpir_at_dir_threshold": float(open_set["fpir"]),
        "fpir_target": float(fpir_target),
        "open_set_identification": open_set,
        "balanced_accuracy": balanced_acc,
        "top3_accuracy": top3,
    }


def compute_grouped_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any,
    groups: Any,
    top_k: int = 3,
    fpir_target: float = 0.001,
) -> dict[str, dict[str, Any]]:
    """Compute identification metrics overall and per group."""
    y_true = as_1d_array(y_true, "y_true")
    y_pred = as_1d_array(y_pred, "y_pred")
    y_score = as_2d_array(y_score, "y_score")
    groups = as_1d_array(groups, "groups")

    if groups.shape[0] != y_true.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true has {y_true.shape[0]} samples, "
            f"groups has {groups.shape[0]} samples"
        )

    overall = compute_classification_metrics(
        y_true,
        y_pred,
        y_score,
        top_k=top_k,
        fpir_target=fpir_target,
    )

    by_group: dict[str, dict[str, float]] = {}
    unique_groups = sorted(set(groups))
    for group in unique_groups:
        mask = groups == group
        if not mask.any():
            continue
        by_group[str(group)] = compute_classification_metrics(
            y_true[mask],
            y_pred[mask],
            y_score[mask],
            top_k=top_k,
            fpir_target=fpir_target,
        )

    return {"overall": overall, "by_group": by_group}
