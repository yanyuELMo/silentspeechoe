"""Authentication metrics for verification decisions.

Authentication answers the question: is this sample from the claimed user?
Metrics here evaluate genuine-vs-impostor scores. The compact authentication
reporting set is EER and ROC-AUC; FAR/FRR details are retained for JSON
diagnostics and threshold analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from ._utils import as_1d_array, as_2d_array, validate_class_indices
from .metrics_identification import compute_classification_metrics


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
    """Compute one-vs-all authentication metrics from subject score matrix.

    The input score matrix can be ArcFace logits, cosine similarity against
    enrolled templates, or any larger-is-more-likely claimed-user score.
    Identification metrics are included for backward compatibility and for
    comparing "which user" performance against "is claimed user" performance.
    """
    y_true = as_1d_array(y_true, "y_true")
    y_score = as_2d_array(y_score, "y_score")

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true has {y_true.shape[0]} samples, "
            f"y_score has {y_score.shape[0]} rows"
        )
    if not 0.0 <= far_target <= 1.0:
        raise ValueError(f"far_target must be in [0, 1], got {far_target}")

    num_classes = y_score.shape[1]
    y_pred = np.argmax(y_score, axis=1)
    validate_class_indices(y_true, y_pred, num_classes)

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
        "authentication_summary": {
            "eer": authentication["eer"],
            "roc_auc": authentication["roc_auc"],
        },
        "authentication": authentication,
        "by_subject": by_subject,
    }
    if np.isclose(far_target, 0.01):
        result["frr_at_far_1pct"] = authentication["frr_at_far_target"]
    return result
