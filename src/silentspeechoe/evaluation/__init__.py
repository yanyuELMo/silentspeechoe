"""Evaluation package: identification, authentication, attack metrics, and plots."""

from __future__ import annotations

from .metrics_attack import compute_all_attempt_asr, compute_attack_success_rate
from .metrics_authentication import compute_authentication_metrics
from .metrics_identification import (
    compute_classification_metrics,
    compute_dir_at_fpir_leave_one_user_out,
    compute_grouped_classification_metrics,
)

__all__ = [
    "compute_all_attempt_asr",
    "compute_attack_success_rate",
    "compute_authentication_metrics",
    "compute_classification_metrics",
    "compute_dir_at_fpir_leave_one_user_out",
    "compute_grouped_classification_metrics",
]
