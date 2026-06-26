"""Backward-compatible metric exports.

New code should prefer importing from:

* ``metrics_identification`` for "which user is this?" metrics.
* ``metrics_authentication`` for "is this the claimed user?" metrics.
* ``metrics_attack`` for future attack-specific metrics.
"""

from __future__ import annotations

from .metrics_attack import compute_attack_success_rate
from .metrics_authentication import compute_authentication_metrics
from .metrics_identification import (
    compute_classification_metrics,
    compute_dir_at_fpir_leave_one_user_out,
    compute_grouped_classification_metrics,
)

__all__ = [
    "compute_attack_success_rate",
    "compute_authentication_metrics",
    "compute_classification_metrics",
    "compute_dir_at_fpir_leave_one_user_out",
    "compute_grouped_classification_metrics",
]
