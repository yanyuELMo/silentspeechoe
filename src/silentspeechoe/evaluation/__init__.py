"""Evaluation package — metrics and plotting helpers."""

from __future__ import annotations

from .metrics import (
    compute_authentication_metrics,
    compute_classification_metrics,
    compute_grouped_classification_metrics,
)

__all__ = [
    "compute_authentication_metrics",
    "compute_classification_metrics",
    "compute_grouped_classification_metrics",
]
