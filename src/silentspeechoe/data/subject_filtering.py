"""Global subject filtering helpers for excluded participants."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

EXCLUDED_SUBJECT_IDS = frozenset({"sub_26", "sub_51"})


def canonical_subject_id(subject_id: Any) -> str:
    """Return a canonical ``sub_XX`` subject ID when possible."""
    value = str(subject_id)
    if value.startswith("sub_"):
        return value
    if value.isdigit():
        return f"sub_{value.zfill(2)}"
    return value


def is_excluded_subject_id(subject_id: Any) -> bool:
    """Return ``True`` when a subject is globally excluded."""
    return canonical_subject_id(subject_id) in EXCLUDED_SUBJECT_IDS


def filter_subject_dataframe(
    df: pd.DataFrame,
    *,
    subject_column: str = "subject_id",
) -> pd.DataFrame:
    """Return a copy of *df* with excluded subjects removed."""
    if subject_column not in df.columns:
        raise ValueError(f"DataFrame is missing subject column: {subject_column}")
    keep_mask = ~df[subject_column].map(is_excluded_subject_id)
    return df.loc[keep_mask].copy()


def filter_subject_records(
    records: Iterable[dict[str, Any]],
    *,
    subject_key: str = "subject_id",
) -> list[dict[str, Any]]:
    """Return only records whose subject is not globally excluded."""
    return [
        record
        for record in records
        if not is_excluded_subject_id(record.get(subject_key, ""))
    ]
