"""Build event metadata from OpenEarable 2.0 annotation workbooks.

The parser converts the left/right Excel label sheets into an event table
that can be saved as ``data/metadata/events.csv``.  It also keeps a
compatibility wrapper for the older dataset code.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_METADATA = Path("data/metadata")
_RAW_DATA = Path("data/raw")

_LABEL_FILES = {
    ("left", "nw"): _METADATA / "left" / "labels_left_nw.xlsx",
    ("left", "silent"): _METADATA / "left" / "labels_left_silent.xlsx",
    ("right", "nw"): _METADATA / "right" / "labels_right_nw.xlsx",
    ("right", "silent"): _METADATA / "right" / "labels_right_silent.xlsx",
}

EVENT_FIELDS = [
    "subject_id",
    "session_id",
    "ear",
    "event_id",
    "sentence_type",
    "sentence_id",
    "label_id",
    "domain",
    "repeat_id",
    "start_time",
    "end_time",
]

_SENTENCE_COUNT = 36
_FIRST_REPEATED_SENTENCE = 1
_LAST_REPEATED_SENTENCE = 10
_NON_SEMANTIC_END = 20
_FRAMES_PER_SECOND = 30.0
_SESSION_RE = re.compile(r"^sensor_(?P<session_id>.+)__bone_acc\.csv$")


def _is_empty(value: Any) -> bool:
    """Return whether a spreadsheet cell should be treated as empty."""

    return (
        value is None
        or pd.isna(value)
        or (isinstance(value, str) and value.strip() == "")
    )


def _time_to_seconds(minutes: Any, seconds: Any, frames: Any) -> float:
    """Convert spreadsheet ``minute, second, frame`` cells to seconds."""

    return float(minutes) * 60.0 + float(seconds) + float(frames) / _FRAMES_PER_SECOND


def _sentence_type(sentence_index: int) -> str:
    """Return the event-table sentence type for a 1-based sentence index."""

    if 1 <= sentence_index <= _NON_SEMANTIC_END:
        return "non_semantic"
    if _NON_SEMANTIC_END < sentence_index <= _SENTENCE_COUNT:
        return "semantic"
    raise ValueError(f"Unexpected sentence index: {sentence_index}")


def _raw_subset(sentence_index: int) -> str:
    """Return the raw-data subset folder for a 1-based sentence index."""

    if _sentence_type(sentence_index) == "non_semantic":
        return "non-semantic"
    return "semantic"


def _event_sentence_id(sentence_index: int) -> str:
    """Return the stable sentence ID string used in ``events.csv``."""

    if _sentence_type(sentence_index) == "non_semantic":
        return f"nonsem_{sentence_index:03d}"
    return f"sem_{sentence_index - _NON_SEMANTIC_END:03d}"


def _legacy_subject_id(subject_id: str) -> str:
    """Convert ``sub_00`` back to the raw folder subject name ``00``."""

    return subject_id.removeprefix("sub_")


def _session_id_from_raw(
    base_dir: Path,
    ear: str,
    subject_id: str,
    subset: str,
) -> str | None:
    """Read the session ID from the matching raw bone-acceleration filename."""

    raw_subject_id = _legacy_subject_id(subject_id)
    directory = base_dir / _RAW_DATA / ear / raw_subject_id / subset
    if not directory.exists():
        logger.debug("Raw subset directory does not exist: %s", directory)
        return None

    candidates = sorted(directory.glob("*__bone_acc.csv"))
    if not candidates:
        logger.debug("No bone_acc file found in %s", directory)
        return None
    if len(candidates) > 1:
        logger.warning(
            "Multiple bone_acc files found in %s; using %s",
            directory,
            candidates[0].name,
        )

    match = _SESSION_RE.match(candidates[0].name)
    if match is None:
        raise ValueError(f"Cannot parse session ID from {candidates[0]}")
    return match.group("session_id")


def _read_window(
    row: pd.Series,
    start_col: int,
    end_col: int,
) -> tuple[float, float] | None:
    """Read one start/end timestamp window from a spreadsheet row."""

    cells = (
        row.iloc[start_col],
        row.iloc[start_col + 1],
        row.iloc[start_col + 2],
        row.iloc[end_col],
        row.iloc[end_col + 1],
        row.iloc[end_col + 2],
    )
    if any(_is_empty(value) for value in cells):
        return None

    start_time = _time_to_seconds(cells[0], cells[1], cells[2])
    end_time = _time_to_seconds(cells[3], cells[4], cells[5])
    if end_time <= start_time:
        return None
    return start_time, end_time


def _append_event_if_present(
    events: list[dict[str, Any]],
    *,
    base_dir: Path,
    subject_id: str,
    ear: str,
    event_id: int,
    sentence_index: int,
    domain: str,
    repeat_id: int,
    row: pd.Series,
    start_col: int,
    end_col: int,
    skip_missing_raw: bool,
) -> None:
    """Append an event for a non-empty spreadsheet timestamp window."""

    window = _read_window(row, start_col=start_col, end_col=end_col)
    if window is None:
        return

    subset = _raw_subset(sentence_index)
    session_id = _session_id_from_raw(base_dir, ear, subject_id, subset)
    if session_id is None:
        if skip_missing_raw:
            logger.debug(
                "Skipping event without raw data: %s %s sentence %d",
                ear,
                subject_id,
                sentence_index,
            )
            return
        session_id = ""

    start_time, end_time = window
    events.append(
        {
            "subject_id": subject_id,
            "session_id": session_id,
            "ear": ear,
            "event_id": event_id,
            "sentence_type": _sentence_type(sentence_index),
            "sentence_id": _event_sentence_id(sentence_index),
            "label_id": sentence_index - 1,
            "domain": domain,
            "repeat_id": repeat_id,
            "start_time": start_time,
            "end_time": end_time,
        }
    )


def _row_for_sentence(sentence_index: int, mode_row_offset: int) -> int:
    """Return the zero-based row index for a sentence/mode row.

    Data starts on Excel row 4, which is zero-based index 3 in pandas.
    Each sentence occupies two rows in both NW and silent workbooks.
    """

    return 3 + (sentence_index - 1) * 2 + mode_row_offset


def _iter_event_slots(
    nw_sheet: pd.DataFrame,
    silent_sheet: pd.DataFrame,
    *,
    base_dir: Path,
    subject_id: str,
    ear: str,
    skip_missing_raw: bool,
) -> list[dict[str, Any]]:
    """Parse one subject sheet into ordered event records."""

    events: list[dict[str, Any]] = []
    event_id = 0

    for sentence_index in range(1, _SENTENCE_COUNT + 1):
        repeats = (
            (1, 2)
            if _FIRST_REPEATED_SENTENCE <= sentence_index <= _LAST_REPEATED_SENTENCE
            else (1,)
        )

        row_specs = (
            ("normal", nw_sheet, _row_for_sentence(sentence_index, 0)),
            ("whisper", nw_sheet, _row_for_sentence(sentence_index, 1)),
            ("silent", silent_sheet, _row_for_sentence(sentence_index, 0)),
        )
        for domain, sheet, row_idx in row_specs:
            if row_idx >= len(sheet):
                event_id += len(repeats)
                continue

            row = sheet.iloc[row_idx]
            for repeat_id in repeats:
                if repeat_id == 1:
                    start_col, end_col = 3, 6
                else:
                    start_col, end_col = 9, 12

                _append_event_if_present(
                    events,
                    base_dir=base_dir,
                    subject_id=subject_id,
                    ear=ear,
                    event_id=event_id,
                    sentence_index=sentence_index,
                    domain=domain,
                    repeat_id=repeat_id,
                    row=row,
                    start_col=start_col,
                    end_col=end_col,
                    skip_missing_raw=skip_missing_raw,
                )
                event_id += 1

    return events


def parse_label_events(
    base_dir: str | Path = ".",
    *,
    skip_missing_raw: bool = True,
) -> list[dict[str, Any]]:
    """Parse left/right label workbooks into event-table records.

    Args:
        base_dir: Project root.
        skip_missing_raw: If true, labels without a matching raw bone stream
            are skipped because their ``session_id`` cannot be resolved.

    Returns:
        Event records with fields matching :data:`EVENT_FIELDS`.
    """

    base = Path(base_dir)
    events: list[dict[str, Any]] = []

    for ear in ("left", "right"):
        nw_path = base / _LABEL_FILES[(ear, "nw")]
        silent_path = base / _LABEL_FILES[(ear, "silent")]
        if not nw_path.exists() or not silent_path.exists():
            logger.warning("Missing label workbook pair for %s ear", ear)
            continue

        nw_sheets = pd.read_excel(
            nw_path,
            sheet_name=None,
            header=None,
            engine="openpyxl",
        )
        silent_sheets = pd.read_excel(
            silent_path,
            sheet_name=None,
            header=None,
            engine="openpyxl",
        )

        for sheet_name in sorted(set(nw_sheets) & set(silent_sheets)):
            subject_id = f"sub_{sheet_name}"
            events.extend(
                _iter_event_slots(
                    nw_sheets[sheet_name],
                    silent_sheets[sheet_name],
                    base_dir=base,
                    subject_id=subject_id,
                    ear=ear,
                    skip_missing_raw=skip_missing_raw,
                )
            )

    logger.info("Parsed %d label events", len(events))
    return events


def write_events_csv(
    output_path: str | Path = _METADATA / "events.csv",
    *,
    base_dir: str | Path = ".",
    events: list[dict[str, Any]] | None = None,
    skip_missing_raw: bool = True,
) -> Path:
    """Write parsed label events to ``events.csv``."""

    base = Path(base_dir)
    path = Path(output_path)
    if not path.is_absolute():
        path = base / path
    path.parent.mkdir(parents=True, exist_ok=True)

    records = events
    if records is None:
        records = parse_label_events(base, skip_missing_raw=skip_missing_raw)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in EVENT_FIELDS})

    logger.info("Wrote %d events to %s", len(records), path)
    return path


def parse_all_labels(
    base_dir: str | Path = ".",
    *,
    skip_missing_raw: bool = True,
) -> list[dict[str, Any]]:
    """Parse labels in the legacy format used by the current dataset code."""

    legacy_records: list[dict[str, Any]] = []
    for event in parse_label_events(base_dir, skip_missing_raw=skip_missing_raw):
        sentence_index = int(event["label_id"]) + 1
        legacy_records.append(
            {
                "subject_id": _legacy_subject_id(str(event["subject_id"])),
                "side": event["ear"],
                "subset": _raw_subset(sentence_index),
                "sentence_id": sentence_index,
                "speech_mode": event["domain"],
                "repeat_id": int(event["repeat_id"]),
                "start_sec": float(event["start_time"]),
                "end_sec": float(event["end_time"]),
                "event_id": int(event["event_id"]),
                "session_id": event["session_id"],
            }
        )
    return legacy_records
