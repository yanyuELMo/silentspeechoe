"""Tests for ``build_binaural_event_records`` using synthetic files only."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from silentspeechoe.data.dataset import build_binaural_event_records
from silentspeechoe.data.labels import EVENT_FIELDS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_events_csv(
    path: Path,
    rows: list[dict],
) -> None:
    """Write a minimal events.csv."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_bone_acc_csv(
    path: Path,
    num_samples: int = 300,
) -> None:
    """Write a tiny bone_acc CSV with timestamp and xyz columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("timestamp,bone_acc.x,bone_acc.y,bone_acc.z\n")
        t0 = 1_000_000_000
        for i in range(num_samples):
            t = t0 + i * 10_000  # 100 Hz approximate
            f.write(f"{t},{0.1 * i},{0.2 * i},{0.3 * i}\n")


def _make_event(
    subject_id: str = "sub_00",
    session_id: str = "002_2083961914",
    ear: str = "left",
    event_id: int = 0,
    sentence_type: str = "non_semantic",
    sentence_id: str = "nonsem_001",
    label_id: int = 0,
    domain: str = "normal",
    repeat_id: int = 1,
    start_time: float = 1.0,
    end_time: float = 2.0,
) -> dict:
    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "ear": ear,
        "event_id": event_id,
        "sentence_type": sentence_type,
        "sentence_id": sentence_id,
        "label_id": label_id,
        "domain": domain,
        "repeat_id": repeat_id,
        "start_time": start_time,
        "end_time": end_time,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildBinauralEventRecords:
    def test_pairs_matching_left_right(
        self,
        tmp_path: Path,
    ):
        """Left and right events with the same pairing key are matched."""
        events_csv = tmp_path / "events.csv"
        raw_dir = tmp_path / "raw"

        # Create two matching events (same key, different ears).
        left_ev = _make_event(ear="left", subject_id="sub_00", event_id=0)
        right_ev = _make_event(ear="right", subject_id="sub_00", event_id=0)
        _write_events_csv(events_csv, [left_ev, right_ev])

        # Create the matching raw bone_acc files.
        left_raw = (
            raw_dir
            / "left"
            / "00"
            / "non-semantic"
            / "sensor_002_2083961914__bone_acc.csv"
        )
        right_raw = (
            raw_dir
            / "right"
            / "00"
            / "non-semantic"
            / "sensor_002_2083961914__bone_acc.csv"
        )
        _write_bone_acc_csv(left_raw)
        _write_bone_acc_csv(right_raw)

        records = build_binaural_event_records(events_csv, raw_dir)

        assert len(records) == 1
        rec = records[0]
        assert rec["subject_id"] == "sub_00"
        assert rec["event_id"] == 0
        assert rec["label_id"] == 0
        assert rec["domain"] == "normal"
        assert rec["left_path"] == left_raw
        assert rec["right_path"] == right_raw
        assert rec["left_start_time"] == 1.0
        assert rec["right_start_time"] == 1.0

    def test_skips_when_raw_missing(
        self,
        tmp_path: Path,
    ):
        """Pair is skipped when either raw file does not exist."""
        events_csv = tmp_path / "events.csv"
        raw_dir = tmp_path / "raw"

        left_ev = _make_event(ear="left", subject_id="sub_00", event_id=0)
        right_ev = _make_event(ear="right", subject_id="sub_00", event_id=0)
        _write_events_csv(events_csv, [left_ev, right_ev])

        # Only create left raw, not right.
        left_raw = (
            raw_dir
            / "left"
            / "00"
            / "non-semantic"
            / "sensor_002_2083961914__bone_acc.csv"
        )
        _write_bone_acc_csv(left_raw)

        records = build_binaural_event_records(events_csv, raw_dir)
        assert len(records) == 0

    def test_multiple_subjects_and_events(
        self,
        tmp_path: Path,
    ):
        """Multiple subjects with different event slots all pair correctly."""
        events_csv = tmp_path / "events.csv"
        raw_dir = tmp_path / "raw"

        events = []
        for subj in ("00", "02"):
            for ev_id in range(2):
                sid = f"sub_{subj}"
                sess = f"00{subj}_1234567890"
                events.append(
                    _make_event(
                        subject_id=sid,
                        session_id=sess,
                        ear="left",
                        event_id=ev_id,
                        label_id=ev_id,
                    )
                )
                events.append(
                    _make_event(
                        subject_id=sid,
                        session_id=sess,
                        ear="right",
                        event_id=ev_id,
                        label_id=ev_id,
                    )
                )

                # Create matching raw files.
                for ear in ("left", "right"):
                    raw_path = (
                        raw_dir
                        / ear
                        / subj
                        / "non-semantic"
                        / f"sensor_{sess}__bone_acc.csv"
                    )
                    _write_bone_acc_csv(raw_path)

        _write_events_csv(events_csv, events)
        records = build_binaural_event_records(events_csv, raw_dir)

        assert len(records) == 4

        # Each subject/event_id combo should appear once.
        keys = {(r["subject_id"], r["event_id"]) for r in records}
        assert keys == {
            ("sub_00", 0),
            ("sub_00", 1),
            ("sub_02", 0),
            ("sub_02", 1),
        }

    def test_label_id_preserved(self, tmp_path: Path):
        """label_id from events.csv flows through to records."""
        events_csv = tmp_path / "events.csv"
        raw_dir = tmp_path / "raw"

        left_ev = _make_event(
            ear="left",
            subject_id="sub_00",
            event_id=5,
            label_id=23,
            sentence_type="semantic",
            sentence_id="sem_003",
        )
        right_ev = _make_event(
            ear="right",
            subject_id="sub_00",
            event_id=5,
            label_id=23,
            sentence_type="semantic",
            sentence_id="sem_003",
        )
        _write_events_csv(events_csv, [left_ev, right_ev])

        raw_path = (
            raw_dir / "left" / "00" / "semantic" / "sensor_002_2083961914__bone_acc.csv"
        )
        _write_bone_acc_csv(raw_path)
        raw_path = (
            raw_dir
            / "right"
            / "00"
            / "semantic"
            / "sensor_002_2083961914__bone_acc.csv"
        )
        _write_bone_acc_csv(raw_path)

        records = build_binaural_event_records(events_csv, raw_dir)

        assert len(records) == 1
        assert records[0]["label_id"] == 23
        assert records[0]["sentence_type"] == "semantic"
        assert records[0]["sentence_id"] == "sem_003"

    def test_domain_field_present(self, tmp_path: Path):
        """domain is preserved from events.csv (not renamed to speech_mode)."""
        events_csv = tmp_path / "events.csv"
        raw_dir = tmp_path / "raw"

        for domain in ("normal", "whisper", "silent"):
            left_ev = _make_event(
                ear="left",
                subject_id="sub_00",
                event_id=0,
                domain=domain,
                label_id=0,
            )
            right_ev = _make_event(
                ear="right",
                subject_id="sub_00",
                event_id=0,
                domain=domain,
                label_id=0,
            )
            _write_events_csv(events_csv, [left_ev, right_ev])

            raw_path = (
                raw_dir
                / "left"
                / "00"
                / "non-semantic"
                / "sensor_002_2083961914__bone_acc.csv"
            )
            _write_bone_acc_csv(raw_path)
            raw_path = (
                raw_dir
                / "right"
                / "00"
                / "non-semantic"
                / "sensor_002_2083961914__bone_acc.csv"
            )
            _write_bone_acc_csv(raw_path)

            records = build_binaural_event_records(events_csv, raw_dir)
            assert len(records) == 1
            assert records[0]["domain"] == domain
            assert "domain" in records[0]

    def test_events_csv_missing_columns_raises(
        self,
        tmp_path: Path,
    ):
        """A CSV without required columns raises ValueError."""
        bad_csv = tmp_path / "bad.csv"
        bad_csv.write_text("subject_id,ear,event_id\nsub_00,left,0\n")
        with pytest.raises(ValueError, match="missing required columns"):
            build_binaural_event_records(bad_csv, tmp_path / "raw")
