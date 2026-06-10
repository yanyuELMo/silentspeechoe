"""Tests for the baseline pipeline: labels, dataset, model, collate."""

from __future__ import annotations

import pytest

from silentspeechoe.data.labels import (
    EVENT_FIELDS,
    parse_all_labels,
    parse_label_events,
    write_events_csv,
)

# ---------------------------------------------------------------------------
# Session‑scoped fixtures (expensive ops cached once)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def all_labels():
    """Parse labels once and reuse across tests."""
    return parse_all_labels(".")


@pytest.fixture(scope="session")
def binaural_records():
    """Build binaural records once and reuse across tests."""
    pytest.importorskip("torch")
    from silentspeechoe.data.dataset import build_binaural_records

    return build_binaural_records(".")


@pytest.fixture(scope="session")
def label_events():
    """Parse event-table labels once and reuse across tests."""
    return parse_label_events(".")


# ---------------------------------------------------------------------------
# Label parser tests
# ---------------------------------------------------------------------------


def test_label_parser_returns_records(all_labels):
    """parse_all_labels should return a non‑empty list of dicts."""
    records = all_labels
    assert isinstance(records, list)
    assert len(records) > 0
    required_keys = {
        "subject_id",
        "side",
        "subset",
        "sentence_id",
        "speech_mode",
        "repeat_id",
        "start_sec",
        "end_sec",
    }
    for r in records:
        assert required_keys <= set(r)


def test_event_parser_returns_events(label_events):
    """parse_label_events should return records with the events.csv fields."""
    assert isinstance(label_events, list)
    assert len(label_events) > 0
    for event in label_events:
        assert set(EVENT_FIELDS) <= set(event)


def test_event_parser_first_sentence_order(label_events):
    """Event IDs should follow normal, whisper, silent slots for a sentence."""
    events = [
        event
        for event in label_events
        if event["subject_id"] == "sub_00"
        and event["ear"] == "left"
        and event["sentence_id"] == "nonsem_001"
    ]
    assert [(e["event_id"], e["domain"], e["repeat_id"]) for e in events] == [
        (0, "normal", 1),
        (1, "normal", 2),
        (2, "whisper", 1),
        (3, "whisper", 2),
        (4, "silent", 1),
        (5, "silent", 2),
    ]


def test_event_parser_left_right_event_alignment(label_events):
    """Left and right ears should use the same event IDs for matching slots."""
    left = {
        (e["event_id"], e["domain"], e["repeat_id"])
        for e in label_events
        if e["subject_id"] == "sub_00"
        and e["ear"] == "left"
        and e["sentence_id"] == "nonsem_001"
    }
    right = {
        (e["event_id"], e["domain"], e["repeat_id"])
        for e in label_events
        if e["subject_id"] == "sub_00"
        and e["ear"] == "right"
        and e["sentence_id"] == "nonsem_001"
    }
    assert left == right


def test_event_parser_sentence_metadata(label_events):
    """Sentence IDs and label IDs should match the 36-class convention."""
    nonsem = next(
        e
        for e in label_events
        if e["subject_id"] == "sub_00"
        and e["ear"] == "left"
        and e["sentence_id"] == "nonsem_001"
    )
    semantic = next(
        e
        for e in label_events
        if e["subject_id"] == "sub_00"
        and e["ear"] == "left"
        and e["sentence_id"] == "sem_001"
    )
    assert nonsem["sentence_type"] == "non_semantic"
    assert nonsem["label_id"] == 0
    assert semantic["sentence_type"] == "semantic"
    assert semantic["label_id"] == 20


def test_write_events_csv(tmp_path, label_events):
    """Event records should be writable with the expected CSV header."""
    output_path = tmp_path / "events.csv"
    written = write_events_csv(output_path, events=label_events[:3])

    lines = written.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == EVENT_FIELDS
    assert len(lines) == 4


def test_label_parser_has_all_speech_modes(all_labels):
    """Records should contain normal, whisper, and silent modes."""
    modes = {r["speech_mode"] for r in all_labels}
    assert modes >= {"normal", "whisper", "silent"}


def test_label_parser_subset_mapping(all_labels):
    """Sentence 1‑20 → non-semantic, 21‑36 → semantic."""
    for r in all_labels:
        sid = r["sentence_id"]
        if 1 <= sid <= 20:
            assert r["subset"] == "non-semantic"
        elif 21 <= sid <= 36:
            assert r["subset"] == "semantic"
        else:
            pytest.fail(f"Unexpected sentence_id: {sid}")


def test_label_parser_time_positive(all_labels):
    """Every window should have end_sec > start_sec."""
    for r in all_labels:
        assert r["end_sec"] > r["start_sec"], (
            f"Invalid window for subj={r['subject_id']} "
            f"sent={r['sentence_id']} mode={r['speech_mode']}"
        )


def test_label_parser_repeat_ids(all_labels):
    """Sentences 1‑10 may have repeat_id=2; 11‑36 must have repeat_id=1."""
    for r in all_labels:
        if 1 <= r["sentence_id"] <= 10:
            assert r["repeat_id"] in (1, 2)
        else:
            assert r["repeat_id"] == 1


def test_label_parser_empty_windows_skipped(all_labels):
    """No record should have missing start_sec or end_sec."""
    for r in all_labels:
        assert r["start_sec"] is not None
        assert r["end_sec"] is not None


# ---------------------------------------------------------------------------
# Dataset building tests
# ---------------------------------------------------------------------------


def test_build_binaural_records_train_val_split(binaural_records):
    """Validation subjects 07/10/13/17 should appear only in val."""
    train_recs, val_recs = binaural_records
    val_subjects = {"07", "10", "13", "17"}

    train_subjects = {r["subject_id"] for r in train_recs}
    val_subjects_found = {r["subject_id"] for r in val_recs}

    # No overlap
    assert train_subjects & val_subjects_found == set()
    # Val subjects must be in val
    for vs in val_subjects:
        if vs in val_subjects_found:
            assert vs not in train_subjects


def test_build_binaural_records_nonempty(binaural_records):
    """Both train and val records should be non‑empty."""
    train_recs, val_recs = binaural_records
    assert len(train_recs) > 0, "Train records should not be empty"
    assert len(val_recs) > 0, "Val records should not be empty"


def test_build_binaural_records_fields(binaural_records):
    """Each record should have the expected binaural keys."""
    train_recs, val_recs = binaural_records
    for rec in train_recs + val_recs:
        for key in (
            "subject_id",
            "event_id",
            "left_session_id",
            "right_session_id",
            "sentence_id",
            "speech_mode",
            "left_start_sec",
            "right_start_sec",
            "left_path",
            "right_path",
        ):
            assert key in rec, f"Missing key {key}"


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


def test_bone_binaural_cnn_forward_shape():
    """Model should output [B, 36] for input [B, 2, T]."""
    torch = pytest.importorskip("torch")
    from silentspeechoe.models.bone_cnn import BoneBinauralCNN

    model = BoneBinauralCNN(in_channels=2, num_classes=36)
    B, C, T = 4, 2, 1000
    x = torch.randn(B, C, T)
    out = model(x)
    assert out.shape == (B, 36)


def test_bone_binaural_cnn_train_mode():
    """Model should be trainable (gradients flow)."""
    torch = pytest.importorskip("torch")
    from silentspeechoe.models.bone_cnn import BoneBinauralCNN

    model = BoneBinauralCNN(in_channels=2, num_classes=36)
    model.train()
    x = torch.randn(2, 2, 500)
    y = torch.randint(0, 36, (2,))
    loss = torch.nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


def test_bone_binaural_cnn_variable_length():
    """Model should handle different input lengths."""
    torch = pytest.importorskip("torch")
    from silentspeechoe.models.bone_cnn import BoneBinauralCNN

    model = BoneBinauralCNN(in_channels=2, num_classes=36)
    for T in (100, 500, 2000):
        x = torch.randn(1, 2, T)
        out = model(x)
        assert out.shape == (1, 36)


# ---------------------------------------------------------------------------
# Collate tests
# ---------------------------------------------------------------------------


def test_pad_collate_same_length():
    """Samples with same length should not be padded."""
    torch = pytest.importorskip("torch")
    from silentspeechoe.data.collate import pad_collate

    items = [
        {"x": torch.randn(2, 100), "y": i, "speech_mode": "normal", "subject_id": "00"}
        for i in range(4)
    ]
    batch = pad_collate(items)
    assert batch["x"].shape == (4, 2, 100)
    assert batch["y"].tolist() == [0, 1, 2, 3]
    assert batch["lengths"].tolist() == [100, 100, 100, 100]
    assert batch["speech_mode"] == ["normal"] * 4


def test_pad_collate_variable_length():
    """Shorter samples should be zero‑padded to the max length."""
    torch = pytest.importorskip("torch")
    from silentspeechoe.data.collate import pad_collate

    items = [
        {"x": torch.randn(2, 100), "y": 0, "speech_mode": "normal", "subject_id": "00"},
        {
            "x": torch.randn(2, 200),
            "y": 1,
            "speech_mode": "whisper",
            "subject_id": "00",
        },
        {"x": torch.randn(2, 150), "y": 2, "speech_mode": "silent", "subject_id": "00"},
    ]
    batch = pad_collate(items)
    assert batch["x"].shape == (3, 2, 200)
    assert batch["lengths"].tolist() == [100, 200, 150]
    # Check that padded region is zero
    assert torch.all(batch["x"][0, :, 100:] == 0)
    # Check that full-length sample has no trailing zeros at its actual content
    assert not torch.all(batch["x"][1, :, :200] == 0)


def test_pad_collate_empty_batch():
    """Empty batch should return empty tensors."""
    pytest.importorskip("torch")
    from silentspeechoe.data.collate import pad_collate

    batch = pad_collate([])
    assert batch["x"].numel() == 0
    assert batch["y"].numel() == 0
