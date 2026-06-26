"""Tests for 18-channel binaural IMU window precomputation helpers."""

from __future__ import annotations

import torch

from scripts.precompute_imu_binaural_windows import (
    build_binaural_pairs,
    make_binaural_window,
)


def test_binaural_pairs_use_same_subject_and_event() -> None:
    records = [
        {
            "subject_id": "sub_01",
            "event_id": 1,
            "ear": "left",
            "sentence_type": "non_semantic",
            "sentence_id": "nonsem_001",
            "label_id": 0,
            "domain": "normal",
            "repeat_id": 1,
        },
        {
            "subject_id": "sub_02",
            "event_id": 1,
            "ear": "right",
            "sentence_type": "non_semantic",
            "sentence_id": "nonsem_001",
            "label_id": 0,
            "domain": "normal",
            "repeat_id": 1,
        },
        {
            "subject_id": "sub_01",
            "event_id": 2,
            "ear": "left",
            "sentence_type": "semantic",
            "sentence_id": "sem_001",
            "label_id": 1,
            "domain": "silent",
            "repeat_id": 1,
        },
        {
            "subject_id": "sub_01",
            "event_id": 2,
            "ear": "right",
            "sentence_type": "semantic",
            "sentence_id": "sem_001",
            "label_id": 1,
            "domain": "silent",
            "repeat_id": 1,
        },
    ]

    pairs = build_binaural_pairs(records)

    assert len(pairs) == 1
    assert pairs[0]["subject_id"] == "sub_01"
    assert pairs[0]["event_id"] == 2
    assert pairs[0]["left_index"] == 2
    assert pairs[0]["right_index"] == 3


def test_make_binaural_window_concatenates_left_then_right() -> None:
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    right = torch.tensor([[0.5], [1.5]])

    fused = make_binaural_window(left, right)

    expected_right = torch.tensor([[0.5, 0.0], [1.5, 0.0]])
    expected = torch.cat([left, expected_right], dim=0)
    assert fused.shape == (4, 2)
    assert torch.allclose(fused, expected)
