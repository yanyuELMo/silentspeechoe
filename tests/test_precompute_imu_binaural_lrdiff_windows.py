"""Tests for 27-channel binaural IMU window precomputation helpers."""

from __future__ import annotations

import torch

from scripts.precompute_imu_binaural_lrdiff_windows import (
    build_binaural_lrdiff_pairs,
    make_lrdiff_window,
)


def test_lrdiff_pairs_require_same_event_metadata() -> None:
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
            "subject_id": "sub_01",
            "event_id": 2,
            "ear": "right",
            "sentence_type": "non_semantic",
            "sentence_id": "nonsem_001",
            "label_id": 0,
            "domain": "normal",
            "repeat_id": 1,
        },
        {
            "subject_id": "sub_01",
            "event_id": 3,
            "ear": "left",
            "sentence_type": "semantic",
            "sentence_id": "sem_001",
            "label_id": 1,
            "domain": "silent",
            "repeat_id": 1,
        },
        {
            "subject_id": "sub_01",
            "event_id": 3,
            "ear": "right",
            "sentence_type": "semantic",
            "sentence_id": "sem_001",
            "label_id": 1,
            "domain": "silent",
            "repeat_id": 1,
        },
    ]

    pairs = build_binaural_lrdiff_pairs(records)

    assert len(pairs) == 1
    assert pairs[0]["event_id"] == 3
    assert pairs[0]["left_index"] == 2
    assert pairs[0]["right_index"] == 3


def test_make_lrdiff_window_concatenates_left_right_and_difference() -> None:
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    right = torch.tensor([[0.5], [1.5]])

    fused = make_lrdiff_window(left, right)

    expected_right = torch.tensor([[0.5, 0.0], [1.5, 0.0]])
    expected = torch.cat([left, expected_right, left - expected_right], dim=0)
    assert fused.shape == (6, 2)
    assert torch.allclose(fused, expected)
