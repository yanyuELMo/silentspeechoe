"""Tests for binaural IMU window pairing."""

from __future__ import annotations

import torch

from scripts.train import _build_binaural_window_pairs, _WindowSubset


class _TinyWindowDataset(torch.utils.data.Dataset):
    """Minimal in-memory window dataset for train wrapper tests."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        return {
            "x": torch.ones(3, 4),
            "y": index,
            "length": 4,
            "domain": "normal",
            "subject_id": "sub_01",
        }


def test_window_subset_keeps_original_and_augmented_views() -> None:
    """Training augmentation should expand each source window into two views."""

    def augmenter(x: torch.Tensor) -> torch.Tensor:
        return x * 2.0

    dataset = _WindowSubset(
        _TinyWindowDataset(),
        indices=[0],
        targets={0: 5},
        augmenter=augmenter,
    )

    assert len(dataset) == 2
    original = dataset[0]
    augmented = dataset[1]
    assert original["augmentation_view"] == "original"
    assert augmented["augmentation_view"] == "augmented"
    assert original["y"] == 5
    assert augmented["y"] == 5
    assert torch.allclose(original["x"], torch.ones(3, 4))
    assert torch.allclose(augmented["x"], torch.full((3, 4), 2.0))
    assert torch.allclose(augmented["x_original"], torch.ones(3, 4))


def test_binaural_pairs_require_matching_metadata() -> None:
    """Pairing should require all key utterance metadata to match."""

    records = [
        {
            "subject_id": "sub_01",
            "event_id": 7,
            "ear": "left",
            "sentence_type": "non_semantic",
            "sentence_id": "nonsem_001",
            "label_id": 1,
            "domain": "normal",
            "repeat_id": 1,
        },
        {
            "subject_id": "sub_01",
            "event_id": 7,
            "ear": "right",
            "sentence_type": "semantic",
            "sentence_id": "sem_999",
            "label_id": 9,
            "domain": "whisper",
            "repeat_id": 2,
        },
        {
            "subject_id": "sub_01",
            "event_id": 8,
            "ear": "left",
            "sentence_type": "non_semantic",
            "sentence_id": "nonsem_002",
            "label_id": 2,
            "domain": "silent",
            "repeat_id": 1,
        },
        {
            "subject_id": "sub_01",
            "event_id": 8,
            "ear": "right",
            "sentence_type": "non_semantic",
            "sentence_id": "nonsem_002",
            "label_id": 2,
            "domain": "silent",
            "repeat_id": 1,
        },
    ]

    pairs = _build_binaural_window_pairs(records)

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["subject_id"] == "sub_01"
    assert pair["event_id"] == 8
    assert pair["sentence_type"] == "non_semantic"
    assert pair["sentence_id"] == "nonsem_002"
    assert pair["domain"] == "silent"
    assert pair["repeat_id"] == 1
