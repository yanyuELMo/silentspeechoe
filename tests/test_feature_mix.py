"""Tests for mixed fixed-feature training subsets."""

from __future__ import annotations

import torch

from scripts.train import _compute_mixed_feature_stats, _MixedFeatureVectorSubset


class _TinyFeatureDataset(torch.utils.data.Dataset):
    """Minimal fixed-vector dataset for mixed feature tests."""

    def __init__(self, offset: float):
        self.records = [
            {
                "subject_id": "sub_00",
                "domain": "normal",
                "sentence_type": "non_semantic",
                "augmentation_view": "original" if offset == 0.0 else "augmented",
            }
        ]
        self.offset = offset

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        return {
            "x": torch.tensor([self.offset + 1.0, self.offset + 3.0]),
            "y": index,
            "domain": "normal",
            "subject_id": "sub_00",
            "sentence_id": "nonsem_001",
            "repeat_id": 1,
            "side": "binaural",
        }


def test_mixed_feature_subset_reads_multiple_sources_and_targets() -> None:
    original = _TinyFeatureDataset(offset=0.0)
    augmented = _TinyFeatureDataset(offset=10.0)
    samples = [(original, 0, 7), (augmented, 0, 7)]
    mean, std = _compute_mixed_feature_stats(samples)

    subset = _MixedFeatureVectorSubset(samples, feature_mean=mean, feature_std=std)

    assert len(subset) == 2
    first = subset[0]
    second = subset[1]
    assert first["y"] == 7
    assert second["y"] == 7
    assert torch.allclose(first["x"], torch.tensor([-1.0, -1.0]))
    assert torch.allclose(second["x"], torch.tensor([1.0, 1.0]))
