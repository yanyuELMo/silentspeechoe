"""Tests for CNN models over fixed-length feature vectors."""

from __future__ import annotations

import torch

from silentspeechoe.models.cnn import IMUBinauralFeatureCNN


def test_binaural_feature_cnn_extracts_from_flat_features() -> None:
    model = IMUBinauralFeatureCNN()
    x = torch.randn(4, 1296)

    features = model.extract_features(x)

    assert features.shape == (4, 512)
    assert model.embedding_dim == 512
    assert torch.isfinite(features).all()
