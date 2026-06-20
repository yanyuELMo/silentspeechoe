"""Tests for fixed-vector IMU MLP models."""

from __future__ import annotations

import torch

from silentspeechoe.models.build import build_model
from silentspeechoe.models.imu_mlp import IMUMLPArcFace


class TestIMUMLPArcFace:
    @staticmethod
    def _make_cfg():
        """Return a minimal config that builds IMUMLPArcFace."""
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "model": {
                    "name": "imu_mlp_arcface",
                    "in_features": 1296,
                    "num_classes": 17,
                    "hidden1": 512,
                    "hidden2": 256,
                    "dropout": 0.0,
                    "arcface_scale": 30.0,
                    "arcface_margin": 0.3,
                }
            }
        )

    def test_forward_shape_without_labels(self):
        model = IMUMLPArcFace(dropout=0.0)
        model.eval()
        x = torch.randn(4, 1296)
        out = model(x)
        assert out.shape == (4, 17)
        assert torch.all(torch.isfinite(out))

    def test_forward_shape_with_arcface_labels(self):
        model = IMUMLPArcFace(dropout=0.0)
        model.eval()
        x = torch.randn(4, 1296)
        labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        plain = model(x)
        margin = model(x, labels=labels)

        assert margin.shape == (4, 17)
        assert torch.all(torch.isfinite(margin))
        assert not torch.allclose(plain, margin)

    def test_extract_features_shape(self):
        model = IMUMLPArcFace(dropout=0.0)
        model.eval()
        x = torch.randn(3, 1296)
        with torch.no_grad():
            features = model.extract_features(x)
        assert features.shape == (3, 256)

    def test_build_via_factory(self):
        cfg = self._make_cfg()
        model = build_model(cfg)
        assert isinstance(model, IMUMLPArcFace)
        x = torch.randn(2, 1296)
        out = model(x)
        assert out.shape == (2, 17)
