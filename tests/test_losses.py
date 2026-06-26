"""Tests for classification loss builders."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from silentspeechoe.training.losses import ArcFaceLoss, build_loss


def test_build_loss_defaults_to_cross_entropy() -> None:
    """Default loss should remain standard cross entropy."""
    loss = build_loss()
    assert isinstance(loss, torch.nn.CrossEntropyLoss)


def test_build_loss_reads_train_loss_config() -> None:
    """Hydra-style train.loss config should build embedding ArcFace loss."""
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "imu_feature_mlp_binaural",
                "hidden_features": [256, 128],
                "num_classes": 36,
            },
            "train": {
                "loss": {
                    "name": "arcface",
                    "margin": 0.25,
                    "scale": 16.0,
                }
            },
        }
    )

    loss = build_loss(cfg)

    assert isinstance(loss, ArcFaceLoss)
    assert loss.embedding_dim == 128
    assert loss.num_classes == 36
    assert loss.margin == 0.25
    assert loss.scale == 16.0


def test_arcface_loss_forward_backward() -> None:
    """ArcFace loss should produce a finite scalar and gradients."""
    features = torch.randn(8, 12, requires_grad=True)
    target = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    loss_fn = ArcFaceLoss(embedding_dim=12, num_classes=4, margin=0.5, scale=8.0)

    loss = loss_fn(features, target)
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert loss_fn.weight.grad is not None
    assert torch.isfinite(loss_fn.weight.grad).all()


def test_arcface_margin_changes_only_target_logits() -> None:
    """The angular margin should only alter each sample's target class."""
    features = torch.randn(2, 5)
    target = torch.tensor([0, 2])
    loss_fn = ArcFaceLoss(embedding_dim=5, num_classes=3, margin=0.3, scale=10.0)

    arc_logits = loss_fn.compute_logits(features, target)
    base_logits = loss_fn.cosine_logits(features)

    assert not torch.isclose(arc_logits[0, 0], base_logits[0, 0])
    assert not torch.isclose(arc_logits[1, 2], base_logits[1, 2])
    assert torch.allclose(arc_logits[0, 1:], base_logits[0, 1:])
    assert torch.allclose(arc_logits[1, :2], base_logits[1, :2])


def test_build_loss_infers_tcn_embedding_dim() -> None:
    """TCN ArcFace embedding size should be hidden_channels * 2."""
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "imu_temporal_cnn_binaural",
                "hidden_channels": 96,
                "num_classes": 36,
            },
            "train": {"loss": {"name": "arcface"}},
        }
    )

    loss = build_loss(cfg)

    assert isinstance(loss, ArcFaceLoss)
    assert loss.embedding_dim == 192
    assert loss.num_classes == 36


def test_build_loss_reads_cnn_embedding_dim() -> None:
    """CNN ArcFace embedding size can be provided explicitly by model config."""
    cfg = OmegaConf.create(
        {
            "model": {
                "name": "imu_binaural_feature_cnn",
                "embedding_dim": 512,
                "num_classes": 36,
            },
            "train": {"loss": {"name": "arcface"}},
        }
    )

    loss = build_loss(cfg)

    assert isinstance(loss, ArcFaceLoss)
    assert loss.embedding_dim == 512
    assert loss.num_classes == 36
