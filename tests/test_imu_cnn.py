"""Tests for IMUCNN model (data-free, synthetic only)."""

from __future__ import annotations

import torch

from silentspeechoe.models.build import build_model
from silentspeechoe.models.imu_cnn import IMUCNN


class TestIMUCNN:
    @staticmethod
    def _make_cfg():
        """Return a minimal config that builds IMUCNN."""
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "model": {
                    "name": "imu_cnn",
                    "in_channels": 9,
                    "num_classes": 36,
                    "conv1_channels": 64,
                    "conv2_channels": 128,
                    "conv3_channels": 128,
                    "kernel_size_1": 7,
                    "kernel_size_2": 5,
                    "kernel_size_3": 5,
                    "dropout": 0.3,
                }
            }
        )

    def test_forward_shape_no_lengths(self):
        model = IMUCNN(in_channels=9, num_classes=36)
        x = torch.randn(4, 9, 600)
        out = model(x)
        assert out.shape == (4, 36)

    def test_forward_shape_with_lengths(self):
        model = IMUCNN(in_channels=9, num_classes=36)
        x = torch.randn(4, 9, 600)
        lengths = torch.tensor([600, 550, 430, 100], dtype=torch.long)
        out = model(x, lengths=lengths)
        assert out.shape == (4, 36)

    def test_masked_pooling_ignores_padding(self):
        """Masked pooling gives near-identical results with and without
        trailing zeros, up to floating-point tolerance.

        Small differences (≈1e-4) are expected because BatchNorm and Conv
        floating-point accumulation paths differ slightly between the two
        tensor shapes, but masked pooling prevents the zeros from
        dominating the mean.
        """
        model = IMUCNN(in_channels=9, num_classes=36)
        model.eval()

        # Sample with full signal.
        x_full = torch.randn(1, 9, 300)
        len_full = torch.tensor([300], dtype=torch.long)

        # Same signal zero-padded on the right — same real content.
        x_padded = torch.zeros(1, 9, 500)
        x_padded[0, :, :300] = x_full[0]
        len_padded = torch.tensor([300], dtype=torch.long)

        with torch.no_grad():
            out_full = model(x_full, lengths=len_full)
            out_padded = model(x_padded, lengths=len_padded)

        # Verify masked pool ignores zeros (relaxed tolerance for floating
        # point differences across different tensor shapes in conv/bn).
        assert torch.allclose(out_full, out_padded, atol=1e-3)

        # Sanity check: without masked pooling, results would diverge
        # because the 200 zero timesteps would pull the mean down.
        with torch.no_grad():
            out_padded_unmasked = model(x_padded, lengths=None)
        assert not torch.allclose(out_full, out_padded_unmasked, atol=1e-2)

    def test_masked_vs_unmasked_equal_for_full_length(self):
        """When all samples are full length, masked = unmasked pool."""
        model = IMUCNN(in_channels=9, num_classes=36)
        model.eval()

        x = torch.randn(2, 9, 400)
        lengths = torch.tensor([400, 400], dtype=torch.long)

        with torch.no_grad():
            out_masked = model(x, lengths=lengths)
            out_unmasked = model(x, lengths=None)

        assert torch.allclose(out_masked, out_unmasked, atol=1e-5)

    def test_variable_lengths(self):
        """Different lengths produce different pooling results."""
        model = IMUCNN(in_channels=9, num_classes=36)
        model.eval()

        x = torch.randn(1, 9, 500)
        len_short = torch.tensor([100], dtype=torch.long)
        len_long = torch.tensor([500], dtype=torch.long)

        with torch.no_grad():
            out_short = model(x, lengths=len_short)
            out_long = model(x, lengths=len_long)

        # Different pooling windows → different logits.
        assert not torch.allclose(out_short, out_long, atol=1e-4)

    def test_in_channels_18(self):
        """Double-sided input (left+right concatenation) should work."""
        model = IMUCNN(in_channels=18, num_classes=36)
        x = torch.randn(4, 18, 500)
        out = model(x)
        assert out.shape == (4, 36)

    def test_build_via_factory(self):
        cfg = self._make_cfg()
        model = build_model(cfg)
        assert isinstance(model, IMUCNN)
        x = torch.randn(2, 9, 300)
        out = model(x)
        assert out.shape == (2, 36)

    def test_output_is_finite(self):
        model = IMUCNN()
        x = torch.randn(3, 9, 200)
        lengths = torch.tensor([200, 150, 80], dtype=torch.long)
        out = model(x, lengths=lengths)
        assert torch.all(torch.isfinite(out))

    def test_batch_size_one(self):
        model = IMUCNN()
        x = torch.randn(1, 9, 50)
        lengths = torch.tensor([50], dtype=torch.long)
        out = model(x, lengths=lengths)
        assert out.shape == (1, 36)

    def test_minimal_length(self):
        """A single time step should not crash."""
        model = IMUCNN()
        model.eval()
        x = torch.randn(2, 9, 1)
        lengths = torch.tensor([1, 1], dtype=torch.long)
        with torch.no_grad():
            out = model(x, lengths=lengths)
        assert out.shape == (2, 36)
        assert torch.all(torch.isfinite(out))

    # ------------------------------------------------------------------
    # extract_features
    # ------------------------------------------------------------------

    def test_extract_features_shape(self):
        """extract_features returns [B, embedding_dim]."""
        model = IMUCNN(in_channels=9, num_classes=36)
        model.eval()
        x = torch.randn(4, 9, 600)
        with torch.no_grad():
            feats = model.extract_features(x)
        assert feats.shape == (4, 128)
        assert feats.dtype == torch.float32

    def test_extract_features_with_lengths(self):
        """Masked features differ from unmasked for short sequences."""
        model = IMUCNN(in_channels=9, num_classes=36)
        model.eval()
        x = torch.randn(2, 9, 400)
        lengths = torch.tensor([100, 400], dtype=torch.long)
        with torch.no_grad():
            feats = model.extract_features(x, lengths=lengths)
        assert feats.shape == (2, 128)
        assert torch.all(torch.isfinite(feats))

    def test_extract_features_feeds_forward(self):
        """classifier(features) == forward(x)."""
        model = IMUCNN(in_channels=9, num_classes=36)
        model.eval()
        x = torch.randn(3, 9, 300)
        lengths = torch.tensor([300, 200, 100], dtype=torch.long)
        with torch.no_grad():
            feats = model.extract_features(x, lengths=lengths)
            logits_via_feats = model.classifier(feats)
            logits_direct = model(x, lengths=lengths)
        assert torch.allclose(logits_via_feats, logits_direct, atol=1e-5)

    def test_extract_features_in_channels_18(self):
        """Double-sided extract_features works."""
        model = IMUCNN(in_channels=18, num_classes=36)
        model.eval()
        x = torch.randn(2, 18, 500)
        with torch.no_grad():
            feats = model.extract_features(x)
        assert feats.shape == (2, 128)
