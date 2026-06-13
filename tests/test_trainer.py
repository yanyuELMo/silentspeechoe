"""Tests for training loop best-snapshot tracking (data-free)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset

from silentspeechoe.training.trainer import run_training


class _DummyDataset(Dataset):
    """Deterministic dataset for snapshot-tracking tests."""

    def __init__(self, n: int = 32, c: int = 9, t: int = 100, num_classes: int = 36):
        self.n = n
        self.c = c
        self.t = t
        self.num_classes = num_classes

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "x": torch.randn(self.c, self.t),
            "y": idx % self.num_classes,
            "length": self.t,
            "domain": "normal",
            "subject_id": f"sub_{idx:02d}",
            "session_id": "000_test",
            "sentence_id": f"nonsem_{(idx % 36) + 1:03d}",
            "repeat_id": 1,
            "side": "left",
        }


def _collate(batch):
    xs = [item["x"] for item in batch]
    max_len = max(x.shape[1] for x in xs)
    C = xs[0].shape[0]
    padded = torch.zeros(len(xs), C, max_len)
    for i, x in enumerate(xs):
        padded[i, :, : x.shape[1]] = x
    return {
        "x": padded,
        "y": torch.tensor([item["y"] for item in batch], dtype=torch.long),
        "lengths": torch.tensor([item["length"] for item in batch], dtype=torch.long),
        "domain": [item["domain"] for item in batch],
    }


class _SimpleModel(torch.nn.Module):
    """Tiny model that learns quickly for snapshot tests."""

    def __init__(self, in_channels=9, num_classes=36):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(in_channels, num_classes)

    def forward(self, x, lengths=None):
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class TestTrainerSnapshots:
    def test_best_loss_tracked(self):
        model = _SimpleModel()
        ds = _DummyDataset(32)
        loader = DataLoader(ds, batch_size=8, collate_fn=_collate)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        history = run_training(
            model,
            loader,
            loader,
            optimizer,
            criterion,
            device=torch.device("cpu"),
            max_epochs=3,
            log_interval=1,
        )

        assert "best_val_loss" in history
        assert history["best_val_loss"] is not None
        assert "epoch" in history["best_val_loss"]
        assert "value" in history["best_val_loss"]
        assert "model_state_dict" in history["best_val_loss"]
        assert "optimizer_state_dict" in history["best_val_loss"]
        assert 1 <= history["best_val_loss"]["epoch"] <= 3

    def test_best_accuracy_tracked(self):
        model = _SimpleModel()
        ds = _DummyDataset(32)
        loader = DataLoader(ds, batch_size=8, collate_fn=_collate)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        history = run_training(
            model,
            loader,
            loader,
            optimizer,
            criterion,
            device=torch.device("cpu"),
            max_epochs=3,
            log_interval=1,
        )

        assert "best_accuracy" in history
        assert history["best_accuracy"] is not None
        assert "epoch" in history["best_accuracy"]
        assert "value" in history["best_accuracy"]
        assert "model_state_dict" in history["best_accuracy"]
        assert "optimizer_state_dict" in history["best_accuracy"]
        assert 1 <= history["best_accuracy"]["epoch"] <= 3
        assert 0.0 <= history["best_accuracy"]["value"] <= 1.0

    def test_best_accuracy_and_loss_may_differ(self):
        """Best loss and best accuracy epochs can differ."""
        model = _SimpleModel()
        ds = _DummyDataset(32)
        loader = DataLoader(ds, batch_size=8, collate_fn=_collate)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        history = run_training(
            model,
            loader,
            loader,
            optimizer,
            criterion,
            device=torch.device("cpu"),
            max_epochs=5,
            log_interval=1,
        )

        # At least both exist with valid model state dicts.
        loss_state = history["best_val_loss"]["model_state_dict"]
        acc_state = history["best_accuracy"]["model_state_dict"]
        assert set(loss_state.keys()) == set(acc_state.keys())
        # Each is a dict of tensors.
        for k in loss_state:
            assert isinstance(loss_state[k], torch.Tensor)
            assert isinstance(acc_state[k], torch.Tensor)

    def test_val_metrics_preserved(self):
        model = _SimpleModel()
        ds = _DummyDataset(32)
        loader = DataLoader(ds, batch_size=8, collate_fn=_collate)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        history = run_training(
            model,
            loader,
            loader,
            optimizer,
            criterion,
            device=torch.device("cpu"),
            max_epochs=2,
            log_interval=1,
        )

        assert len(history["train_loss"]) == 2
        assert len(history["val_metrics"]) == 2
        for vm in history["val_metrics"]:
            assert "val_loss" in vm
            assert "overall" in vm
            assert "accuracy" in vm["overall"]
