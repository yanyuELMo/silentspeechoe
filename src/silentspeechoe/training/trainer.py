"""Minimal training loop for silentspeechoe baseline experiments."""

from __future__ import annotations

import logging
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from silentspeechoe.evaluation.metrics import compute_grouped_classification_metrics

logger = logging.getLogger(__name__)


def _copy_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy model weights to CPU for stable checkpoint selection."""
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def _forward_train(
    model: nn.Module,
    x: torch.Tensor,
    lengths: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Forward helper for models that need labels during training."""
    if bool(getattr(model, "requires_labels_for_training", False)):
        return model(x, lengths=lengths, labels=y)
    return model(x, lengths=lengths)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch, return average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        lengths = batch["lengths"].to(device)

        optimizer.zero_grad()
        logits = _forward_train(model, x, lengths, y)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Run validation, returning loss + grouped classification metrics."""
    model.eval()

    total_loss = 0.0
    num_batches = 0

    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    all_y_score: list[torch.Tensor] = []
    all_groups: list[str] = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        lengths = batch["lengths"].to(device)
        groups = batch["domain"]  # list of str

        logits = model(x, lengths=lengths)
        loss = criterion(logits, y)

        total_loss += float(loss.item())
        num_batches += 1

        preds = logits.argmax(dim=-1)

        all_y_true.extend(y.cpu().tolist())
        all_y_pred.extend(preds.cpu().tolist())
        all_y_score.append(logits.cpu())
        all_groups.extend(groups)

    avg_loss = total_loss / max(num_batches, 1)

    # Concatenate all logits into [N, C]
    import torch as _torch

    y_true = _torch.tensor(all_y_true)
    y_pred = _torch.tensor(all_y_pred)
    y_score = _torch.cat(all_y_score, dim=0) if all_y_score else _torch.empty(0, 36)

    metrics = compute_grouped_classification_metrics(
        y_true, y_pred, y_score, all_groups, top_k=3
    )
    metrics["val_loss"] = avg_loss

    return metrics


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_epochs: int = 50,
    log_interval: int = 1,
) -> dict:
    """Full training loop.

    Returns a dict with per‑epoch ``train_loss`` and ``val_metrics``.
    The dict also includes the best model snapshots by validation loss
    and overall validation accuracy.
    """
    history: dict[str, list] = defaultdict(list)
    best_loss: dict | None = None
    best_accuracy: dict | None = None

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_metrics"].append(val_metrics)

        val_loss = float(val_metrics["val_loss"])
        val_accuracy = float(val_metrics["overall"]["accuracy"])
        if best_loss is None or val_loss < float(best_loss["value"]):
            best_loss = {
                "epoch": epoch,
                "value": val_loss,
                "model_state_dict": _copy_model_state(model),
                "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                "val_metrics": val_metrics,
            }
        if best_accuracy is None or val_accuracy > float(best_accuracy["value"]):
            best_accuracy = {
                "epoch": epoch,
                "value": val_accuracy,
                "model_state_dict": _copy_model_state(model),
                "optimizer_state_dict": deepcopy(optimizer.state_dict()),
                "val_metrics": val_metrics,
            }

        if epoch % log_interval == 0:
            logger.info(
                "Epoch %3d | train loss %.4f | val loss %.4f | "
                "val acc %.4f | val top3 %.4f",
                epoch,
                train_loss,
                val_loss,
                val_accuracy,
                val_metrics["overall"]["top3_accuracy"],
            )

    result = dict(history)
    result["best_val_loss"] = best_loss
    result["best_accuracy"] = best_accuracy
    return result
