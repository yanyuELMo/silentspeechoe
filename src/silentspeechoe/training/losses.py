"""Loss functions for closed-set classification experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# ArcFace
# ---------------------------------------------------------------------------


class ArcFaceLoss(nn.Module):
    """Standard embedding-based ArcFace angular-margin cross-entropy.

    The loss owns normalized class-center weights and expects pre-classifier
    embeddings with shape ``[B, D]``. During training it computes cosine logits
    between each embedding and each class center, applies an additive angular
    margin to the target class, and then optimizes cross-entropy.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        *,
        margin: float = 0.5,
        scale: float = 30.0,
        easy_margin: bool = False,
        label_smoothing: float = 0.0,
        eps: float = 1e-7,
    ):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {embedding_dim}")
        if num_classes <= 1:
            raise ValueError(f"num_classes must be greater than 1, got {num_classes}")
        if margin < 0:
            raise ValueError(f"margin must be non-negative, got {margin}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}"
            )
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.embedding_dim = int(embedding_dim)
        self.num_classes = int(num_classes)
        self.margin = float(margin)
        self.scale = float(scale)
        self.easy_margin = bool(easy_margin)
        self.label_smoothing = float(label_smoothing)
        self.eps = float(eps)
        self.requires_features = True

        self.weight = nn.Parameter(torch.empty(self.num_classes, self.embedding_dim))
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.margin_adjustment = math.sin(math.pi - self.margin) * self.margin
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize learnable class-center weights."""
        nn.init.xavier_uniform_(self.weight)

    def cosine_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Return scaled cosine logits without target-class margin."""
        if features.ndim != 2:
            raise ValueError(
                f"Expected features with shape [B, D], got {features.shape}"
            )
        if features.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected {self.embedding_dim} feature dimensions, got "
                f"{features.shape[1]}"
            )

        normalized_features = F.normalize(features, p=2, dim=1, eps=self.eps)
        normalized_weight = F.normalize(self.weight, p=2, dim=1, eps=self.eps)
        cosine = F.linear(normalized_features, normalized_weight)
        return cosine.clamp(-1.0 + self.eps, 1.0 - self.eps) * self.scale

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        """Return inference logits for prediction and metrics."""
        return self.cosine_logits(features)

    def compute_logits(
        self,
        features: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Return ArcFace logits with margin applied to target classes."""
        if target.ndim != 1:
            raise ValueError(f"Expected target with shape [B], got {target.shape}")

        cosine = self.cosine_logits(features) / self.scale
        if cosine.shape[0] != target.shape[0]:
            raise ValueError(
                "features and target batch sizes must match, got "
                f"{cosine.shape[0]} and {target.shape[0]}"
            )

        target = target.to(device=features.device, dtype=torch.long)
        if target.numel() > 0:
            min_target = int(target.min())
            max_target = int(target.max())
            if min_target < 0 or max_target >= self.num_classes:
                raise ValueError(
                    "target values must be in [0, num_classes), got "
                    f"min={min_target}, max={max_target}, classes={self.num_classes}"
                )

        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp_min(self.eps))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0.0, phi, cosine)
        else:
            phi = torch.where(
                cosine > self.threshold,
                phi,
                cosine - self.margin_adjustment,
            )

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        return (one_hot * phi + (1.0 - one_hot) * cosine) * self.scale

    def forward(self, features: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute ArcFace loss from embeddings."""
        arc_logits = self.compute_logits(features, target)
        return F.cross_entropy(
            arc_logits,
            target.to(device=features.device, dtype=torch.long),
            label_smoothing=self.label_smoothing,
        )


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss (SupCon)
# ---------------------------------------------------------------------------


class SupConLoss(nn.Module):
    """Supervised contrastive loss over L2-normalised embeddings.

    Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    (https://arxiv.org/abs/2004.11362)

    For each anchor sample the loss pulls together all embeddings that share
    the same label (positives) and pushes apart embeddings with different
    labels (negatives), using the standard InfoNCE temperature-scaled
    formulation.

    Inputs
    ------
    * **features** — ``[B, D]`` float tensor.  L2-normalisation is applied
      internally so raw pre-classifier embeddings may be passed directly.
    * **labels** — ``[B]`` long tensor of class indices ``0 … num_classes-1``.

    The batch must contain **at least two samples per class** for every class
    present, otherwise the corresponding anchors produce NaN gradients.
    """

    def __init__(
        self,
        temperature: float = 0.07,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = float(temperature)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the supervised contrastive loss.

        Args:
            features: ``[B, D]`` float tensor of (optionally un-normalised)
                embeddings.
            labels: ``[B]`` long tensor of class indices.

        Returns:
            Scalar loss.  Returns ``0.0`` when the batch contains fewer
            than 2 samples.
        """
        if features.ndim != 2:
            raise ValueError(
                f"Expected features with shape [B, D], got {features.shape}"
            )
        if labels.ndim != 1:
            raise ValueError(f"Expected labels with shape [B], got {labels.shape}")
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                "features and labels batch sizes must match, got "
                f"{features.shape[0]} and {labels.shape[0]}"
            )

        device = features.device
        batch_size = features.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        labels = labels.to(device=device, dtype=torch.long)

        # L2-normalise each embedding.
        features = F.normalize(features, p=2, dim=1)

        # Cosine similarity matrix [B, B].
        similarity = torch.matmul(features, features.T)  # [B, B]
        similarity = similarity / self.temperature

        # Build positive mask: 1 when labels[i] == labels[j] and i != j.
        label_eq = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        diag = torch.eye(batch_size, dtype=torch.bool, device=device)
        pos_mask = label_eq & ~diag  # [B, B]

        # Log-sum-exp over all samples except i (denominator).
        # Subtract max for numerical stability.
        sim_max = similarity.max(dim=1, keepdim=True).values.detach()
        sim_stable = similarity - sim_max
        exp_sim = sim_stable.exp()
        exp_sim = exp_sim.masked_fill(diag, 0.0)  # exclude self

        denom = exp_sim.sum(dim=1, keepdim=True)  # [B, 1]

        # For each anchor, compute mean log-prob over its positives.
        log_prob = sim_stable - denom.log()  # [B, B]
        pos_log_prob = log_prob * pos_mask.float()  # only keep positives
        pos_count = pos_mask.sum(dim=1).float().clamp(min=1)  # [B]

        # Per-anchor loss; skip anchors with no positives.
        loss_per_anchor = -(pos_log_prob.sum(dim=1) / pos_count)  # [B]
        loss_per_anchor = loss_per_anchor[pos_count > 1e-8]

        if loss_per_anchor.numel() == 0:
            return torch.tensor(0.0, device=device)

        return loss_per_anchor.mean()


# ---------------------------------------------------------------------------
# Combined CE + SupCon loss (for models that expose embeddings during training)
# ---------------------------------------------------------------------------


class SupConCrossEntropyLoss(nn.Module):
    """Combined supervised-contrastive + cross-entropy loss.

    Designed for models whose ``forward()`` returns a ``(features, logits)``
    tuple when the model attribute ``requires_labels_for_training`` is
    ``True`` and ``labels`` are passed.  When only logits are returned the
    module degrades gracefully to pure cross-entropy.

    Parameters
    ----------
    temperature:
        Temperature for the SupCon term.
    supcon_weight:
        Weight applied to the contrastive term.  The CE term always has
        weight 1.0.
    label_smoothing:
        Passed to :class:`~torch.nn.CrossEntropyLoss`.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        supcon_weight: float = 1.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.supcon = SupConLoss(temperature=temperature)
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.supcon_weight = float(supcon_weight)

    def forward(
        self,
        model_output: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            model_output: Either a ``[B, C]`` logits tensor or a
                ``(features [B, D], logits [B, C])`` tuple.
            target: ``[B]`` long tensor of class indices.

        Returns:
            Scalar loss (SupCon + CE when features are available, pure CE
            otherwise).
        """
        if isinstance(model_output, tuple):
            features, logits = model_output
            loss_ce = self.ce(logits, target)
            loss_supcon = self.supcon(features, target)
            return loss_ce + self.supcon_weight * loss_supcon

        # Fallback: plain logits → pure cross-entropy.
        return self.ce(model_output, target)


def _contains(config: Any, key: str) -> bool:
    """Return whether a mapping-like config contains a key."""
    try:
        return key in config
    except TypeError:
        return False


def _get(config: Any, key: str, default: Any = None) -> Any:
    """Read a key from dict/OmegaConf-like objects."""
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _resolve_loss_config(config: Any | None) -> Any | None:
    """Resolve a loss subconfig from root, train, or direct loss config."""
    if config is None:
        return None
    if _contains(config, "train") and _contains(_get(config, "train"), "loss"):
        return _get(_get(config, "train"), "loss")
    if _contains(config, "loss"):
        return _get(config, "loss")

    name = _get(config, "name")
    if isinstance(name, str) and name.lower() in {
        "cross_entropy",
        "ce",
        "arcface",
        "arc_face",
    }:
        return config
    return None


def _resolve_num_classes(config: Any | None, loss_cfg: Any | None) -> int:
    """Resolve the number of classes for embedding-based classification losses."""
    explicit = _get(loss_cfg, "num_classes", None)
    if explicit is not None:
        return int(explicit)
    if config is not None and _contains(config, "model"):
        model_cfg = _get(config, "model")
        if _contains(model_cfg, "num_classes"):
            return int(_get(model_cfg, "num_classes"))
    raise ValueError(
        "ArcFace loss requires num_classes. Set loss.num_classes or "
        "model.num_classes in the Hydra config."
    )


def _resolve_embedding_dim(config: Any | None, loss_cfg: Any | None) -> int:
    """Resolve the pre-classifier embedding dimension from loss/model config."""
    explicit = _get(loss_cfg, "embedding_dim", None)
    if explicit is not None:
        return int(explicit)
    if config is None or not _contains(config, "model"):
        raise ValueError(
            "ArcFace loss requires embedding_dim. Set loss.embedding_dim or "
            "provide a model config with an inferable embedding size."
        )

    model_cfg = _get(config, "model")
    if _contains(model_cfg, "embedding_dim"):
        return int(_get(model_cfg, "embedding_dim"))

    hidden_features = _get(model_cfg, "hidden_features", None)
    if hidden_features:
        return int(hidden_features[-1])

    if _contains(model_cfg, "hidden_channels"):
        return int(_get(model_cfg, "hidden_channels")) * 2

    raise ValueError(
        "Could not infer ArcFace embedding_dim from model config. Set "
        "train.loss.embedding_dim explicitly."
    )


def build_loss(config: Any | None = None) -> nn.Module:
    """Build a classification loss from Hydra-style configuration."""
    loss_cfg = _resolve_loss_config(config)
    name = str(_get(loss_cfg, "name", "cross_entropy")).lower()

    if name in {"cross_entropy", "ce"}:
        return nn.CrossEntropyLoss(
            label_smoothing=float(_get(loss_cfg, "label_smoothing", 0.0))
        )

    if name in {"arcface", "arc_face"}:
        return ArcFaceLoss(
            embedding_dim=_resolve_embedding_dim(config, loss_cfg),
            num_classes=_resolve_num_classes(config, loss_cfg),
            margin=float(_get(loss_cfg, "margin", 0.5)),
            scale=float(_get(loss_cfg, "scale", 30.0)),
            easy_margin=bool(_get(loss_cfg, "easy_margin", False)),
            label_smoothing=float(_get(loss_cfg, "label_smoothing", 0.0)),
            eps=float(_get(loss_cfg, "eps", 1e-7)),
        )

    if name == "supcon":
        return SupConLoss(
            temperature=float(_get(loss_cfg, "temperature", 0.07)),
        )

    if name in {"supcon_ce", "supcon_cross_entropy"}:
        return SupConCrossEntropyLoss(
            temperature=float(_get(loss_cfg, "temperature", 0.07)),
            supcon_weight=float(_get(loss_cfg, "supcon_weight", 1.0)),
            label_smoothing=float(_get(loss_cfg, "label_smoothing", 0.0)),
        )

    raise ValueError(
        f"Unknown loss name: {name!r}. "
        "Expected 'cross_entropy', 'arcface', 'supcon', or 'supcon_ce'."
    )


__all__ = [
    "ArcFaceLoss",
    "SupConCrossEntropyLoss",
    "SupConLoss",
    "build_loss",
]
