"""MLP models for fixed-length IMU and MFCC feature vectors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MFCCMLP(nn.Module):
    """Simple MLP for pre-computed MFCC feature vectors.

    Input: ``[B, D]`` fixed-length feature vector (D=234 by default).
    Output: ``[B, num_classes]`` logits.

    Architecture::

        Linear -> BN -> ReLU -> Dropout
        -> Linear -> BN -> ReLU -> Dropout
        -> Linear -> num_classes
    """

    def __init__(
        self,
        in_features: int = 234,
        num_classes: int = 36,
        hidden1: int = 256,
        hidden2: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden2, num_classes)

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """Return class logits for fixed-length MFCC features."""
        del lengths
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        return self.classifier(x)

    def extract_features(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """Return pre-classifier features ``[B, hidden2]``."""
        del lengths
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class IMUMLP(nn.Module):
    """MLP for fixed-length binaural IMU feature vectors.

    Intended for handcrafted IMU features such as the 1296-dimensional
    binaural temporal-envelope vector ``[left, right, left-right]``.
    """

    def __init__(
        self,
        in_features: int = 1296,
        num_classes: int = 17,
        hidden1: int = 512,
        hidden2: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.fc1 = nn.Linear(in_features, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden2, num_classes)

    def _check_input(self, x: torch.Tensor) -> None:
        """Validate fixed-vector input shape."""
        if x.dim() != 2:
            raise ValueError(f"Expected input shape [B, D], got {tuple(x.shape)}")
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected feature dimension {self.in_features}, got {x.shape[1]}"
            )

    def extract_features(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """Return pre-classifier embeddings for fixed IMU features."""
        del lengths
        self._check_input(x)

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """Return class logits for fixed IMU feature vectors."""
        features = self.extract_features(x, lengths=lengths)
        features = self.dropout(features)
        return self.classifier(features)


class ArcFaceHead(nn.Module):
    """ArcFace classification head for normalized identity embeddings."""

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.3,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.num_classes = int(num_classes)
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return scaled cosine logits, with ArcFace margin when labels exist."""
        cosine = F.linear(
            F.normalize(embeddings, dim=1),
            F.normalize(self.weight, dim=1),
        ).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None:
            return cosine * self.scale

        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.margin)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).to(
            device=cosine.device,
            dtype=cosine.dtype,
        )
        logits = cosine * (1.0 - one_hot) + target_logits * one_hot
        return logits * self.scale


class IMUMLPArcFace(IMUMLP):
    """IMU MLP with an ArcFace identity classification head."""

    requires_labels_for_training = True

    def __init__(
        self,
        in_features: int = 1296,
        num_classes: int = 17,
        hidden1: int = 512,
        hidden2: int = 256,
        dropout: float = 0.3,
        arcface_scale: float = 30.0,
        arcface_margin: float = 0.3,
    ):
        super().__init__(
            in_features=in_features,
            num_classes=num_classes,
            hidden1=hidden1,
            hidden2=hidden2,
            dropout=dropout,
        )
        self.classifier = ArcFaceHead(
            in_features=hidden2,
            num_classes=num_classes,
            scale=arcface_scale,
            margin=arcface_margin,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths=None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ArcFace logits during training and cosine logits otherwise."""
        features = self.extract_features(x, lengths=lengths)
        features = self.dropout(features)
        return self.classifier(features, labels=labels)
