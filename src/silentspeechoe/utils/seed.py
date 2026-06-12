"""Random seed helpers for reproducible experiments."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Also configures PyTorch deterministic algorithms where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Request deterministic algorithms (may impact performance).
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
