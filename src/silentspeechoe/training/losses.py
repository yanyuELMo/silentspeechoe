"""Loss functions for sentence classification.

The primary loss is standard cross‑entropy. Multi‑task or domain‑adaptation
losses can be added later.
"""

from __future__ import annotations

import torch.nn as nn


def build_loss() -> nn.Module:
    """Return the default cross‑entropy loss for 36‑way classification."""
    return nn.CrossEntropyLoss()
