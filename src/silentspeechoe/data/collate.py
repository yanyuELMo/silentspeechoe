"""Batch collation with time‑axis padding.

Because utterance windows vary in duration and the sensor streams are not
resampled yet, each sample may have a different number of time steps.
The collate function pads every sample in a batch to the longest sequence.
"""

from __future__ import annotations

import torch


def pad_collate(batch: list[dict]) -> dict:
    """Collate a list of dataset items into a padded batch.

    Each dataset item is expected to be a dict with at least::

        {
            "x":           FloatTensor [C, T_i],
            "y":           int,
            "speech_mode": str,
            "subject_id":  str,
        }

    Returns a dict with the same keys, where ``x`` is padded to
    ``[B, C, max_T]``, ``y`` is ``LongTensor [B]``, and the string
    fields are plain lists of length ``B``.

    An additional key ``lengths`` (``LongTensor [B]``) stores the
    original (un‑padded) time dimension of each sample.
    """
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    modes: list[str] = []
    subjects: list[str] = []
    lengths: list[int] = []

    for item in batch:
        x = item["x"]
        if x.dim() != 2:
            raise ValueError(f"Expected x of shape [C, T], got {x.shape}")
        xs.append(x)
        ys.append(int(item["y"]))
        modes.append(item["speech_mode"])
        subjects.append(item["subject_id"])
        lengths.append(x.shape[1])

    max_len = max(lengths) if lengths else 0
    C = xs[0].shape[0] if xs else 0

    padded = torch.zeros(len(xs), C, max_len, dtype=torch.float32)
    for i, x in enumerate(xs):
        T = x.shape[1]
        padded[i, :, :T] = x

    return {
        "x": padded,
        "y": torch.tensor(ys, dtype=torch.long),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "speech_mode": modes,
        "subject_id": subjects,
    }
