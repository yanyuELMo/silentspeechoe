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
            "x":             FloatTensor [C, T_i],
            "y":             int,
            "domain":        str,
            "subject_id":    str,
            "event_id":      int,
            "sentence_id":   str,
            "repeat_id":     int,
            "length":        int,
            "left_length":   int,
            "right_length":  int,
        }

    Returns a dict with:

    * ``x`` — ``FloatTensor [B, C, max_T]`` (zero‑padded)
    * ``y`` — ``LongTensor [B]``
    * ``lengths`` — ``LongTensor [B]``
    * ``left_lengths`` — ``LongTensor [B]``
    * ``right_lengths`` — ``LongTensor [B]``
    * ``domain``, ``subject_id``, ``sentence_id`` — list of str
    * ``event_id``, ``repeat_id`` — list of int
    """
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    domains: list[str] = []
    subjects: list[str] = []
    event_ids: list[int] = []
    sentence_ids: list[str] = []
    repeat_ids: list[int] = []
    lengths: list[int] = []
    left_lengths: list[int] = []
    right_lengths: list[int] = []

    for item in batch:
        x = item["x"]
        if x.dim() != 2:
            raise ValueError(f"Expected x of shape [C, T], got {x.shape}")
        xs.append(x)
        ys.append(int(item["y"]))
        domains.append(item.get("domain", item.get("speech_mode", "")))
        subjects.append(item["subject_id"])
        event_ids.append(int(item.get("event_id", -1)))
        sentence_ids.append(str(item.get("sentence_id", "")))
        repeat_ids.append(int(item.get("repeat_id", -1)))
        lengths.append(int(item.get("length", x.shape[1])))
        left_lengths.append(int(item.get("left_length", x.shape[1])))
        right_lengths.append(int(item.get("right_length", x.shape[1])))

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
        "left_lengths": torch.tensor(left_lengths, dtype=torch.long),
        "right_lengths": torch.tensor(right_lengths, dtype=torch.long),
        "domain": domains,
        "subject_id": subjects,
        "event_id": event_ids,
        "sentence_id": sentence_ids,
        "repeat_id": repeat_ids,
    }
