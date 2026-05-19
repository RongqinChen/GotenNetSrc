"""Small dataset transforms shared by supported datasets."""

from __future__ import annotations

import torch


def normalize_positions(batch):
    """Center positions using ``center_of_mass`` when available, otherwise the centroid."""
    center = getattr(batch, "center_of_mass", None)
    if center is None:
        center = batch.pos.mean(dim=0, keepdim=True)
    else:
        center = torch.as_tensor(center, device=batch.pos.device, dtype=batch.pos.dtype)
        if center.ndim == 1:
            center = center.unsqueeze(0)

    batch.pos = batch.pos - center
    return batch
