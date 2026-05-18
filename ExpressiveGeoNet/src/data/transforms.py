"""Small dataset transforms shared by supported datasets."""

from __future__ import annotations


def normalize_positions(batch):
    """Normalize positions by subtracting the precomputed center of mass."""
    batch.pos = batch.pos - batch.center_of_mass
    return batch
