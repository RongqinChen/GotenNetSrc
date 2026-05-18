"""Prediction heads used by task-specific output assembly."""

from .atomwise import Atomwise, AtomwiseV3
from .blocks import GatedEquivariantBlock
from .molecular import Dipole, ElectronicSpatialExtentV2

__all__ = [
    "Atomwise",
    "AtomwiseV3",
    "Dipole",
    "ElectronicSpatialExtentV2",
    "GatedEquivariantBlock",
]
