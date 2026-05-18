"""Prediction heads used by task-specific output assembly."""

from src.model.heads.atomwise import Atomwise, AtomwiseV3
from src.model.heads.blocks import GatedEquivariantBlock
from src.model.heads.molecular import Dipole, ElectronicSpatialExtentV2

__all__ = [
    "Atomwise",
    "AtomwiseV3",
    "Dipole",
    "ElectronicSpatialExtentV2",
    "GatedEquivariantBlock",
]
