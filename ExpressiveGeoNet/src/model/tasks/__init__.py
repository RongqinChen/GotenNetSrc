"""Task implementations for supported molecular prediction datasets."""

from .md import MDTask
from .molecule3d import Molecule3DTask
from .qm9 import QM9Task

TASK_DICT = {
    "QM9": QM9Task,
    "rMD17": MDTask,
    "MD22": MDTask,
    "Molecule3D": Molecule3DTask,
}

__all__ = ["MDTask", "Molecule3DTask", "QM9Task", "TASK_DICT"]
