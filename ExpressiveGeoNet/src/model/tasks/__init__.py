"""Task implementations for supported molecular prediction datasets."""

from src.model.tasks.md import MDTask
from src.model.tasks.molecule3d import Molecule3DTask
from src.model.tasks.qm9 import QM9Task

TASK_DICT = {
    "QM9": QM9Task,
    "rMD17": MDTask,
    "MD22": MDTask,
    "Molecule3D": Molecule3DTask,
}

__all__ = ["MDTask", "Molecule3DTask", "QM9Task", "TASK_DICT"]
