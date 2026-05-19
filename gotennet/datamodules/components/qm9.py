import torch
from typing import Optional

from torch_geometric.datasets import QM9 as QM9_geometric
from torch_geometric.transforms import Compose

# ---------------------------------------------------------------------------
# QM9 target properties and their column indices in the raw dataset.
# The dict is the single source of truth – all lookups derive from it.
# ---------------------------------------------------------------------------
qm9_target_dict: dict[int, str] = {
    0: "mu",
    1: "alpha",
    2: "homo",
    3: "lumo",
    4: "gap",
    5: "r2",
    6: "zpve",
    7: "U0",
    8: "U",
    9: "H",
    10: "G",
    11: "Cv",
}

# Reverse mapping: property name -> column index (built once, shared).
_qm9_label_to_idx: dict[str, int] = {
    label: idx for idx, label in qm9_target_dict.items()
}


class QM9(QM9_geometric):
    """
    QM9 dataset wrapper for PyTorch Geometric.

    Extends the PyTorch Geometric ``QM9`` dataset to support:

    * **Property filtering** – each instance stores only the target property
      specified via ``dataset_arg`` (shape ``(1,)``).
    * **Summary statistics** – :meth:`mean`, :meth:`std`, and :meth:`min`
      iterate over the full dataset on demand.
    * **Atomic references** – :meth:`get_atomref` returns per-element reference
      values for the active property.
    """

    available_properties: list[str] = list(qm9_target_dict.values())

    def __init__(
        self,
        root: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        dataset_arg: Optional[str] = None,
    ):
        """
        Args:
            root: Root directory where the dataset is stored / will be saved.
            transform: Per-sample transform. If ``None``, defaults to
                        :meth:`_filter_label` (selecting only the target
                        property column).
            pre_transform: Transform applied before saving to disk.
            pre_filter: Filter function that receives a data object and returns
                        ``True`` if it should be kept.
            dataset_arg: Target property name (e.g. ``"U0"``, ``"gap"``).
                         **Required** – must be one of
                         :attr:`available_properties`.
        """
        assert dataset_arg is not None, (
            f"Pass the desired property via 'dataset_arg'. "
            f"Available properties: {', '.join(qm9_target_dict.values())}."
        )

        self.label = dataset_arg
        self.label_idx = _qm9_label_to_idx[self.label]

        # Build the transform pipeline: label-filtering is always the
        # last transform applied so that ``batch.y`` always has shape ``(N, 1)``.
        filter_tfm = self._filter_label
        if transform is None:
            transform = filter_tfm
        else:
            transform = Compose([transform, filter_tfm])

        super().__init__(
            root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @staticmethod
    def label_to_idx(label: str) -> int:
        """Return the column index in the raw QM9 data for *label*."""
        return _qm9_label_to_idx[label]

    def get_atomref(self, max_z: int = 100) -> Optional[torch.Tensor]:
        """
        Atomic reference values for the target property.

        Args:
            max_z: Maximum atomic number to pad / truncate to.

        Returns:
            Tensor of shape ``(max_z, 1)``, or ``None`` if no reference
            data exists for this property.
        """
        atomref = self.atomref(self.label_idx)
        if atomref is None:
            return None

        if atomref.size(0) == max_z:
            return atomref

        padded = torch.zeros(max_z, 1)
        n = min(max_z, atomref.size(0))
        padded[:n] = atomref[:n]
        return padded

    # ------------------------------------------------------------------
    # Summary statistics (iterate over *all* samples – use sparingly)
    # ------------------------------------------------------------------

    def mean(self, divide_by_atoms: bool = True) -> float:
        """Mean of the target property over the whole dataset."""
        return self._agg("mean", divide_by_atoms)

    def std(self, divide_by_atoms: bool = True) -> float:
        """Standard deviation of the target property over the whole dataset."""
        return self._agg("std", divide_by_atoms)

    def min(self, divide_by_atoms: bool = True) -> float:
        """Minimum of the target property over the whole dataset."""
        return self._agg("min", divide_by_atoms)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _agg(self, reduction: str, divide_by_atoms: bool) -> float:
        """
        Compute *reduction* over the target property of every sample.

        Parameters
        ----------
        reduction:
            One of ``"mean"``, ``"std"``, ``"min"`` – the name of the
            :class:`torch.Tensor` reduction method to apply.
        divide_by_atoms:
            If ``True``, each label is normalised by the number of atoms
            in its molecule **before** the reduction.
        """
        values = []
        for i in range(len(self)):
            sample = self.get(i)
            # After ``_filter_label``, ``sample.y`` has shape ``(1, 1)``.
            y = sample.y
            if divide_by_atoms:
                y = y / sample.pos.shape[0]
            values.append(y)

        all_y = torch.cat(values, dim=0)  # (N, 1)
        return getattr(all_y, reduction)(dim=0).item()

    def _filter_label(self, batch):
        """
        Transform that keeps **only** the target property column.

        After this transform ``batch.y`` has shape ``(N, 1)`` where ``N`` is
        the batch size.
        """
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch
