"""QM9 dataset wrapper — 133k small organic molecules with 12 DFT properties."""

from __future__ import annotations

from functools import lru_cache

import torch
from torch_geometric.datasets import QM9 as QM9_pyg
from torch_geometric.transforms import Compose

# ---------------------------------------------------------------------------
# QM9 target properties: column index → property name.
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

# Reverse mapping: property name → column index.
_qm9_label_to_idx: dict[str, int] = {
    label: idx for idx, label in qm9_target_dict.items()
}


class QM9(QM9_pyg):
    """QM9 dataset — one target property per sample.

    Extends the PyTorch Geometric ``QM9`` dataset with:

    * **Property filtering** — ``label`` selects a single target column;
      ``batch.y`` will have shape ``(N, 1)``.
    * **Summary statistics** — :meth:`mean`, :meth:`std`, :meth:`min` iterate
      the full dataset once and cache results.
    * **Atomic references** — :meth:`get_atomref` returns per-element reference
      energies for the active property.
    """

    available_properties: list[str] = list(qm9_target_dict.values())

    def __init__(
        self, root, transform=None, pre_transform=None, pre_filter=None, label=None
    ):
        self.label = self._validate_label(label)
        self.label_idx = _qm9_label_to_idx[self.label]

        # The label filter transform is always applied last so that
        # ``batch.y`` consistently has shape ``(N, 1)``.
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

    # ── Public helpers ──────────────────────────────────────────────────

    @staticmethod
    def _validate_label(label: str | None) -> str:
        if label is None:
            raise ValueError(
                f"Pass a property via 'label'. Available: {', '.join(qm9_target_dict.values())}."
            )
        if label not in _qm9_label_to_idx:
            raise ValueError(f"Unknown QM9 property '{label}'.")
        return label

    @staticmethod
    def label_to_idx(label: str) -> int:
        """Return the column index in the raw QM9 data for *label*."""
        try:
            return _qm9_label_to_idx[label]
        except KeyError as exc:
            raise ValueError(f"Unknown QM9 property '{label}'.") from exc

    def get_atomref(self, max_z: int = 100) -> torch.Tensor | None:
        """Atomic reference energies, padded/truncated to *max_z* elements."""
        atomref = self.atomref(self.label_idx)
        if atomref is None:
            return None
        if atomref.size(0) == max_z:
            return atomref

        padded = torch.zeros(max_z, 1)
        n = min(max_z, atomref.size(0))
        padded[:n] = atomref[:n]
        return padded

    # ── Summary statistics ──────────────────────────────────────────────

    def mean(self, divide_by_atoms: bool = True) -> float:
        return self._get_values(divide_by_atoms).mean(dim=0).item()

    def std(self, divide_by_atoms: bool = True) -> float:
        return self._get_values(divide_by_atoms).std(dim=0).item()

    def min(self, divide_by_atoms: bool = True) -> float:
        return self._get_values(divide_by_atoms).min(dim=0).values.item()

    @lru_cache(maxsize=2)
    def _get_values(self, divide_by_atoms: bool) -> torch.Tensor:
        """Collect target values for all samples in one pass, caching the result."""
        values = []
        for i in range(len(self)):
            sample = self.get(i)
            y = sample.y
            if divide_by_atoms:
                y = y / sample.pos.shape[0]
            values.append(y)
        return torch.cat(values, dim=0)

    def _filter_label(self, batch):
        """Keep only the target property column; ``batch.y`` → shape ``(N, 1)``."""
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch
