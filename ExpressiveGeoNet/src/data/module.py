"""Unified Lightning datamodule for all supported molecular datasets."""

from __future__ import annotations

import os.path
from typing import Any

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
from tqdm import tqdm

from src.common.logging import get_logger
from src.data.datasets import MD22, Molecule3D, QM9, rMD17
from src.data.splits import MissingLabelException, make_splits
from src.data.transforms import normalize_positions

log = get_logger(__name__)
SplitSize = float | int | None


class DataModule(LightningDataModule):
    """Lightning datamodule for all project molecular datasets."""

    _DATASET_CLASSES = {
        "QM9": QM9,
        "MD22": MD22,
        "Molecule3D": Molecule3D,
        "rMD17": rMD17,
    }

    # Mapping from config key to the private *_prepare* method name.
    _PREPARERS = {
        "QM9": "_prepare_qm9",
        "MD22": "_prepare_md22",
        "Molecule3D": "_prepare_molecule3d",
        "rMD17": "_prepare_rmd17",
    }

    def __init__(self, hparams: dict[str, Any] | Any):
        super().__init__()
        self.hparams.update(self._normalize_hparams(hparams))
        self._reset_dataset_state()

    # ── hparams normalization ───────────────────────────────────────────

    @staticmethod
    def _normalize_hparams(hparams: Any) -> dict[str, Any]:
        """Convert various hparams container types (dict, Namespace, etc.) to a plain dict."""
        if hasattr(hparams, "items"):
            return dict(hparams.items())
        if hasattr(hparams, "__dict__"):
            return dict(hparams.__dict__)
        return dict(hparams)

    def _reset_dataset_state(self) -> None:
        """Clear all cached dataset state and split references."""
        self._mean: float | None = None
        self._std: float | None = None
        self._saved_dataloaders: dict[str, DataLoader] = {}
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.loaded = False

    # ── Dataset properties ──────────────────────────────────────────────

    @property
    def dataset_class(self):
        """Return the dataset class matching the configured *dataset* key."""
        key = self.hparams["dataset"]
        if key not in self._DATASET_CLASSES:
            raise ValueError(f"Unsupported dataset type: {key}")
        return self._DATASET_CLASSES[key]

    @property
    def atomref(self):
        """Atomic reference values from the underlying dataset, if available."""
        if self.dataset is not None and hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self) -> float | None:
        return self._mean

    @property
    def std(self) -> float | None:
        return self._std

    # ── Dataset lifecycle ───────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Lazy-load the dataset on first access."""
        if not self.loaded:
            self.prepare_dataset()
            self.loaded = True

    def _update_label(self, label: str | None) -> None:
        """Switch to a different target label, resetting cached state if needed."""
        if label is None or self.hparams.get("label") == label:
            return
        self.hparams["label"] = label
        self._reset_dataset_state()

    def get_metadata(self, label: str | None = None) -> dict[str, Any]:
        """Return lightweight dataset metadata needed by task heads.

        The full dataset object is intentionally excluded so logger/checkpoint
        hyperparameter serialization cannot accidentally traverse cached PyG
        tensors such as ``edge_attr``.
        """
        self._update_label(label)
        self._ensure_loaded()
        atomref = self.atomref
        if isinstance(atomref, torch.Tensor):
            atomref = atomref.detach().clone()
        return {
            "atomref": atomref,
            "mean": self.mean,
            "std": self.std,
        }

    def prepare_dataset(self) -> None:
        """Load the full dataset, create train/val/test index splits, and optionally standardize targets."""
        dataset_type = self.hparams["dataset"]
        preparer_name = self._PREPARERS.get(dataset_type)
        if preparer_name is None:
            raise ValueError(f"Dataset {dataset_type} is not supported.")

        self._saved_dataloaders.clear()
        preparer = getattr(self, preparer_name)
        idx_train, idx_val, idx_test = preparer()

        log.info(
            "Splits: train=%s, val=%s, test=%s",
            len(idx_train),
            len(idx_val),
            len(idx_test),
        )
        self.train_dataset = self.dataset[idx_train]
        self.val_dataset = self.dataset[idx_val]
        self.test_dataset = self.dataset[idx_test]

        if self.hparams["standardize"]:
            self._standardize()

    # ── DataLoaders ─────────────────────────────────────────────────────

    def train_dataloader(self):
        self._ensure_loaded()
        return self._build_dataloader(self.train_dataset, stage="train")

    def val_dataloader(self):
        self._ensure_loaded()
        return self._build_dataloader(self.val_dataset, stage="val")

    def test_dataloader(self):
        self._ensure_loaded()
        return self._build_dataloader(self.test_dataset, stage="test")

    def _build_dataloader(
        self, dataset, stage: str, *, cache: bool = True
    ) -> DataLoader:
        """Build or return a cached DataLoader for *stage* (train/val/test).

        Caching is skipped when ``reload=True`` in the configuration.
        """
        cache = cache and not self.hparams.get("reload", False)
        if cache and stage in self._saved_dataloaders:
            return self._saved_dataloaders[stage]

        is_train = stage == "train"
        num_workers = int(self.hparams["num_workers"])
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": (
                self.hparams["batch_size"]
                if is_train
                else self.hparams["inference_batch_size"]
            ),
            "shuffle": is_train,
            "num_workers": num_workers,
            "pin_memory": self.hparams.get("pin_memory", True),
        }

        if num_workers > 0:
            dataloader_kwargs["persistent_workers"] = self.hparams.get(
                "persistent_workers", False
            )
            prefetch_factor = self.hparams.get("prefetch_factor")
            if prefetch_factor is not None:
                dataloader_kwargs["prefetch_factor"] = int(prefetch_factor)

        dataloader = DataLoader(**dataloader_kwargs)

        if cache:
            self._saved_dataloaders[stage] = dataloader
        return dataloader

    # ── Subset / size helpers ───────────────────────────────────────────

    @staticmethod
    def _resolve_size(size: SplitSize, total: int, name: str) -> int:
        """Convert *size* (float, int, or None) to an absolute count capped by *total*.

        * ``None`` → *total*
        * ``float`` in ``(0, 1]`` → fraction of *total*
        * ``int`` → used as-is (must be non-negative, ≤ *total*)
        """
        if size is None:
            return total
        if isinstance(size, float):
            if not 0 < size <= 1:
                raise ValueError(
                    f"{name}_size as a float must be in (0, 1], got {size}."
                )
            size = round(total * size)
        size = int(size)
        if size < 0:
            raise ValueError(f"{name}_size must be non-negative, got {size}.")
        if size > total:
            raise ValueError(
                f"{name}_size={size} exceeds available {name} samples ({total})."
            )
        return size

    def _subset_indices(
        self, indices: torch.Tensor, size: SplitSize, name: str
    ) -> torch.Tensor:
        """Take the first *size* elements from *indices*."""
        return indices[: self._resolve_size(size, len(indices), name)]

    # ── Target extraction & standardization ─────────────────────────────

    @staticmethod
    def _extract_targets(batch, atomref: torch.Tensor | None) -> torch.Tensor:
        """Collect batch targets, optionally subtracting atomic reference energies."""
        if batch.y is None:
            raise MissingLabelException()

        targets = batch.y
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)

        if atomref is not None:
            atomref_energy = scatter(
                atomref[batch.z],
                batch.batch,
                dim=0,
                dim_size=targets.size(0),
            )
            targets = targets - atomref_energy

        return targets.reshape(targets.size(0), -1).clone()

    @rank_zero_only
    def _standardize(self) -> None:
        """Compute mean and std of training targets for later normalization.

        Uses *val* stage behaviour (no shuffling) to iterate the training set exactly once.
        """
        loader = tqdm(
            self._build_dataloader(self.train_dataset, stage="val", cache=False),
            desc="computing mean and std",
        )
        try:
            atomref = (
                self.atomref if self.hparams.get("prior_model") == "Atomref" else None
            )
            targets = torch.cat(
                [self._extract_targets(batch, atomref) for batch in loader],
                dim=0,
            )
        except MissingLabelException:
            rank_zero_warn(
                "Standardization was requested but labels were unavailable. "
                "This dataset may contain forces only."
            )
            return

        self._mean = targets.mean().item()
        self._std = targets.std().item()
        log.info("Standardization stats: mean=%s, std=%s", self._mean, self._std)

    # ── Split helpers ───────────────────────────────────────────────────

    def _splits_path(self) -> str:
        return os.path.join(self.hparams["output_dir"], "splits.npz")

    def _random_split(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a fresh random train/val/test split or load a pre-saved one."""
        return make_splits(
            len(self.dataset),
            self.hparams["train_size"],
            self.hparams["val_size"],
            None,
            self.hparams["seed"],
            self._splits_path(),
            self.hparams["splits"],
        )

    # ── Dataset-specific preparation ────────────────────────────────────

    def _prepare_qm9(self):
        transform = normalize_positions if self.hparams["normalize_positions"] else None
        if transform is not None:
            log.warning("Normalizing QM9 positions before batching.")
        self.dataset = QM9(
            root=self.hparams["dataset_root"],
            label=self.hparams["label"],
            transform=transform,
        )
        return self._random_split()

    def _prepare_md22(self):
        self.dataset = MD22(
            root=self.hparams["dataset_root"],
            label=self.hparams["label"],
        )
        return self._random_split()

    def _prepare_molecule3d(self):
        split_config = self.hparams.get("split_config", "Molecule3D_random_split")
        self.dataset = Molecule3D(
            root=self.hparams["dataset_root"],
            label=self.hparams["label"],
            split_config=split_config,
        )
        train_full, val_full, test_full = self.dataset.get_split()
        return (
            self._subset_indices(
                torch.as_tensor(train_full, dtype=torch.long),
                self.hparams["train_size"],
                "train",
            ),
            self._subset_indices(
                torch.as_tensor(val_full, dtype=torch.long),
                self.hparams["val_size"],
                "val",
            ),
            self._subset_indices(
                torch.as_tensor(test_full, dtype=torch.long),
                self.hparams["test_size"],
                "test",
            ),
        )

    def _prepare_rmd17(self):
        self.dataset = rMD17(
            root=self.hparams["dataset_root"],
            label=self.hparams["label"],
        )

        splits = self.hparams.get("splits")
        if splits is None:
            return self._random_split()

        # Use a pre-defined train/test split from the dataset's CSV files.
        idx_train_pre, idx_test_pre = self.dataset.get_split(splits)

        # Further split the pre-defined training partition into train + validation.
        idx_train_local, idx_val_local, _ = make_splits(
            len(idx_train_pre),
            self.hparams["train_size"],
            self.hparams["val_size"],
            None,
            self.hparams["seed"],
            self._splits_path(),
            splits=None,
        )
        return (
            torch.as_tensor(idx_train_pre, dtype=torch.long)[idx_train_local],
            torch.as_tensor(idx_train_pre, dtype=torch.long)[idx_val_local],
            self._subset_indices(
                torch.as_tensor(idx_test_pre, dtype=torch.long),
                self.hparams["test_size"],
                "test",
            ),
        )
