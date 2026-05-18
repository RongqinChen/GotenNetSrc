"""Unified Lightning datamodule for all supported molecular datasets."""

from __future__ import annotations

import os.path
from typing import Any, Callable, Optional

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


class DataModule(LightningDataModule):
    """Lightning datamodule with shared logic across all project datasets."""

    _DATASET_CLASSES = {
        "QM9": QM9,
        "MD22": MD22,
        "Molecule3D": Molecule3D,
        "rMD17": rMD17,
    }

    def __init__(self, hparams: dict[str, Any] | Any):
        super().__init__()

        if hasattr(hparams, "items"):
            params = dict(hparams.items())
        elif hasattr(hparams, "__dict__"):
            params = dict(hparams.__dict__)
        else:
            params = dict(hparams)
        self.hparams.update(params)

        self._mean: float | None = None
        self._std: float | None = None
        self._saved_dataloaders: dict[str, DataLoader] = {}
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.loaded = False

    @property
    def dataset_class(self):
        """Return the dataset class for the current datamodule configuration."""
        dataset_type = self.hparams["dataset"]
        if dataset_type not in self._DATASET_CLASSES:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")
        return self._DATASET_CLASSES[dataset_type]

    @property
    def atomref(self):
        """Return atom reference values when the dataset provides them."""
        if self.dataset is not None and hasattr(self.dataset, "get_atomref"):
            return self.dataset.get_atomref()
        return None

    @property
    def mean(self) -> float | None:
        """Return the standardized target mean when available."""
        return self._mean

    @property
    def std(self) -> float | None:
        """Return the standardized target std when available."""
        return self._std

    def _ensure_dataset_loaded(self) -> None:
        if not self.loaded:
            self.prepare_dataset()
            self.loaded = True

    def get_metadata(self, label: Optional[str] = None) -> dict[str, Any]:
        """Return dataset metadata needed by task heads."""
        if label is not None:
            self.hparams["dataset_arg"] = label

        self._ensure_dataset_loaded()
        return {
            "atomref": self.atomref,
            "dataset": self.dataset,
            "mean": self.mean,
            "std": self.std,
        }

    def prepare_dataset(self) -> None:
        """Load the configured dataset and materialize train/val/test subsets."""
        dataset_type = self.hparams["dataset"]
        preparer = self._dataset_preparers().get(dataset_type)
        if preparer is None:
            raise ValueError(f"Dataset {dataset_type} is not supported.")

        self.idx_train, self.idx_val, self.idx_test = preparer()
        log.info(
            "Prepared splits: train=%s, val=%s, test=%s",
            len(self.idx_train),
            len(self.idx_val),
            len(self.idx_test),
        )

        self.train_dataset = self.dataset[self.idx_train]
        self.val_dataset = self.dataset[self.idx_val]
        self.test_dataset = self.dataset[self.idx_test]

        if self.hparams["standardize"]:
            self._standardize()

    def _dataset_preparers(self) -> dict[str, Callable[[], tuple[torch.Tensor, ...]]]:
        return {
            "QM9": self._prepare_qm9,
            "MD22": self._prepare_md22,
            "Molecule3D": self._prepare_molecule3d,
            "rMD17": self._prepare_rmd17,
        }

    def train_dataloader(self):
        self._ensure_dataset_loaded()
        return self._get_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        self._ensure_dataset_loaded()
        return self._get_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        self._ensure_dataset_loaded()
        return self._get_dataloader(self.test_dataset, "test")

    def _get_dataloader(self, dataset, stage: str, store_dataloader: bool = True):
        """Build or reuse a dataloader for a given dataset split."""
        store_dataloader = store_dataloader and not self.hparams["reload"]
        if stage in self._saved_dataloaders and store_dataloader:
            return self._saved_dataloaders[stage]

        if stage == "train":
            batch_size = self.hparams["batch_size"]
            shuffle = True
        elif stage in {"val", "test"}:
            batch_size = self.hparams["inference_batch_size"]
            shuffle = False
        else:
            raise ValueError(f"Unsupported dataloader stage: {stage}")

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
        )

        if store_dataloader:
            self._saved_dataloaders[stage] = dataloader
        return dataloader

    @rank_zero_only
    def _standardize(self) -> None:
        """Compute target statistics from the training set when labels are available."""

        def get_label(batch, atomref):
            if batch.y is None:
                raise MissingLabelException()

            force_targets = batch.dy.squeeze().clone() if "dy" in batch else None
            if atomref is None:
                return batch.y.clone(), force_targets

            atomref_energy = scatter(atomref[batch.z], batch.batch, dim=0)
            return (batch.y.squeeze() - atomref_energy.squeeze()).clone(), force_targets

        dataloader = tqdm(
            self._get_dataloader(self.train_dataset, "val", store_dataloader=False),
            desc="computing mean and std",
        )
        try:
            atomref = self.atomref if self.hparams.get("prior_model") == "Atomref" else None
            values = [get_label(batch, atomref) for batch in dataloader]
            targets, _ = zip(*values, strict=False)
            targets = torch.cat(targets)
        except MissingLabelException:
            rank_zero_warn(
                "Standardization was requested but labels were unavailable. "
                "This dataset may contain forces only."
            )
            return

        self._mean = targets.mean(dim=0)[0].item()
        self._std = targets.std(dim=0)[0].item()
        log.info("Computed standardization stats: mean=%s, std=%s", self._mean, self._std)

    def _split_output_path(self) -> str:
        return os.path.join(self.hparams["output_dir"], "splits.npz")

    def _random_split_dataset(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return make_splits(
            len(self.dataset),
            self.hparams["train_size"],
            self.hparams["val_size"],
            None,
            self.hparams["seed"],
            self._split_output_path(),
            self.hparams["splits"],
        )

    def _prepare_qm9(self):
        transform = normalize_positions if self.hparams["normalize_positions"] else None
        if transform is not None:
            log.warning("Normalizing QM9 positions before batching.")

        self.dataset = QM9(
            root=self.hparams["dataset_root"],
            dataset_arg=self.hparams["dataset_arg"],
            transform=transform,
        )
        return self._random_split_dataset()

    def _prepare_md22(self):
        self.dataset = MD22(
            root=self.hparams["dataset_root"],
            dataset_arg=self.hparams["dataset_arg"],
        )
        return self._random_split_dataset()

    def _prepare_molecule3d(self):
        split_config = self.hparams.get("split_config", "Molecule3D_random_split")
        self.dataset = Molecule3D(
            root=self.hparams["dataset_root"],
            dataset_arg=self.hparams["dataset_arg"],
            split_config=split_config,
        )

        idx_train_full, idx_val_full, idx_test = self.dataset.get_split()
        idx_train_full = torch.tensor(idx_train_full)
        idx_val_full = torch.tensor(idx_val_full)
        idx_test = torch.tensor(idx_test)

        train_size = self.hparams["train_size"]
        val_size = self.hparams["val_size"]

        idx_train = idx_train_full[:train_size] if train_size and train_size < len(idx_train_full) else idx_train_full
        idx_val = idx_val_full[:val_size] if val_size and val_size < len(idx_val_full) else idx_val_full

        log.info(
            "Molecule3D splits: train=%s, val=%s, test=%s",
            len(idx_train),
            len(idx_val),
            len(idx_test),
        )
        return idx_train, idx_val, idx_test

    def _prepare_rmd17(self):
        self.dataset = rMD17(
            root=self.hparams["dataset_root"],
            dataset_arg=self.hparams["dataset_arg"],
        )

        train_size = self.hparams["train_size"]
        val_size = self.hparams["val_size"]
        splits = self.hparams.get("splits")

        if splits is None:
            return self._random_split_dataset()

        split_indices = self.dataset.get_split(splits)
        if len(split_indices) != 2:
            raise ValueError("Expected rMD17 split metadata to contain exactly two partitions.")

        expected_train_val_size = train_size + val_size
        if len(split_indices[0]) != expected_train_val_size:
            raise ValueError(
                "Expected train/validation partition of size "
                f"{expected_train_val_size}, received {len(split_indices[0])}."
            )

        idx_train_local, idx_val_local, _ = make_splits(
            len(split_indices[0]),
            train_size,
            val_size,
            None,
            self.hparams["seed"],
            self._split_output_path(),
            splits=None,
        )
        train_val = torch.tensor(split_indices[0])
        idx_train = train_val[idx_train_local]
        idx_val = train_val[idx_val_local]
        idx_test = split_indices[1]

        log.info(
            "[ID: %s] train=%s, val=%s, test=%s",
            splits,
            len(idx_train),
            len(idx_val),
            len(idx_test),
        )
        return idx_train, idx_val, idx_test
