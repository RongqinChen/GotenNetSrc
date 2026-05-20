"""Train/validation/test splitting utilities for molecular datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_warn


def train_val_test_split(
    dset_len: int,
    train_size: float | int | None,
    val_size: float | int | None,
    test_size: float | int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly split *dset_len* indices into train/val/test subsets.

    Each size may be an absolute count (``int``), a fraction of the dataset
    (``float`` in ``(0, 1]``), or ``None`` (inferred from the other two).
    At most one of the three sizes may be ``None``.

    A single index is trimmed from the last float-specified split when rounding
    causes a total overshoot of *dset_len*.
    """
    none_cnt = (train_size is None) + (val_size is None) + (test_size is None)
    if none_cnt > 1:
        raise ValueError("At most one of train_size, val_size, test_size may be None.")

    # Record which splits were originally floats (needed for rounding correction).
    float_flags = (
        isinstance(train_size, float),
        isinstance(val_size, float),
        isinstance(test_size, float),
    )

    train_size = round(dset_len * train_size) if float_flags[0] else train_size
    val_size = round(dset_len * val_size) if float_flags[1] else val_size
    test_size = round(dset_len * test_size) if float_flags[2] else test_size

    # Infer the (single) missing size from the other two.
    if train_size is None:
        train_size = dset_len - val_size - test_size
    elif val_size is None:
        val_size = dset_len - train_size - test_size
    elif test_size is None:
        test_size = dset_len - train_size - val_size

    total = train_size + val_size + test_size

    # Fix rounding overshoot by trimming one element from a float-specified split.
    if total > dset_len:
        if float_flags[2]:
            test_size -= 1
        elif float_flags[1]:
            val_size -= 1
        elif float_flags[0]:
            train_size -= 1
        total = train_size + val_size + test_size

    if min(train_size, val_size, test_size) < 0:
        raise ValueError(
            f"Negative split size: train={train_size}, val={val_size}, test={test_size}."
        )
    if dset_len < total:
        raise ValueError(
            f"Dataset ({dset_len}) is smaller than the combined split sizes ({total})."
        )
    if total < dset_len:
        rank_zero_warn(f"{dset_len - total} samples were excluded from the dataset.")

    perm = np.random.default_rng(seed).permutation(dset_len)
    return (
        np.array(perm[:train_size]),
        np.array(perm[train_size : train_size + val_size]),
        np.array(perm[train_size + val_size : total]),
    )


def make_splits(
    dataset_len: int,
    train_size: float | int | None,
    val_size: float | int | None,
    test_size: float | int | None,
    seed: int,
    filename: str | None = None,
    splits: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create or load dataset splits, optionally saving to *filename*.

    When *splits* is provided (a path to a ``.npz`` file), the saved indices are
    loaded instead of generating new ones.  Otherwise new indices are produced
    via :func:`train_val_test_split`.
    """
    if splits is not None:
        saved = np.load(splits)
        idx_train, idx_val, idx_test = (
            saved["idx_train"],
            saved["idx_val"],
            saved["idx_test"],
        )
    else:
        idx_train, idx_val, idx_test = train_val_test_split(
            dataset_len,
            train_size,
            val_size,
            test_size,
            seed,
        )

    if filename is not None:
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        np.savez(filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test)

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
        torch.from_numpy(idx_test),
    )


class MissingLabelException(Exception):
    """Raised when a required label or property is absent from the dataset."""

    pass
