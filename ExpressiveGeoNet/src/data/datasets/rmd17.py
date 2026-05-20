"""Revised MD17 dataset — molecular dynamics trajectories with energy + forces."""

from __future__ import annotations

import os
import os.path as osp
import tarfile

import numpy as np
import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

from src.data.datasets.utils import (
    USER_AGENT,
    DOWNLOAD_TIMEOUT,
    DOWNLOAD_CHUNK,
    parse_csv_selection,
    stream_download,
)

_REVISED_URL = (
    "https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/23950376/rmd17.tar.bz2"
)

# rMD17 uses a larger chunk size for the compressed archive.
_RMD17_CHUNK = 4 * 1024 * 1024


class rMD17(InMemoryDataset):
    """Revised MD17 dataset with energy + forces for 10 small organic molecules.

    Available molecules: ``aspirin``, ``azobenzene``, ``benzene``, ``ethanol``,
    ``malonaldehyde``, ``naphthalene``, ``paracetamol``, ``salicylic``,
    ``toluene``, ``uracil``.

    Supports 5 pre-defined train/test split configurations accessible via
    :meth:`get_split` (indices 0–4).
    """

    molecule_files = dict(
        aspirin="rmd17_aspirin.npz",
        azobenzene="rmd17_azobenzene.npz",
        benzene="rmd17_benzene.npz",
        ethanol="rmd17_ethanol.npz",
        malonaldehyde="rmd17_malonaldehyde.npz",
        naphthalene="rmd17_naphthalene.npz",
        paracetamol="rmd17_paracetamol.npz",
        salicylic="rmd17_salicylic.npz",
        toluene="rmd17_toluene.npz",
        uracil="rmd17_uracil.npz",
    )

    available_molecules = list(molecule_files.keys())

    def __init__(
        self, root, transform=None, pre_transform=None, pre_filter=None, label=None
    ):
        self.molecules = parse_csv_selection(
            label,
            available=self.available_molecules,
            normalize=str.lower,
            field_name="label",
        )
        self._dataset_tag = "+".join(self.molecules)

        if len(self.molecules) > 1:
            rank_zero_warn(
                "MD17 molecules have different reference energies, "
                "which is not accounted for during training."
            )

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [
            osp.join("rmd17", "npz_data", self.molecule_files[m])
            for m in self.molecules
        ]

    @property
    def processed_file_names(self):
        return [f"rmd17-{self._dataset_tag}.pt"]

    # ── Splits ──────────────────────────────────────────────────────────

    def get_split(self, idx):
        """Return pre-defined train/test index lists for split *idx* (0–4)."""
        idx = int(idx)
        if idx not in range(5):
            raise ValueError("rMD17 split index must be in [0, 4].")

        def _load_csv(name: str) -> list[int]:
            path = osp.join(
                self.root, "raw", "rmd17", "splits", f"{name}_0{idx + 1}.csv"
            )
            with open(path) as f:
                return [int(line.strip()) for line in f]

        return [_load_csv("index_train"), _load_csv("index_test")]

    # ── Download ────────────────────────────────────────────────────────

    def _is_downloaded(self) -> bool:
        split_dir = osp.join(self.raw_dir, "rmd17", "splits")
        return osp.isdir(split_dir) and all(
            osp.exists(p) and osp.getsize(p) > 0 for p in self.raw_paths
        )

    def download(self):
        """Download and extract rMD17 tar.bz2 archive from figshare."""
        if self._is_downloaded():
            return

        os.makedirs(self.raw_dir, exist_ok=True)
        archive = osp.join(self.raw_dir, "rmd17.tar.bz2")

        stream_download(
            _REVISED_URL,
            archive,
            description="rmd17.tar.bz2",
            headers={"User-Agent": USER_AGENT},
            timeout=DOWNLOAD_TIMEOUT,
            chunk_size=_RMD17_CHUNK,
        )

        with tarfile.open(archive, mode="r:bz2") as tf:
            tf.extractall(self.raw_dir)

        os.unlink(archive)

    # ── Processing ──────────────────────────────────────────────────────

    def process(self):
        """Convert raw .npz files to PyTorch Geometric Data objects."""
        samples = []
        for raw_path in self.raw_paths:
            data = np.load(raw_path)
            atomic_numbers = torch.from_numpy(data["nuclear_charges"]).long()
            positions = torch.from_numpy(data["coords"]).float()
            energies = torch.from_numpy(data["energies"]).float().unsqueeze_(1)
            forces = torch.from_numpy(data["forces"]).float()

            for pos, energy, force in tqdm(
                zip(positions, energies, forces), total=len(energies)
            ):
                sample = Data(z=atomic_numbers, pos=pos, y=energy, dy=force)

                if self.pre_filter is not None and not self.pre_filter(sample):
                    continue
                if self.pre_transform is not None:
                    sample = self.pre_transform(sample)

                samples.append(sample)

        torch.save(self.collate(samples), self.processed_paths[0])
