"""MD22 molecular dynamics dataset backed by ColabFit parquet shards."""

from __future__ import annotations

import json
import os.path as osp

import requests
import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch_geometric.data import Data, InMemoryDataset
import pyarrow.parquet as pq

from src.data.datasets.utils import (
    USER_AGENT,
    DOWNLOAD_TIMEOUT,
    DOWNLOAD_CHUNK,
    parse_csv_selection,
    stream_download,
    is_manifest_complete,
)

_PROCESS_BATCH = 256


class MD22(InMemoryDataset):
    """MD22 molecular dynamics dataset backed by ColabFit parquet shards.

    Available molecules: ``at_at``, ``at_at_cg_cg``, ``ac_ala3_nhme``,
    ``dha``, ``buckycatcher``, ``double_walled_nanotube``, ``stachyose``.
    Each sample includes energy (``y``) and forces (``dy``).
    """

    molecule_repos = dict(
        at_at="colabfit/MD22_AT_AT",
        at_at_cg_cg="colabfit/MD22_AT_AT_CG_CG",
        ac_ala3_nhme="colabfit/MD22_Ac_Ala3_NHMe",
        dha="colabfit/MD22_DHA",
        buckycatcher="colabfit/MD22_buckyball_catcher",
        double_walled_nanotube="colabfit/MD22_double_walled_nanotube",
        stachyose="colabfit/MD22_stachyose",
    )

    available_molecules = list(molecule_repos.keys())
    _required_columns = ("atomic_numbers", "positions", "energy", "atomic_forces")

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
                "MD22 molecules have different reference energies, "
                "which is not accounted for during training."
            )

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [osp.join("md22", m, "manifest.json") for m in self.molecules]

    @property
    def processed_file_names(self):
        return [f"md22-{self._dataset_tag}.pt"]

    # ── Download ────────────────────────────────────────────────────────

    @classmethod
    def _fetch_manifest(cls, repo_id):
        """Fetch Hugging Face dataset metadata and list parquet shards."""
        url = f"https://huggingface.co/api/datasets/{repo_id}"
        resp = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=DOWNLOAD_TIMEOUT
        )
        resp.raise_for_status()
        meta = resp.json()
        parquet_files = sorted(
            s["rfilename"]
            for s in meta.get("siblings", [])
            if s["rfilename"].startswith("co/") and s["rfilename"].endswith(".parquet")
        )
        if not parquet_files:
            raise RuntimeError(f"No parquet shards found for {repo_id}.")
        return {
            "repo_id": repo_id,
            "revision": meta.get("sha", "main"),
            "parquet_files": parquet_files,
        }

    def download(self):
        """Download MD22 parquet shards from Hugging Face."""
        for mol in self.molecules:
            manifest_path = osp.join(self.raw_dir, "md22", mol, "manifest.json")
            if is_manifest_complete(manifest_path):
                continue

            manifest = self._fetch_manifest(self.molecule_repos[mol])
            mol_dir = osp.join(self.raw_dir, "md22", mol)

            for rel_path in manifest["parquet_files"]:
                local = osp.join(mol_dir, rel_path)
                if osp.exists(local) and osp.getsize(local) > 0:
                    continue
                url = (
                    f"https://huggingface.co/datasets/{manifest['repo_id']}"
                    f"/resolve/{manifest['revision']}/{rel_path}"
                )
                stream_download(
                    url,
                    local,
                    description=f"{mol}:{osp.basename(rel_path)}",
                    headers={"User-Agent": USER_AGENT},
                    timeout=DOWNLOAD_TIMEOUT,
                    chunk_size=DOWNLOAD_CHUNK,
                )

            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

    # ── Processing ──────────────────────────────────────────────────────

    def process(self):
        """Convert parquet shards to PyTorch Geometric Data objects."""
        samples = []
        for mol in self.molecules:
            manifest_path = osp.join(self.raw_dir, "md22", mol, "manifest.json")
            if not is_manifest_complete(manifest_path):
                rank_zero_warn(f"Cache for {mol} is incomplete; re-downloading.")
                self.download()

            with open(manifest_path) as f:
                manifest = json.load(f)

            mol_dir = osp.join(self.raw_dir, "md22", mol)
            for rel_path in manifest["parquet_files"]:
                pf = pq.ParquetFile(osp.join(mol_dir, rel_path))
                for batch in pf.iter_batches(
                    batch_size=_PROCESS_BATCH, columns=list(self._required_columns)
                ):
                    rows = batch.to_pydict()
                    for z, pos, e, f in zip(
                        rows["atomic_numbers"],
                        rows["positions"],
                        rows["energy"],
                        rows["atomic_forces"],
                    ):
                        data = Data(
                            z=torch.tensor(z, dtype=torch.long),
                            pos=torch.tensor(pos, dtype=torch.float32),
                            y=torch.tensor([[e]], dtype=torch.float32),
                            dy=torch.tensor(f, dtype=torch.float32),
                        )
                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue
                        if self.pre_transform is not None:
                            data = self.pre_transform(data)
                        samples.append(data)

        torch.save(self.collate(samples), self.processed_paths[0])
