"""Molecule3D dataset — ~3.9M molecules with DFT properties from Hugging Face."""

from __future__ import annotations

import json
import os
import os.path as osp

import requests
import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch_geometric.data import Data, InMemoryDataset

from src.data.datasets.utils import (
    USER_AGENT,
    DOWNLOAD_TIMEOUT,
    DOWNLOAD_CHUNK,
    is_manifest_complete,
    stream_download,
)

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None

try:
    from rdkit import Chem
except ImportError:
    Chem = None

_PROCESS_BATCH = 1024

# Property index → name mapping.
PROPERTY_NAMES: dict[int, str] = {
    0: "homo",
    1: "lumo",
    2: "gap",
    3: "scf_energy",
    4: "dipole_x",
    5: "dipole_y",
    6: "dipole_z",
}

PROPERTY_LABELS = list(PROPERTY_NAMES.values())
_PROPERTY_LABEL_TO_IDX = {label: idx for idx, label in PROPERTY_NAMES.items()}

# Internal → parquet column name.
_PROPERTY_COLUMN: dict[str, str] = {
    "homo": "homo",
    "lumo": "lumo",
    "gap": "Y",
    "scf_energy": "scf energy",
    "dipole_x": "dipole x",
    "dipole_y": "dipole y",
    "dipole_z": "dipole z",
}


class Molecule3D(InMemoryDataset):
    """Molecule3D dataset (~3.9M molecules) with DFT properties.

    Fetches parquet shards from ``maomlab/Molecule3D`` on Hugging Face.
    Properties: homo, lumo, gap, scf_energy, dipole_x, dipole_y, dipole_z.

    Supports ``Molecule3D_random_split`` and ``Molecule3D_scaffold_split``
    split configurations.
    """

    available_properties = PROPERTY_LABELS
    available_configs = ["Molecule3D_random_split", "Molecule3D_scaffold_split"]

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        label=None,
        split_config="Molecule3D_random_split",
    ):
        if label is None:
            raise ValueError(
                f"Pass a property via 'label'. Available: {', '.join(self.available_properties)}"
            )
        if label not in self.available_properties:
            raise ValueError(f"Unknown property '{label}'.")
        if split_config not in self.available_configs:
            raise ValueError(f"Unknown split_config '{split_config}'.")

        self.label = label
        self.split_config = split_config
        self._target_column = _PROPERTY_COLUMN[label]

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @staticmethod
    def _normalize_relpath(split_config: str, rel_path: str) -> str:
        """Map Hugging Face split-prefixed paths into config-local filenames."""
        normalized = rel_path.replace("\\", "/").lstrip("./")
        prefix = f"{split_config}/"
        if normalized.startswith(prefix):
            return normalized[len(prefix) :]
        return normalized

    def _config_dir(self) -> str:
        return osp.join(self.raw_dir, self.split_config)

    def _manifest_path(self) -> str:
        return osp.join(self._config_dir(), "manifest.json")

    @property
    def raw_file_names(self):
        return [osp.join(self.split_config, "manifest.json")]

    @property
    def processed_file_names(self):
        return [
            f"molecule3d-{self.split_config}-{self.label}.pt",
            f"molecule3d-{self.split_config}-{self.label}-splits.json",
        ]

    @staticmethod
    def label_to_idx(label):
        try:
            return _PROPERTY_LABEL_TO_IDX[label]
        except KeyError as exc:
            raise ValueError(f"Unknown Molecule3D property '{label}'.") from exc

    def get_split(self):
        """Load cached train/val/test index lists from the processed split file."""
        path = self.processed_paths[1]
        if not osp.exists(path):
            raise RuntimeError(
                "Dataset must be processed before splits can be retrieved."
            )
        with open(path) as f:
            s = json.load(f)
        return s["idx_train"], s["idx_val"], s["idx_test"]

    @staticmethod
    def get_atomref(max_z=100):
        return None

    # ── Download ────────────────────────────────────────────────────────

    @classmethod
    def _fetch_manifest(cls):
        """Fetch Hugging Face dataset metadata for all parquet shards."""
        url = "https://huggingface.co/api/datasets/maomlab/Molecule3D"
        resp = requests.get(
            url, headers={"User-Agent": USER_AGENT}, timeout=DOWNLOAD_TIMEOUT
        )
        resp.raise_for_status()
        meta = resp.json()
        return {
            "revision": meta.get("sha", "main"),
            "parquet_files": sorted(
                s["rfilename"]
                for s in meta.get("siblings", [])
                if s["rfilename"].endswith(".parquet")
            ),
        }

    def download(self):
        """Download parquet shards for the configured split."""
        os.makedirs(self.raw_dir, exist_ok=True)

        manifest_path = self._manifest_path()
        if is_manifest_complete(manifest_path):
            return

        manifest = self._fetch_manifest()
        prefix = f"{self.split_config}/"
        remote_files = sorted(
            f for f in manifest["parquet_files"] if f.startswith(prefix)
        )
        if not remote_files:
            raise RuntimeError(f"No parquet files for config '{self.split_config}'.")

        cfg_dir = self._config_dir()
        config_files = []
        for remote_rel in remote_files:
            rel = self._normalize_relpath(self.split_config, remote_rel)
            config_files.append(rel)
            local = osp.join(cfg_dir, rel)
            if osp.exists(local) and osp.getsize(local) > 0:
                continue
            os.makedirs(osp.dirname(local), exist_ok=True)

            url = f"https://huggingface.co/datasets/maomlab/Molecule3D/resolve/{manifest['revision']}/{remote_rel}"
            stream_download(
                url,
                local,
                description=osp.basename(rel),
                headers={"User-Agent": USER_AGENT},
                timeout=DOWNLOAD_TIMEOUT,
                chunk_size=DOWNLOAD_CHUNK,
            )

        with open(manifest_path, "w") as f:
            json.dump(
                {"revision": manifest["revision"], "parquet_files": config_files},
                f,
                indent=2,
            )

    # ── SDF parsing ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_sdf(sdf_string: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Parse an SDF string into (atomic_numbers, positions).

        Returns ``(None, None)`` on any failure.
        """
        if not sdf_string:
            return None, None

        try:
            lines = sdf_string.splitlines()
            # Pad header when the V2000/V3000 marker appears before line 4.
            for i, line in enumerate(lines[:4]):
                if "V2000" in line or "V3000" in line:
                    if i < 3:
                        lines = [""] * (3 - i) + lines
                        sdf_string = "\n".join(lines)
                    break

            mol = Chem.MolFromMolBlock(sdf_string, removeHs=False, sanitize=True)
            if mol is None:
                return None, None

            conf = mol.GetConformer()
            return (
                torch.tensor(
                    [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long
                ),
                torch.tensor(conf.GetPositions(), dtype=torch.float32),
            )
        except Exception:
            return None, None

    # ── Processing ──────────────────────────────────────────────────────

    def process(self):
        """Convert parquet shards to PyTorch Geometric Data + split index file."""
        if pq is None:
            raise ImportError("Install pyarrow: pip install pyarrow")
        if Chem is None:
            raise ImportError("Install rdkit: pip install rdkit")

        manifest_path = self._manifest_path()
        if not is_manifest_complete(manifest_path):
            rank_zero_warn("Cache incomplete; re-downloading.")
            self.download()

        with open(manifest_path) as f:
            manifest = json.load(f)

        cfg_dir = self._config_dir()

        # Group files by split (train / validation / test).
        split_files: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
        for rel in manifest["parquet_files"]:
            base = osp.basename(rel)
            if base.startswith("train"):
                split_files["train"].append(rel)
            elif base.startswith("validation"):
                split_files["validation"].append(rel)
            elif base.startswith("test"):
                split_files["test"].append(rel)

        samples = []
        boundaries: dict[str, int] = {"train": 0}

        for split in ("train", "validation", "test"):
            for rel in split_files.get(split, []):
                pf = pq.ParquetFile(osp.join(cfg_dir, rel))
                for batch in pf.iter_batches(batch_size=_PROCESS_BATCH):
                    rows = batch.to_pydict()
                    for sdf_str, val in zip(
                        rows["sdf"],
                        rows.get(self._target_column, [None] * len(rows["sdf"])),
                    ):
                        if val is None:
                            continue
                        z, pos = self._parse_sdf(sdf_str)
                        if z is None:
                            continue
                        data = Data(
                            z=z,
                            pos=pos,
                            y=torch.tensor([[float(val)]], dtype=torch.float32),
                        )
                        if self.pre_filter is not None and not self.pre_filter(data):
                            continue
                        if self.pre_transform is not None:
                            data = self.pre_transform(data)
                        samples.append(data)
            boundaries[split] = len(samples)

        torch.save(self.collate(samples), self.processed_paths[0])

        split_indices = {
            "idx_train": list(range(0, boundaries["train"])),
            "idx_val": list(range(boundaries["train"], boundaries["validation"])),
            "idx_test": list(range(boundaries["validation"], boundaries["test"])),
        }
        with open(self.processed_paths[1], "w") as f:
            json.dump(split_indices, f)
