import json
import os
import os.path as osp
import shlex
import subprocess

import requests
import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - handled with a runtime error in process()
    pq = None


class MD22(InMemoryDataset):
    """MD22 molecular dynamics dataset backed by ColabFit parquet shards.

    The original `quantum-machine.org` MD22 `.npz` mirrors are no longer a
    reliable source for every molecule. ColabFit republishes each MD22
    molecule as a Hugging Face dataset with parquet files containing the
    fields this project needs: `atomic_numbers`, `positions`, `energy`,
    and `atomic_forces`.

    Available molecules:
        AT-AT, AT-AT-CG-CG, Ac-Ala3-NHMe, DHA, buckycatcher,
        double-walled nanotube, stachyose
    """

    hf_api_base = "https://huggingface.co/api/datasets"
    hf_resolve_base = "https://huggingface.co/datasets"
    download_chunk_size = 512 * 1024
    download_timeout = (30, 600)
    process_batch_size = 256

    molecule_repos = dict(
        at_at="colabfit/MD22_AT_AT",
        at_at_cg_cg="colabfit/MD22_AT_AT_CG_CG",
        ac_ala3_nhme="colabfit/MD22_Ac_Ala3_NHMe",
        dha="colabfit/MD22_DHA",
        buckycatcher="colabfit/MD22_buckyball_catcher",
        double_walled_nanotube="colabfit/MD22_double_walled_nanotube",
        stachyose="colabfit/MD22_stachyose",
    )

    molecule_aliases = {
        "buckyball_catcher": "buckycatcher",
        "double_walled_nanotube": "double_walled_nanotube",
    }

    available_molecules = list(molecule_repos.keys())
    required_columns = ("atomic_numbers", "positions", "energy", "atomic_forces")

    def __init__(self, root, transform=None, pre_transform=None, dataset_arg=None):
        """Initialize the MD22 dataset.

        Args:
            root (str): Root directory where the dataset should be stored.
            transform: Transform applied to each data object at access time.
            pre_transform: Transform applied to each data object before saving.
            dataset_arg (str): Comma-separated list of molecule names, or
                'all' to use every available molecule.
        """
        assert dataset_arg is not None, (
            "Please provide the desired comma separated molecule(s) through "
            f"'dataset_arg'. Available molecules are {', '.join(MD22.available_molecules)} "
            "or 'all' to train on the combined dataset."
        )

        if dataset_arg == "all":
            dataset_arg = ",".join(MD22.available_molecules)

        requested_molecules = [self._normalize_molecule_name(mol) for mol in dataset_arg.split(",")]
        unknown_molecules = [
            mol for mol in requested_molecules if mol not in MD22.available_molecules
        ]
        if unknown_molecules:
            raise ValueError(
                "Unknown MD22 molecule(s): "
                f"{', '.join(unknown_molecules)}. "
                f"Available molecules: {', '.join(MD22.available_molecules)}."
            )

        self.molecules = list(dict.fromkeys(requested_molecules))
        self.dataset_tag = "+".join(self.molecules)

        if len(self.molecules) > 1:
            rank_zero_warn(
                "MD22 molecules have different reference energies, "
                "which is not accounted for during training."
            )

        super().__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def len(self):
        return super().len()

    @property
    def raw_file_names(self):
        return [osp.join("md22", mol, "manifest.json") for mol in self.molecules]

    @property
    def processed_file_names(self):
        return [f"md22-{self.dataset_tag}.pt"]

    @classmethod
    def _normalize_molecule_name(cls, molecule_name):
        normalized = molecule_name.strip().lower().replace("-", "_").replace(" ", "_")
        return cls.molecule_aliases.get(normalized, normalized)

    @staticmethod
    def _request_headers():
        return {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }

    @classmethod
    def _curl_args(cls):
        return [
            "curl",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            str(cls.download_timeout[0]),
            "--max-time",
            str(cls.download_timeout[1]),
            "--retry",
            "5",
            "--retry-delay",
            "2",
            "-A",
            cls._request_headers()["User-Agent"],
        ]

    @classmethod
    def _run_curl(cls, args, capture_output=False):
        command = " ".join(shlex.quote(arg) for arg in args)
        shell_path = os.environ.get("SHELL", "/bin/zsh")
        return subprocess.run(
            [shell_path, "-lc", command],
            check=True,
            capture_output=capture_output,
            text=capture_output,
        )

    @classmethod
    def _curl_get_json(cls, url):
        result = cls._run_curl([*cls._curl_args(), url], capture_output=True)
        return json.loads(result.stdout)

    @classmethod
    def _curl_download_file(cls, url, path):
        cls._run_curl([*cls._curl_args(), "-o", path, url])

    @classmethod
    def _dataset_api_url(cls, repo_id):
        return f"{cls.hf_api_base}/{repo_id}"

    @classmethod
    def _dataset_file_url(cls, repo_id, revision, relative_path):
        return f"{cls.hf_resolve_base}/{repo_id}/resolve/{revision}/{relative_path}"

    def _raw_molecule_dir(self, molecule_name):
        return osp.join(self.raw_dir, "md22", molecule_name)

    def _manifest_path(self, molecule_name):
        return osp.join(self._raw_molecule_dir(molecule_name), "manifest.json")

    @classmethod
    def _fetch_repo_manifest(cls, repo_id):
        try:
            response = requests.get(
                cls._dataset_api_url(repo_id),
                headers=cls._request_headers(),
                timeout=cls.download_timeout,
            )
            response.raise_for_status()
            metadata = response.json()
        except requests.RequestException as exc:
            rank_zero_warn(
                f"Requests failed to fetch MD22 metadata for {repo_id} ({exc}). "
                "Falling back to curl."
            )
            metadata = cls._curl_get_json(cls._dataset_api_url(repo_id))

        parquet_files = sorted(
            sibling["rfilename"]
            for sibling in metadata.get("siblings", [])
            if sibling["rfilename"].startswith("co/") and sibling["rfilename"].endswith(".parquet")
        )
        if not parquet_files:
            raise RuntimeError(f"No parquet shards were found for MD22 repo {repo_id}.")

        return {
            "repo_id": repo_id,
            "revision": metadata.get("sha", "main"),
            "parquet_files": parquet_files,
        }

    @staticmethod
    def _load_manifest(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    @classmethod
    def _is_valid_parquet_cache(cls, path):
        try:
            if not osp.exists(path) or osp.getsize(path) < 8:
                return False
            with open(path, "rb") as handle:
                header = handle.read(4)
                handle.seek(-4, os.SEEK_END)
                footer = handle.read(4)
            return header == b"PAR1" and footer == b"PAR1"
        except OSError:
            return False

    @classmethod
    def _manifest_complete(cls, manifest_path):
        if not osp.exists(manifest_path):
            return False

        try:
            manifest = cls._load_manifest(manifest_path)
        except (OSError, ValueError):
            return False

        molecule_dir = osp.dirname(manifest_path)
        parquet_files = manifest.get("parquet_files", [])
        if not parquet_files:
            return False

        for relative_path in parquet_files:
            local_path = osp.join(molecule_dir, relative_path)
            if not osp.exists(local_path) or osp.getsize(local_path) == 0:
                return False
            if relative_path.endswith(".parquet") and not cls._is_valid_parquet_cache(local_path):
                return False

        return True

    @classmethod
    def _download_single_file(cls, url, path, description):
        try:
            with requests.get(
                url,
                headers=cls._request_headers(),
                allow_redirects=True,
                stream=True,
                timeout=cls.download_timeout,
            ) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))

                with open(path, "wb") as handle, tqdm(
                    desc=description,
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=cls.download_chunk_size):
                        if chunk:
                            handle.write(chunk)
                            bar.update(len(chunk))
        except requests.RequestException as exc:
            rank_zero_warn(
                f"Requests failed to download {description} ({exc}). "
                "Falling back to curl."
            )
            cls._curl_download_file(url, path)

        if osp.getsize(path) == 0:
            raise RuntimeError(f"Downloaded file {path} is empty.")

    def download(self):
        """Download the selected MD22 molecule parquet shards from Hugging Face."""
        os.makedirs(self.raw_dir, exist_ok=True)

        for molecule_name in self.molecules:
            manifest_path = self._manifest_path(molecule_name)
            if self._manifest_complete(manifest_path):
                continue

            repo_id = self.molecule_repos[molecule_name]
            manifest = self._fetch_repo_manifest(repo_id)
            molecule_dir = self._raw_molecule_dir(molecule_name)
            os.makedirs(molecule_dir, exist_ok=True)

            for relative_path in manifest["parquet_files"]:
                local_path = osp.join(molecule_dir, relative_path)
                os.makedirs(osp.dirname(local_path), exist_ok=True)

                if (
                    osp.exists(local_path)
                    and osp.getsize(local_path) > 0
                    and (
                        not relative_path.endswith(".parquet")
                        or self._is_valid_parquet_cache(local_path)
                    )
                ):
                    continue

                if osp.exists(local_path):
                    os.unlink(local_path)

                url = self._dataset_file_url(repo_id, manifest["revision"], relative_path)
                try:
                    self._download_single_file(
                        url=url,
                        path=local_path,
                        description=f"{molecule_name}:{osp.basename(relative_path)}",
                    )
                except Exception:
                    if osp.exists(local_path):
                        os.unlink(local_path)
                    raise

                if relative_path.endswith(".parquet") and not self._is_valid_parquet_cache(local_path):
                    if osp.exists(local_path):
                        os.unlink(local_path)
                    raise RuntimeError(
                        f"Downloaded parquet shard is incomplete or corrupted: {local_path}"
                    )

            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)

    @classmethod
    def _validate_parquet_columns(cls, parquet_path, parquet_file):
        if hasattr(parquet_file, "schema_arrow"):
            available_columns = set(parquet_file.schema_arrow.names)
        else:
            available_columns = set(parquet_file.schema.to_arrow_schema().names)
        missing_columns = [
            column_name for column_name in cls.required_columns if column_name not in available_columns
        ]
        if missing_columns:
            raise RuntimeError(
                f"Missing required MD22 columns in {parquet_path}: {', '.join(missing_columns)}."
            )

    def process(self):
        """Process ColabFit MD22 parquet shards into PyTorch Geometric Data objects."""
        if pq is None:
            raise ImportError(
                "Processing MD22 parquet data requires `pyarrow`. "
                "Install it with `pip install pyarrow`."
            )

        samples = []

        for molecule_name in self.molecules:
            manifest_path = self._manifest_path(molecule_name)
            if not self._manifest_complete(manifest_path):
                rank_zero_warn(
                    f"MD22 cache for {molecule_name} is incomplete or corrupted. "
                    "Re-downloading raw parquet shards."
                )
                self.download()

            manifest = self._load_manifest(manifest_path)
            molecule_dir = self._raw_molecule_dir(molecule_name)

            for relative_path in manifest["parquet_files"]:
                parquet_path = osp.join(molecule_dir, relative_path)
                parquet_file = pq.ParquetFile(parquet_path)
                self._validate_parquet_columns(parquet_path, parquet_file)

                total_rows = parquet_file.metadata.num_rows
                row_iterator = parquet_file.iter_batches(
                    batch_size=self.process_batch_size,
                    columns=list(self.required_columns),
                )

                with tqdm(total=total_rows, desc=f"process {molecule_name}") as bar:
                    for batch in row_iterator:
                        batch_dict = batch.to_pydict()
                        batch_size = len(batch_dict["energy"])

                        for atomic_numbers, positions, energy, forces in zip(
                            batch_dict["atomic_numbers"],
                            batch_dict["positions"],
                            batch_dict["energy"],
                            batch_dict["atomic_forces"],
                        ):
                            data = Data(
                                z=torch.tensor(atomic_numbers, dtype=torch.long),
                                pos=torch.tensor(positions, dtype=torch.float32),
                                y=torch.tensor([[energy]], dtype=torch.float32),
                                dy=torch.tensor(forces, dtype=torch.float32),
                            )

                            if self.pre_filter is not None and not self.pre_filter(data):
                                continue

                            if self.pre_transform is not None:
                                data = self.pre_transform(data)

                            samples.append(data)

                        bar.update(batch_size)

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])
