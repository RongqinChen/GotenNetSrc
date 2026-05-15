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

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - handled with a runtime error in process()
    Chem = None

molecule3d_property_dict = {
    0: "homo",
    1: "lumo",
    2: "gap",
    3: "scf_energy",
    4: "dipole_x",
    5: "dipole_y",
    6: "dipole_z",
}


class Molecule3D(InMemoryDataset):
    """Molecule3D dataset backed by Hugging Face parquet shards.

    The `maomlab/Molecule3D` dataset on Hugging Face contains ~3.9 million
    molecules with DFT-computed ground-state properties including HOMO, LUMO,
    HOMO-LUMO gap, SCF energy, and dipole moment components. The dataset
    provides pre-defined train/validation/test splits via two configs:
    ``Molecule3D_random_split`` and ``Molecule3D_scaffold_split``.

    Each parquet row contains a SDF string with 3D atomic coordinates that
    is parsed via RDKit to extract atomic numbers and positions.
    """

    hf_repo = "maomlab/Molecule3D"
    hf_api_base = "https://huggingface.co/api/datasets"
    hf_resolve_base = "https://huggingface.co/datasets"
    download_chunk_size = 512 * 1024
    download_timeout = (30, 600)
    process_batch_size = 1024

    homo = "homo"
    lumo = "lumo"
    gap = "gap"
    scf_energy = "scf_energy"
    dipole_x = "dipole_x"
    dipole_y = "dipole_y"
    dipole_z = "dipole_z"

    available_properties = [
        homo,
        lumo,
        gap,
        scf_energy,
        dipole_x,
        dipole_y,
        dipole_z,
    ]

    available_configs = [
        "Molecule3D_random_split",
        "Molecule3D_scaffold_split",
    ]

    # Number of parquet shards per split (same for both configs).
    _split_shard_counts = {
        "train": 7,
        "validation": 3,
        "test": 3,
    }

    # Mapping from internal property name to parquet column name.
    _property_to_column = {
        "homo": "homo",
        "lumo": "lumo",
        "gap": "Y",
        "scf_energy": "scf energy",
        "dipole_x": "dipole x",
        "dipole_y": "dipole y",
        "dipole_z": "dipole z",
    }

    required_columns = ("sdf",)

    def __init__(
        self,
        root: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        dataset_arg: str = None,
        split_config: str = "Molecule3D_random_split",
    ):
        """Initialize the Molecule3D dataset.

        Args:
            root: Root directory where the dataset should be stored.
            transform: Transform applied to each data object at access time.
            pre_transform: Transform applied to each data object before saving.
            pre_filter: Function that takes a data object and returns a boolean,
                indicating whether the item should be included.
            dataset_arg: The property to train on. Must be one of
                ``available_properties``.
            split_config: Which Hugging Face config to use. One of
                ``available_configs``. Default is ``Molecule3D_random_split``.

        Raises:
            AssertionError: If ``dataset_arg`` is None or invalid.
            ValueError: If ``split_config`` is not one of the available configs.
        """
        assert dataset_arg is not None, (
            "Please pass the desired property to "
            'train on via "dataset_arg". Available '
            f'properties are {", ".join(self.available_properties)}.'
        )

        if dataset_arg not in self.available_properties:
            raise ValueError(
                f"Unknown property '{dataset_arg}'. "
                f"Available properties: {', '.join(self.available_properties)}."
            )

        if split_config not in self.available_configs:
            raise ValueError(
                f"Unknown split_config '{split_config}'. "
                f"Available configs: {', '.join(self.available_configs)}."
            )

        self.label = dataset_arg
        label2idx = dict(
            zip(molecule3d_property_dict.values(), molecule3d_property_dict.keys(), strict=False)
        )
        self.label_idx = label2idx[self.label]
        self.split_config = split_config

        self._target_column = self._property_to_column[self.label]

        if transform is None:
            transform = self._filter_label
        else:
            from torch_geometric.transforms import Compose

            transform = Compose([transform, self._filter_label])

        self._split_indices = None

        super().__init__(root, transform, pre_transform, pre_filter)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def len(self):
        return self.data.y.size(0)

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
    def label_to_idx(label: str) -> int:
        """Convert a property label to its corresponding index.

        Args:
            label: The property label to convert.

        Returns:
            The index corresponding to the property label.
        """
        label2idx = dict(
            zip(molecule3d_property_dict.values(), molecule3d_property_dict.keys(), strict=False)
        )
        return label2idx[label]

    def get_split(self):
        """Return pre-defined train/validation/test split indices.

        The split boundaries are determined by the order in which parquet
        shards are processed: train shards first, then validation, then test.

        Returns:
            A tuple ``(idx_train, idx_val, idx_test)`` of index lists.

        Raises:
            RuntimeError: If split information has not been saved yet.
        """
        split_path = self.processed_paths[1]
        if not osp.exists(split_path):
            raise RuntimeError(
                f"Split file {split_path} not found. The dataset must be "
                "processed before splits can be retrieved."
            )
        with open(split_path, "r", encoding="utf-8") as handle:
            split_data = json.load(handle)
        return (
            split_data["idx_train"],
            split_data["idx_val"],
            split_data["idx_test"],
        )

    def mean(self, divide_by_atoms: bool = False) -> float:
        """Calculate the mean of the target property.

        Args:
            divide_by_atoms: If True, normalize the property by atom count.

        Returns:
            The mean value of the target property.
        """
        if not divide_by_atoms:
            get_labels = lambda i: self.get(i).y
        else:
            get_labels = lambda i: self.get(i).y / self.get(i).pos.shape[0]

        y = torch.cat([get_labels(i) for i in range(len(self))], dim=0)
        if len(y.shape) == 2 and y.shape[1] != 1:
            y = y[:, self.label_idx]
        else:
            y = y[:, 0]
        return y.mean().item()

    def std(self, divide_by_atoms: bool = False) -> float:
        """Calculate the standard deviation of the target property.

        Args:
            divide_by_atoms: If True, normalize the property by atom count.

        Returns:
            The standard deviation of the target property.
        """
        if not divide_by_atoms:
            get_labels = lambda i: self.get(i).y
        else:
            get_labels = lambda i: self.get(i).y / self.get(i).pos.shape[0]

        y = torch.cat([get_labels(i) for i in range(len(self))], dim=0)
        if len(y.shape) == 2 and y.shape[1] != 1:
            y = y[:, self.label_idx]
        else:
            y = y[:, 0]
        return y.std().item()

    def min(self, divide_by_atoms: bool = False) -> float:
        """Calculate the minimum of the target property.

        Args:
            divide_by_atoms: If True, normalize the property by atom count.

        Returns:
            The minimum value of the target property.
        """
        if not divide_by_atoms:
            get_labels = lambda i: self.get(i).y
        else:
            get_labels = lambda i: self.get(i).y / self.get(i).pos.shape[0]

        y = torch.cat([get_labels(i) for i in range(len(self))], dim=0)
        if len(y.shape) == 2 and y.shape[1] != 1:
            y = y[:, self.label_idx]
        else:
            y = y[:, 0]
        return y.min().item()

    def get_atomref(self, max_z: int = 100):
        """Get atomic reference values (not available for Molecule3D).

        Args:
            max_z: Maximum atomic number (unused).

        Returns:
            None: Molecule3D does not provide atom reference values.
        """
        return None

    def _filter_label(self, batch):
        """Filter the batch to only include the target property.

        Args:
            batch: A batch of data from the dataset.

        Returns:
            The filtered batch with only the target property.
        """
        batch.y = batch.y[:, self.label_idx].unsqueeze(1)
        return batch

    # ------------------------------------------------------------------
    #  Download helpers
    # ------------------------------------------------------------------

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
    def _dataset_api_url(cls):
        return f"{cls.hf_api_base}/{cls.hf_repo}"

    @classmethod
    def _dataset_file_url(cls, revision, relative_path):
        return f"{cls.hf_resolve_base}/{cls.hf_repo}/resolve/{revision}/{relative_path}"

    def _config_raw_dir(self):
        return osp.join(self.raw_dir, self.split_config)

    def _manifest_path(self):
        return osp.join(self._config_raw_dir(), "manifest.json")

    @classmethod
    def _fetch_repo_manifest(cls):
        """Fetch repository metadata from the Hugging Face API.

        Returns:
            A dict with ``revision`` and a list of ``parquet_files`` relative
            paths for the selected config.

        Raises:
            RuntimeError: If no parquet files are found for the config.
        """
        try:
            response = requests.get(
                cls._dataset_api_url(),
                headers=cls._request_headers(),
                timeout=cls.download_timeout,
            )
            response.raise_for_status()
            metadata = response.json()
        except requests.RequestException as exc:
            rank_zero_warn(
                f"Requests failed to fetch Molecule3D metadata ({exc}). "
                "Falling back to curl."
            )
            metadata = cls._curl_get_json(cls._dataset_api_url())

        revision = metadata.get("sha", "main")
        all_files = [
            sibling["rfilename"]
            for sibling in metadata.get("siblings", [])
            if sibling["rfilename"].endswith(".parquet")
        ]

        return {
            "revision": revision,
            "all_parquet_files": all_files,
        }

    @classmethod
    def _filter_config_files(cls, all_files, config_name):
        """Filter parquet file paths belonging to a specific config.

        Args:
            all_files: List of all parquet file relative paths.
            config_name: The config name to filter for.

        Returns:
            Sorted list of parquet file paths for the config.
        """
        prefix = f"{config_name}/"
        return sorted(f for f in all_files if f.startswith(prefix))

    @classmethod
    def _is_valid_parquet_cache(cls, path):
        """Check if a cached parquet file is valid.

        Args:
            path: Path to the cached parquet file.

        Returns:
            True if the file has valid PAR1 magic bytes.
        """
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
        """Check if a manifest file and its referenced parquet files are intact.

        Args:
            manifest_path: Path to the manifest JSON file.

        Returns:
            True if all referenced parquet files exist and are valid.
        """
        if not osp.exists(manifest_path):
            return False

        try:
            with open(manifest_path, "r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, ValueError):
            return False

        config_dir = osp.dirname(manifest_path)
        parquet_files = manifest.get("parquet_files", [])
        if not parquet_files:
            return False

        for relative_path in parquet_files:
            local_path = osp.join(config_dir, relative_path)
            if not osp.exists(local_path) or osp.getsize(local_path) == 0:
                return False
            if not cls._is_valid_parquet_cache(local_path):
                return False

        return True

    @classmethod
    def _download_single_file(cls, url, path, description):
        """Download a single file with progress bar.

        Args:
            url: The URL to download from.
            path: Local path to save to.
            description: Description for the progress bar.

        Raises:
            RuntimeError: If the downloaded file is empty.
        """
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
        """Download Molecule3D parquet shards from Hugging Face."""
        os.makedirs(self.raw_dir, exist_ok=True)

        manifest_path = self._manifest_path()
        if self._manifest_complete(manifest_path):
            return

        repo_meta = self._fetch_repo_manifest()
        config_files = self._filter_config_files(
            repo_meta["all_parquet_files"], self.split_config
        )

        if not config_files:
            raise RuntimeError(
                f"No parquet files found for config '{self.split_config}' "
                f"in repo {self.hf_repo}."
            )

        config_dir = self._config_raw_dir()
        os.makedirs(config_dir, exist_ok=True)

        for relative_path in config_files:
            local_path = osp.join(config_dir, relative_path)
            os.makedirs(osp.dirname(local_path), exist_ok=True)

            if (
                osp.exists(local_path)
                and osp.getsize(local_path) > 0
                and self._is_valid_parquet_cache(local_path)
            ):
                continue

            if osp.exists(local_path):
                os.unlink(local_path)

            url = self._dataset_file_url(repo_meta["revision"], relative_path)
            try:
                self._download_single_file(
                    url=url,
                    path=local_path,
                    description=osp.basename(relative_path),
                )
            except Exception:
                if osp.exists(local_path):
                    os.unlink(local_path)
                raise

            if not self._is_valid_parquet_cache(local_path):
                if osp.exists(local_path):
                    os.unlink(local_path)
                raise RuntimeError(
                    f"Downloaded parquet shard is incomplete or corrupted: {local_path}"
                )

        manifest = {
            "revision": repo_meta["revision"],
            "parquet_files": config_files,
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    # ------------------------------------------------------------------
    #  Processing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_sdf_to_graph(sdf_string):
        """Parse an SDF (V2000/V3000) string into atomic numbers and positions.

        Args:
            sdf_string: A single-molecule SDF block.

        Returns:
            A tuple ``(atomic_numbers, positions)``, or ``(None, None)`` if
            parsing fails.
        """
        if Chem is None:
            raise ImportError(
                "Parsing SDF coordinates requires RDKit. "
                "Install it with `pip install rdkit`."
            )

        try:
            mol = Chem.MolFromMolBlock(sdf_string, removeHs=False, sanitize=True)
            if mol is None:
                return None, None

            conf = mol.GetConformer()
            positions = torch.tensor(conf.GetPositions(), dtype=torch.float32)
            atomic_numbers = torch.tensor(
                [atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long
            )
            return atomic_numbers, positions
        except Exception:
            return None, None

    @classmethod
    def _validate_parquet_columns(cls, parquet_path, parquet_file):
        """Validate that required columns exist in a parquet file.

        Args:
            parquet_path: Path to the parquet file (for error messages).
            parquet_file: A ``pyarrow.parquet.ParquetFile`` instance.

        Raises:
            RuntimeError: If required columns are missing.
        """
        if hasattr(parquet_file, "schema_arrow"):
            available_columns = set(parquet_file.schema_arrow.names)
        else:
            available_columns = set(parquet_file.schema.to_arrow_schema().names)

        missing_columns = [
            col for col in cls.required_columns if col not in available_columns
        ]
        if missing_columns:
            raise RuntimeError(
                f"Missing required columns in {parquet_path}: "
                f"{', '.join(missing_columns)}."
            )

    @classmethod
    def _split_name_from_filename(cls, filename):
        """Determine the split name from a parquet file name.

        Args:
            filename: The parquet file basename.

        Returns:
            One of ``"train"``, ``"validation"``, or ``"test"``.
        """
        basename = osp.basename(filename)
        if basename.startswith("train"):
            return "train"
        elif basename.startswith("validation"):
            return "validation"
        elif basename.startswith("test"):
            return "test"
        raise ValueError(f"Cannot determine split from filename: {filename}")

    def process(self):
        """Process Molecule3D parquet shards into PyTorch Geometric Data objects.

        Parses SDF columns via RDKit for 3D coordinates and extracts the
        target property. Samples are processed in split order (train,
        validation, test) so that split indices can be recorded.
        """
        if pq is None:
            raise ImportError(
                "Processing Molecule3D parquet data requires `pyarrow`. "
                "Install it with `pip install pyarrow`."
            )
        if Chem is None:
            raise ImportError(
                "Processing Molecule3D SDF coordinates requires RDKit. "
                "Install it with `pip install rdkit`."
            )

        manifest_path = self._manifest_path()
        if not self._manifest_complete(manifest_path):
            rank_zero_warn(
                "Molecule3D cache is incomplete or corrupted. "
                "Re-downloading raw parquet shards."
            )
            self.download()

        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        config_dir = self._config_raw_dir()

        # Group files by split to build split indices later.
        split_files = {"train": [], "validation": [], "test": []}
        for relative_path in manifest["parquet_files"]:
            split_name = self._split_name_from_filename(relative_path)
            if split_name in split_files:
                split_files[split_name].append(relative_path)

        samples = []
        split_boundaries = {}
        current_offset = 0

        for split_name in ("train", "validation", "test"):
            split_boundaries[split_name] = {"start": current_offset}
            files = split_files.get(split_name, [])
            if not files:
                split_boundaries[split_name]["end"] = current_offset
                continue

            for relative_path in files:
                parquet_path = osp.join(config_dir, relative_path)
                parquet_file = pq.ParquetFile(parquet_path)
                self._validate_parquet_columns(parquet_path, parquet_file)

                total_rows = parquet_file.metadata.num_rows
                row_iterator = parquet_file.iter_batches(
                    batch_size=self.process_batch_size,
                )

                with tqdm(
                    total=total_rows, desc=f"process {split_name}"
                ) as bar:
                    for batch in row_iterator:
                        batch_dict = batch.to_pydict()

                        for sdf_string, property_values in zip(
                            batch_dict["sdf"],
                            zip(
                                batch_dict.get("homo", [None] * len(batch_dict["sdf"])),
                                batch_dict.get("lumo", [None] * len(batch_dict["sdf"])),
                                batch_dict.get("Y", [None] * len(batch_dict["sdf"])),
                                batch_dict.get("scf energy", [None] * len(batch_dict["sdf"])),
                                batch_dict.get("dipole x", [None] * len(batch_dict["sdf"])),
                                batch_dict.get("dipole y", [None] * len(batch_dict["sdf"])),
                                batch_dict.get("dipole z", [None] * len(batch_dict["sdf"])),
                            ),
                        ):
                            z, pos = self._parse_sdf_to_graph(sdf_string)
                            if z is None or pos is None:
                                continue

                            y = torch.tensor(
                                [[float(v) if v is not None else 0.0 for v in property_values]],
                                dtype=torch.float32,
                            )

                            data = Data(z=z, pos=pos, y=y)

                            if self.pre_filter is not None and not self.pre_filter(data):
                                continue

                            if self.pre_transform is not None:
                                data = self.pre_transform(data)

                            samples.append(data)

                        bar.update(len(batch_dict["sdf"]))

            split_boundaries[split_name]["end"] = len(samples)
            current_offset = len(samples)

        data, slices = self.collate(samples)
        torch.save((data, slices), self.processed_paths[0])

        # Save split indices.
        split_indices = {
            "idx_train": list(range(split_boundaries["train"]["start"], split_boundaries["train"]["end"])),
            "idx_val": list(range(split_boundaries["validation"]["start"], split_boundaries["validation"]["end"])),
            "idx_test": list(range(split_boundaries["test"]["start"], split_boundaries["test"]["end"])),
        }
        with open(self.processed_paths[1], "w", encoding="utf-8") as handle:
            json.dump(split_indices, handle)
