import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
import requests
import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


class MD22(InMemoryDataset):
    """MD22 molecular dynamics dataset.

    The MD22 dataset contains molecular dynamics trajectories for seven
    large molecules, providing energies and forces computed at the
    PBE+MBD level of theory.

    Data source: http://www.sgdpml.org/#datasets

    Available molecules:
        AT-AT, AT-AT-CG-CG, Ac-Ala3-NHMe, DHA, buckycatcher,
        double-walled nanotube, stachyose
    """

    base_url = 'http://www.quantum-machine.org/gdml/data/npz'
    download_chunk_size = 4 * 1024 * 1024
    parallel_download_threshold = 64 * 1024 * 1024
    download_timeout = (30, 600)

    molecule_files = dict(
        at_at='md22_AT-AT.npz',
        at_at_cg_cg='md22_AT-AT-CG-CG.npz',
        ac_ala3_nhme='md22_Ac-Ala3-NHMe.npz',
        dha='md22_DHA.npz',
        buckycatcher='md22_buckycatcher.npz',
        double_walled_nanotube='md22_double-walled_nanotube.npz',
        stachyose='md22_stachyose.npz',
    )

    available_molecules = list(molecule_files.keys())

    def __init__(self, root, transform=None, pre_transform=None, dataset_arg=None):
        """Initialize the MD22 dataset.

        Args:
            root (str): Root directory where the dataset should be stored.
            transform: Transform applied to each data object at access time.
            pre_transform: Transform applied to each data object before saving.
            dataset_arg (str): Comma-separated list of molecule names, or
                'all' to use every available molecule.

        Raises:
            AssertionError: If dataset_arg is None.
        """
        assert dataset_arg is not None, (
            "Please provide the desired comma separated molecule(s) through"
            f"'dataset_arg'. Available molecules are {', '.join(MD22.available_molecules)} "
            "or 'all' to train on the combined dataset."
        )

        if dataset_arg == "all":
            dataset_arg = ",".join(MD22.available_molecules)
        self.molecules = dataset_arg.split(",")

        if len(self.molecules) > 1:
            rank_zero_warn(
                "MD22 molecules have different reference energies, "
                "which is not accounted for during training."
            )

        super(MD22, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def len(self):
        return self.data.y.size(0)

    @property
    def raw_file_names(self):
        return [osp.join('md22', MD22.molecule_files[mol]) for mol in self.molecules]

    @property
    def processed_file_names(self):
        return [f"md22-{mol}.pt" for mol in self.molecules]

    @staticmethod
    def _request_headers(byte_range=None):
        """Build HTTP request headers with a realistic User-Agent."""
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
        }
        if byte_range is not None:
            headers['Range'] = byte_range
        return headers

    @staticmethod
    def _download_workers():
        """Return the number of parallel download workers.

        The value can be overridden via the environment variable
        ``GOTENNET_MD22_DOWNLOAD_WORKERS`` (default: 4).
        """
        try:
            return max(1, min(32, int(os.environ.get('GOTENNET_MD22_DOWNLOAD_WORKERS', '4'))))
        except ValueError:
            return 4

    @classmethod
    def _download_single_file(cls, url, path, molecule_name):
        """Download a single molecule file with a progress bar.

        Args:
            url (str): Full URL to the file.
            path (str): Local file path to write to.
            molecule_name (str): Human-readable molecule name for the progress bar.
        """
        with requests.Session() as session, session.get(
            url,
            headers=cls._request_headers(),
            allow_redirects=True,
            stream=True,
            timeout=cls.download_timeout,
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))

            with open(path, 'wb') as f, tqdm(
                desc=molecule_name,
                total=total,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=cls.download_chunk_size):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

    @classmethod
    def _download_range(cls, url, path, start, end, bar, bar_lock):
        """Download a byte range of a file, updating a shared progress bar.

        Args:
            url (str): Full URL to the file.
            path (str): Local file path to write to.
            start (int): Start byte offset (inclusive).
            end (int): End byte offset (inclusive).
            bar (tqdm): Shared progress bar instance.
            bar_lock (threading.Lock): Lock guarding the progress bar.
        """
        with requests.Session() as session, session.get(
            url,
            headers=cls._request_headers(f'bytes={start}-{end}'),
            stream=True,
            timeout=cls.download_timeout,
        ) as response:
            response.raise_for_status()
            if response.status_code != 206:
                raise RuntimeError(
                    f"Expected HTTP 206 for range {start}-{end}, got {response.status_code}."
                )
            content_range = response.headers.get('content-range', '')
            if not content_range.startswith(f'bytes {start}-'):
                raise RuntimeError(
                    f"Unexpected Content-Range header for bytes {start}-{end}: {content_range!r}."
                )

            with open(path, 'r+b') as f:
                f.seek(start)
                downloaded = 0
                for chunk in response.iter_content(chunk_size=cls.download_chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        with bar_lock:
                            bar.update(len(chunk))

            expected = end - start + 1
            if downloaded != expected:
                raise RuntimeError(
                    f"Range {start}-{end} downloaded {downloaded} bytes, expected {expected}."
                )

    @classmethod
    def _parallel_download_file(cls, url, path, molecule_name, total, workers):
        """Download a file using parallel ranged requests.

        Args:
            url (str): Full URL to the file.
            path (str): Local file path to write to.
            molecule_name (str): Human-readable molecule name for the progress bar.
            total (int): Total file size in bytes.
            workers (int): Number of parallel download workers.
        """
        part_size = (total + workers - 1) // workers
        ranges = []
        for worker_idx in range(workers):
            start = worker_idx * part_size
            end = min(total - 1, start + part_size - 1)
            if start <= end:
                ranges.append((start, end))

        with open(path, 'wb') as f:
            f.truncate(total)

        bar_lock = Lock()
        with tqdm(
            desc=molecule_name,
            total=total,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
                futures = [
                    executor.submit(cls._download_range, url, path, start, end, bar, bar_lock)
                    for start, end in ranges
                ]
                for future in futures:
                    future.result()

    def download(self):
        """Download the MD22 dataset from the sGDML website.

        Each molecule is downloaded as a separate .npz file.  Large files
        are downloaded in parallel when the server supports ranged reads;
        otherwise a single-stream request is used.
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        md22_raw_dir = osp.join(self.raw_dir, 'md22')
        os.makedirs(md22_raw_dir, exist_ok=True)

        for mol in self.molecules:
            filename = MD22.molecule_files[mol]
            url = f"{MD22.base_url}/{filename}"
            path = osp.join(md22_raw_dir, filename)

            if osp.exists(path):
                continue

            try:
                with requests.Session() as session, session.get(
                    url,
                    headers=self._request_headers(),
                    allow_redirects=True,
                    stream=True,
                    timeout=self.download_timeout,
                ) as response:
                    response.raise_for_status()
                    total = int(response.headers.get('content-length', 0))
                    resolved_url = response.url
                    supports_ranges = response.headers.get('accept-ranges', '').lower() == 'bytes'
                    workers = self._download_workers()
                    should_parallelize = (
                        supports_ranges
                        and total >= self.parallel_download_threshold
                        and workers > 1
                    )

                    if should_parallelize:
                        response.close()
                        try:
                            self._parallel_download_file(resolved_url, path, mol, total, workers)
                        except Exception as exc:
                            if osp.exists(path):
                                os.unlink(path)
                            rank_zero_warn(
                                f"Parallel MD22 download for {mol} failed ({exc}). "
                                "Falling back to a single-stream download."
                            )
                            self._download_single_file(url, path, mol)
                    else:
                        self._download_single_file(resolved_url, path, mol)
            except Exception:
                if osp.exists(path):
                    os.unlink(path)
                raise

            if osp.getsize(path) == 0:
                os.unlink(path)
                raise RuntimeError(f"Downloaded file {path} is empty.")

    def process(self):
        """Process raw MD22 .npz files into PyTorch Geometric Data objects.

        Each molecule's .npz file contains:
            - R  (n_samples, n_atoms, 3): Cartesian coordinates.
            - z  (n_atoms,):             Nuclear charges.
            - E  (n_samples,):           Total energies.
            - F  (n_samples, n_atoms, 3): Atomic forces.
        """
        for path, processed_path in zip(self.raw_paths, self.processed_paths):
            data_npz = np.load(path)
            z = torch.from_numpy(data_npz["z"]).long()
            positions = torch.from_numpy(data_npz["R"]).float()
            energies = torch.from_numpy(data_npz["E"]).float()
            forces = torch.from_numpy(data_npz["F"]).float()
            energies.unsqueeze_(1)

            samples = []

            for pos, y, dy in tqdm(zip(positions, energies, forces), total=energies.size(0)):
                data = Data(z=z, pos=pos, y=y.unsqueeze(1), dy=dy)

                if self.pre_filter is not None:
                    data = self.pre_filter(data)

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                samples.append(data)

            data, slices = self.collate(samples)
            torch.save((data, slices), processed_path)
