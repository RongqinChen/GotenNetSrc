import os
import os.path as osp
import tarfile
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import numpy as np
import requests
import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch_geometric.data import Data, InMemoryDataset, extract_tar
from tqdm import tqdm


class rMD17(InMemoryDataset):
    revised_url = 'https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/23950376/rmd17.tar.bz2'
    download_chunk_size = 4 * 1024 * 1024
    parallel_download_threshold = 64 * 1024 * 1024
    download_timeout = (30, 600)

    molecule_files = dict(
        aspirin='rmd17_aspirin.npz',
        azobenzene='rmd17_azobenzene.npz',
        benzene='rmd17_benzene.npz',
        ethanol='rmd17_ethanol.npz',
        malonaldehyde='rmd17_malonaldehyde.npz',
        naphthalene='rmd17_naphthalene.npz',
        paracetamol='rmd17_paracetamol.npz',
        salicylic='rmd17_salicylic.npz',
        toluene='rmd17_toluene.npz',
        uracil='rmd17_uracil.npz',
    )

    available_molecules = list(molecule_files.keys())

    def __init__(self, root, transform=None, pre_transform=None, dataset_arg=None):
        assert dataset_arg is not None, (
            "Please provide the desired comma separated molecule(s) through"
            f"'dataset_arg'. Available molecules are {', '.join(rMD17.available_molecules)} "
            "or 'all' to train on the combined dataset."
        )

        if dataset_arg == "all":
            dataset_arg = ",".join(rMD17.available_molecules)
        self.molecules = dataset_arg.split(",")

        if len(self.molecules) > 1:
            rank_zero_warn(
                "MD17 molecules have different reference energies, "
                "which is not accounted for during training."
            )

        super(rMD17, self).__init__(root, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    def len(self):
        return self.data.y.size(0)

    @property
    def raw_file_names(self):
        return [osp.join('rmd17', 'npz_data', rMD17.molecule_files[mol]) for mol in self.molecules]

    def get_split(self, idx):
        assert idx in [0, 1, 2, 3, 4]

        sets = ['index_train', 'index_test']

        out = []
        for set_name in sets:
            split_path = osp.join(self.root, 'raw', 'rmd17', 'splits', set_name + f'_0{idx + 1}.csv')
            if not osp.exists(split_path):
                raise FileNotFoundError(f"File {split_path} not found")
            with open(split_path, 'r') as f:
                split = [int(line.strip()) for line in f.readlines()]
            out.append(split)
        return out

    @property
    def processed_file_names(self):
        return [f"rmd17-{mol}.pt" for mol in self.molecules]

    @staticmethod
    def _request_headers(byte_range=None):
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
        try:
            return max(1, min(32, int(os.environ.get('GOTENNET_RMD17_DOWNLOAD_WORKERS', '8'))))
        except ValueError:
            return 8

    @classmethod
    def _stream_response_to_file(cls, response, path, archive_name, total):
        with open(path, 'wb') as f, tqdm(
            desc=archive_name,
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
    def _parallel_download(cls, url, path, archive_name, total, workers):
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
            desc=archive_name,
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

    @classmethod
    def _single_stream_download(cls, url, path, archive_name):
        with requests.Session() as session, session.get(
            url,
            headers=cls._request_headers(),
            allow_redirects=True,
            stream=True,
            timeout=cls.download_timeout,
        ) as response:
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))
            cls._stream_response_to_file(response, path, archive_name, total)

    def download(self):
        """Download the rMD17 dataset from figshare.

        Uses a probe GET instead of HEAD because figshare-style redirect URLs
        can reject separate HEAD requests. When the resolved host advertises
        ranged reads, large archives are downloaded in parallel to better
        saturate available bandwidth; otherwise, it falls back to a single
        streamed request.
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        archive_name = 'rmd17.tar.bz2'
        path = osp.join(self.raw_dir, archive_name)

        try:
            with requests.Session() as session, session.get(
                self.revised_url,
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
                        self._parallel_download(resolved_url, path, archive_name, total, workers)
                    except Exception as exc:
                        if osp.exists(path):
                            os.unlink(path)
                        rank_zero_warn(
                            f"Parallel rMD17 download failed ({exc}). Falling back to a single-stream download."
                        )
                        self._single_stream_download(self.revised_url, path, archive_name)
                else:
                    self._stream_response_to_file(response, path, archive_name, total)
        except Exception:
            if osp.exists(path):
                os.unlink(path)
            raise

        if osp.getsize(path) == 0:
            os.unlink(path)
            raise RuntimeError(f"Downloaded file {path} is empty.")

        # ------------------------------------------------------------------
        # 3.  Extract the archive.  Try torch_geometric's extract_tar first;
        #     fall back to the stdlib tarfile if that fails (e.g. due to
        #     version-specific quirks with bz2 mode).
        # ------------------------------------------------------------------
        try:
            try:
                extract_tar(path, self.raw_dir, mode='r:bz2')
            except TypeError:
                # Older PyG versions don't accept 'mode' as a keyword.
                extract_tar(path, self.raw_dir)
        except Exception:
            # Fallback: use Python's tarfile directly.
            try:
                with tarfile.open(path, mode='r:bz2') as tf:
                    tf.extractall(self.raw_dir)
            except Exception:
                if osp.exists(path):
                    os.unlink(path)
                raise
        finally:
            if osp.exists(path):
                os.unlink(path)

    def process(self):

        for path, processed_path in zip(self.raw_paths, self.processed_paths):
            data_npz = np.load(path)
            z = torch.from_numpy(data_npz["nuclear_charges"]).long()
            positions = torch.from_numpy(data_npz["coords"]).float()
            energies = torch.from_numpy(data_npz["energies"]).float()
            forces = torch.from_numpy(data_npz["forces"]).float()
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
