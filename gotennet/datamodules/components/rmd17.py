import os
import os.path as osp
import tarfile

import numpy as np
import requests
import torch
from pytorch_lightning.utilities import rank_zero_warn
from torch_geometric.data import Data, InMemoryDataset, extract_tar
from tqdm import tqdm


class rMD17(InMemoryDataset):
    revised_url = 'https://ndownloader.figshare.com/files/23950376'

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

    def download(self):
        """Download the rMD17 dataset from figshare.

        Uses requests instead of urllib because figshare's ndownloader URL
        redirects to a short-lived S3 presigned URL. A separate HEAD request
        to resolve that redirect often fails with 403, so the archive must be
        streamed with a single GET request.
        """
        os.makedirs(self.raw_dir, exist_ok=True)
        archive_name = 'rmd17.tar.bz2'
        path = osp.join(self.raw_dir, archive_name)

        headers = {
            'User-Agent': (
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
        }
        session = requests.Session()
        try:
            with session.get(
                self.revised_url,
                headers=headers,
                allow_redirects=True,
                stream=True,
                timeout=(30, 600),   # (connect, read) timeouts
            ) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get('content-length', 0))
                with open(path, 'wb') as f, tqdm(
                    desc=archive_name,
                    total=total,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:   # skip keep-alive empty chunks
                            f.write(chunk)
                            bar.update(len(chunk))
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
