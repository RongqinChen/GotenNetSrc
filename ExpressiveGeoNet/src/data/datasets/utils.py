"""Helpers shared by dataset wrappers."""

from __future__ import annotations

import json
import os
import os.path as osp
from collections.abc import Callable, Sequence
from typing import TypeVar

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Shared download configuration
# ---------------------------------------------------------------------------
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
DOWNLOAD_TIMEOUT = (30, 600)
DOWNLOAD_CHUNK = 512 * 1024

T = TypeVar("T")


def parse_csv_selection(
    value: str | None,
    *,
    available: Sequence[T],
    normalize: Callable[[str], T] | None = None,
    all_token: str = "all",
    field_name: str = "label",
) -> list[T]:
    """Parse a comma-separated selection string into validated unique values."""
    if value is None:
        raise ValueError(
            f"Pass {field_name!r}; available values: {', '.join(map(str, available))}"
        )

    raw_value = value.strip()
    if not raw_value:
        raise ValueError(f"{field_name!r} cannot be empty.")

    if raw_value.lower() == all_token:
        return list(available)

    selected: list[T] = []
    for item in raw_value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        selected.append(normalize(stripped) if normalize is not None else stripped)

    if not selected:
        raise ValueError(f"{field_name!r} did not contain any values.")

    unknown = [item for item in selected if item not in available]
    if unknown:
        raise ValueError(
            f"Unknown {field_name}(s): {unknown}. Available values: {', '.join(map(str, available))}"
        )
    return list(dict.fromkeys(selected))


def stream_download(
    url: str,
    path: str,
    *,
    description: str,
    headers: dict[str, str] | None = None,
    timeout: tuple[int, int] | int = DOWNLOAD_TIMEOUT,
    chunk_size: int = DOWNLOAD_CHUNK,
) -> None:
    """Download a file with a progress bar; raises on empty payloads."""
    os.makedirs(osp.dirname(path), exist_ok=True)
    response = requests.get(
        url,
        headers=headers or {"User-Agent": USER_AGENT},
        allow_redirects=True,
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with open(path, "wb") as handle, tqdm(
        desc=description,
        total=total,
        unit="B",
        unit_scale=True,
    ) as progress:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            handle.write(chunk)
            progress.update(len(chunk))

    if osp.getsize(path) == 0:
        os.unlink(path)
        raise RuntimeError(f"Downloaded file is empty: {path}")


def is_manifest_complete(manifest_path: str) -> bool:
    """Check if all files referenced in a JSON manifest exist and are non-empty."""
    if not osp.exists(manifest_path):
        return False
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return False

    parent = osp.dirname(manifest_path)
    return all(
        osp.exists(osp.join(parent, rp)) and osp.getsize(osp.join(parent, rp)) > 0
        for rp in manifest.get("parquet_files", [])
    )
