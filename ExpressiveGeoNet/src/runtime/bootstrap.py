"""Runtime bootstrap helpers shared by standalone entry scripts."""

from __future__ import annotations

import os
from typing import Final

import dotenv
import torch
from omegaconf import OmegaConf

from src.common.project import find_config_directory

_PATCH_FLAG: Final[str] = "_ExpGeoNet_patched"


def configure_torch_load_compatibility() -> None:
    """Force legacy checkpoint loads to default to `weights_only=False`."""
    os.environ["TORCH_LOAD_WEIGHTS_ONLY"] = "0"

    if getattr(torch.load, _PATCH_FLAG, False):
        return

    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    setattr(patched_torch_load, _PATCH_FLAG, True)
    torch.load = patched_torch_load


def load_environment() -> None:
    """Load environment variables from `.env` if one is present."""
    dotenv.load_dotenv(override=True)


def configure_tf32(allow_tf32: bool = True) -> None:
    """Enable or disable TF32 matmul acceleration when CUDA is available."""
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def register_config_resolvers() -> None:
    """Register OmegaConf resolvers used by Hydra config files."""
    if OmegaConf.has_resolver("non_null"):
        return

    OmegaConf.register_new_resolver(
        "non_null",
        lambda value, fallback: (
            fallback if value in (None, "", "null", "None") else value
        ),
    )


def bootstrap_entrypoint(*, allow_tf32: bool = True) -> str:
    """Prepare runtime state for a Hydra entrypoint and return config path."""
    configure_torch_load_compatibility()
    load_environment()
    configure_tf32(allow_tf32=allow_tf32)
    register_config_resolvers()
    return find_config_directory()
