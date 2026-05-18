"""Checkpoint helpers for standalone representation loading."""

from __future__ import annotations

from pathlib import Path
from typing import Type

import torch


def load_representation_from_checkpoint(
    cls: Type[torch.nn.Module],
    checkpoint_path: str,
    *,
    device: str = "cpu",
):
    """Load a representation-only module from a Lightning checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} does not exist.")

    checkpoint = torch.load(path, map_location=device)
    if "representation" in checkpoint:
        checkpoint = checkpoint["representation"]

    if "hyper_parameters" not in checkpoint:
        raise KeyError("Checkpoint must contain 'hyper_parameters'.")

    hyper_parameters = checkpoint["hyper_parameters"]
    if "representation" not in hyper_parameters:
        raise KeyError("Checkpoint hyperparameters must contain 'representation'.")

    representation_config = dict(hyper_parameters["representation"])
    representation_config.pop("_target_", None)
    representation_config.pop("__target__", None)

    if "state_dict" not in checkpoint:
        raise KeyError("Checkpoint must contain 'state_dict'.")

    original_state_dict = checkpoint["state_dict"]
    new_state_dict: dict[str, torch.Tensor] = {}
    for key, value in original_state_dict.items():
        if key.startswith("output_modules."):
            continue
        if key.startswith("representation."):
            new_state_dict[key.replace("representation.", "")] = value
        else:
            new_state_dict[key] = value

    representation = cls(**representation_config)
    representation.load_state_dict(new_state_dict, strict=True)
    return representation
