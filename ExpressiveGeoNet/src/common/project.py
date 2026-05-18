"""Generic project helpers that are not tied to training or evaluation runtime."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.common.logging import get_logger

log = get_logger(__name__)


def humanbytes(size_in_bytes: float) -> str:
    """Render a byte count using human-readable binary units."""
    size = float(size_in_bytes)
    units = ("Bytes", "KB", "MB", "GB", "TB")

    if size < 1024:
        singular = "Byte" if size == 1 else "Bytes"
        return f"{size:.0f} {singular}"

    for unit in units[1:]:
        size /= 1024.0
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"

    return f"{size:.2f} TB"


def find_config_directory() -> str:
    """Locate the Hydra config directory for the standalone project."""
    common_dir = Path(__file__).resolve().parent
    current_dir = Path.cwd().resolve()

    candidates = [
        (current_dir / "configs", current_dir),
        (common_dir.parent.parent / "configs", common_dir.parent.parent),
    ]

    for config_dir, project_root in candidates:
        if config_dir.is_dir():
            os.environ["PROJECT_ROOT"] = str(project_root)
            return str(config_dir)

    searched_paths = "\n".join(f"  - {path}" for path, _ in candidates)
    raise FileNotFoundError(
        "Could not find the Hydra config directory in any of the following locations:\n"
        f"{searched_paths}\n\n"
        f"Current working directory: {current_dir}\n"
        f"Common module location: {common_dir}"
    )


def get_function_name(func: Any) -> str:
    """Return a stable display name for a metric or loss callable."""
    return func.name if hasattr(func, "name") else type(func).__name__.split(".")[-1]
