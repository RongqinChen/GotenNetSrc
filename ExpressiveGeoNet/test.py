"""Hydra entrypoint for standalone ExpressiveGeoNet evaluation runs."""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from runtime.bootstrap import bootstrap_entrypoint
from runtime.runner import test
from runtime.ui import extras, get_metric_value

CONFIG_DIR = bootstrap_entrypoint(allow_tf32=True)


@hydra.main(version_base="1.3", config_path=CONFIG_DIR, config_name="test.yaml")
def main(cfg: DictConfig) -> float | None:
    """Run evaluation and return the configured optimization metric when present."""
    extras(cfg)
    metric_dict, _ = test(cfg)
    return get_metric_value(
        metric_dict=metric_dict,
        metric_name=cfg.get("optimized_metric"),
    )


if __name__ == "__main__":
    main()
