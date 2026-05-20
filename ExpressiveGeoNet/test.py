"""Hydra entrypoint for standalone ExpGeoNet evaluation runs."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.runtime.bootstrap import bootstrap_entrypoint
from src.runtime.runner import test
from src.runtime.ui import extras, get_metric_value

CONFIG_DIR = bootstrap_entrypoint(allow_tf32=False)


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
