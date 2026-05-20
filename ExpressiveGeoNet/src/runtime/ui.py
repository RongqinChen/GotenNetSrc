"""CLI-facing helpers for config printing, cleanup, and metric extraction."""

from __future__ import annotations

import warnings
from functools import wraps
from importlib.util import find_spec
from typing import Any, Callable, Sequence

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only

from src.common.logging import get_logger

log = get_logger(__name__)


def task_wrapper(task_func: Callable) -> Callable:
    """Wrap long-running tasks to ensure cleanup and consistent logging."""

    @wraps(task_func)
    def wrap(cfg: DictConfig):
        try:
            metric_dict, object_dict = task_func(cfg=cfg)
        except Exception as error:
            log.exception("Task execution failed.")
            raise error
        finally:
            log.info("Output dir: %s", cfg.paths.output_dir)

            if find_spec("wandb"):
                import wandb

                if wandb.run:
                    log.info("Closing wandb run.")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def extras(config: DictConfig) -> None:
    """Apply optional CLI helpers such as warning suppression and config printing."""
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings because <config.ignore_warnings=True>.")
        warnings.filterwarnings("ignore")

    if config.get("print_config"):
        log.info("Printing the resolved Hydra config tree.")
        print_config(config, resolve=True)


@rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "datamodule",
        "model",
        "callbacks",
        "logger",
        "trainer",
    ),
    resolve: bool = True,
) -> None:
    """Pretty-print the Hydra config tree using Rich."""
    import rich.syntax
    import rich.tree

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    ordered_fields = [field for field in print_order if field in config]
    for field in config:
        if field not in ordered_fields:
            ordered_fields.append(field)

    for field in ordered_fields:
        branch = tree.add(field, style=style, guide_style=style)
        config_group = config[field]
        branch_content = (
            OmegaConf.to_yaml(config_group, resolve=resolve)
            if isinstance(config_group, DictConfig)
            else str(config_group)
        )
        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)


def get_metric_value(
    metric_dict: dict[str, Any],
    metric_name: str | None,
) -> float | None:
    """Safely retrieve a named scalar metric from Lightning callback metrics."""
    if not metric_name:
        log.info("Metric name is empty; skipping metric retrieval.")
        return None

    if metric_name not in metric_dict:
        raise KeyError(
            f"Metric value not found for <metric_name={metric_name}>. "
            "Ensure the Lightning module logs this metric and the Hydra config references it correctly."
        )

    metric_value = metric_dict[metric_name].item()
    log.info("Retrieved metric value <%s=%s>.", metric_name, metric_value)
    return metric_value
