"""Shared config, trainer, logger, and callback factories."""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, LightningDataModule, seed_everything
from pytorch_lightning.loggers import Logger

from src.common.logging import get_logger

log = get_logger(__name__)


def clone_config(cfg: DictConfig) -> DictConfig:
    """Create a mutable copy of a Hydra config for runtime adjustments."""
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def configure_matmul_precision(
    cfg: DictConfig, *, default: str, default_allow_tf32: bool = False
) -> None:
    """Apply matmul precision and optional TF32 acceleration from config."""
    precision = cfg.get("matmul_precision", default)
    allow_tf32 = bool(cfg.get("allow_tf32", default_allow_tf32))
    torch.set_float32_matmul_precision(precision)
    if hasattr(torch.backends, "cuda"):
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = allow_tf32
    log.info(
        "Running with %s precision (allow_tf32=%s).", precision, allow_tf32
    )


def seed_from_config(cfg: DictConfig) -> None:
    """Seed all supported RNGs when the config requests it."""
    if cfg.get("seed") is not None:
        seed_everything(cfg.seed, workers=True)


def resolve_label(datamodule: LightningDataModule, label: object) -> tuple[object, str]:
    """Resolve dataset labels while preserving the original string form for run names."""
    label_str = str(label)
    resolved_label = label

    if hasattr(datamodule, "dataset_class"):
        dataset_class = datamodule.dataset_class
        if isinstance(label, str) and hasattr(dataset_class, "label_to_idx"):
            resolved_label = dataset_class.label_to_idx(label)
            label_str = label
            log.info("Resolved label %s to index %s.", label, resolved_label)
        elif (
            isinstance(label, int)
            and hasattr(dataset_class, "available_properties")
            and 0 <= label < len(dataset_class.available_properties)
        ):
            label_str = dataset_class.available_properties[label]

    return resolved_label, label_str


def resolve_experiment_name(cfg: DictConfig) -> str:
    """Return the selected experiment config basename, or fall back to ``cfg.name``."""
    experiment_name = str(cfg.get("name", "default"))

    if HydraConfig.initialized():
        experiment_choice = HydraConfig.get().runtime.choices.get("experiment")
        if experiment_choice not in (None, "", "null", "None"):
            experiment_name = str(experiment_choice)

    return experiment_name


def populate_run_metadata(
    cfg: DictConfig, datamodule: LightningDataModule
) -> DictConfig:
    """Populate derived runtime metadata without mutating the caller config."""
    cfg.label, cfg.label_str = resolve_label(datamodule, cfg.label)
    cfg.experiment_name = resolve_experiment_name(cfg)
    cfg.name = f"{cfg.experiment_name}_{cfg.label_str}"

    if hasattr(datamodule, "label"):
        datamodule.label = cfg.label

    return cfg


def get_dataset_metadata(
    datamodule: LightningDataModule, label: object
) -> Optional[dict]:
    """Fetch dataset metadata when supported by the datamodule."""
    if hasattr(datamodule, "get_metadata"):
        return datamodule.get_metadata(label)
    return None


def resolve_resume_checkpoint(trainer_cfg: DictConfig) -> DictConfig:
    """Convert relative resume checkpoints to absolute paths."""
    resume_from_checkpoint = trainer_cfg.get("resume_from_checkpoint")
    if resume_from_checkpoint and not os.path.isabs(resume_from_checkpoint):
        trainer_cfg.resume_from_checkpoint = os.path.join(
            hydra.utils.get_original_cwd(), resume_from_checkpoint
        )
    return trainer_cfg


def apply_mps_fallback(trainer_cfg: DictConfig) -> DictConfig:
    """Force CPU execution when MPS is available but radius_graph would fail."""
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    accelerator = trainer_cfg.get("accelerator", "auto")

    if (
        mps_available
        and not torch.cuda.is_available()
        and accelerator in {"auto", "gpu", "mps"}
    ):
        log.warning(
            "MPS is available, but torch_cluster.radius_graph requires CPU or CUDA. "
            "Falling back to CPU execution."
        )
        trainer_cfg.accelerator = "cpu"
        trainer_cfg.devices = 1
        trainer_cfg.strategy = "auto"

    return trainer_cfg


def build_trainer_config(cfg: DictConfig, *, apply_mps_guard: bool) -> DictConfig:
    """Create a fully resolved trainer config ready for instantiation."""
    trainer_cfg = OmegaConf.create(OmegaConf.to_container(cfg.trainer, resolve=True))
    trainer_cfg = resolve_resume_checkpoint(trainer_cfg)
    if apply_mps_guard:
        trainer_cfg = apply_mps_fallback(trainer_cfg)
    return trainer_cfg


def build_callbacks(
    cfg: DictConfig,
    *,
    disabled_names: Sequence[str] = (),
    allowed_names: Optional[Iterable[str]] = None,
) -> list[Callback]:
    """Instantiate callbacks from config with optional filtering."""
    callbacks: list[Callback] = []
    allowed_name_set = set(allowed_names) if allowed_names is not None else None
    disabled_name_set = set(disabled_names)

    for name, callback_cfg in cfg.get("callbacks", {}).items():
        if allowed_name_set is not None and name not in allowed_name_set:
            continue
        if name in disabled_name_set or "_target_" not in callback_cfg:
            continue

        log.info("Instantiating callback <%s>.", callback_cfg._target_)
        callbacks.append(hydra.utils.instantiate(callback_cfg))

    return callbacks


def build_loggers(cfg: DictConfig, *, enabled: bool) -> list[Logger]:
    """Instantiate configured loggers when logging is enabled."""
    loggers: list[Logger] = []
    if not enabled:
        return loggers

    for logger_cfg in cfg.get("logger", {}).values():
        if "_target_" not in logger_cfg:
            continue

        log.info("Instantiating logger <%s>.", logger_cfg._target_)
        loggers.append(hydra.utils.instantiate(logger_cfg))

    return loggers
