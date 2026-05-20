"""Training and evaluation orchestration for the script-first project layout."""

from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

from src.common.logging import get_logger, log_hyperparameters
from src.model.system import GotenModel
from src.runtime.factories import (
    build_callbacks,
    build_loggers,
    build_trainer_config,
    clone_config,
    configure_matmul_precision,
    get_dataset_metadata,
    populate_run_metadata,
    seed_from_config,
)
from src.runtime.ui import task_wrapper

log = get_logger(__name__)


def _supports_inference_mode(model: LightningModule) -> bool:
    """Return whether Lightning's inference mode is safe for this model."""
    return not getattr(model, "requires_force_derivatives", False)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute model training and optional best-checkpoint evaluation."""
    runtime_cfg = clone_config(cfg)
    configure_matmul_precision(runtime_cfg, default="highest")
    seed_from_config(runtime_cfg)

    log.info("Instantiating datamodule <%s>.", runtime_cfg.datamodule._target_)
    datamodule: LightningDataModule = hydra.utils.instantiate(runtime_cfg.datamodule)

    runtime_cfg = populate_run_metadata(runtime_cfg, datamodule)
    dataset_meta = get_dataset_metadata(datamodule, runtime_cfg.label_str)

    log.info("Instantiating model <%s>.", runtime_cfg.model._target_)
    model: LightningModule = hydra.utils.instantiate(
        runtime_cfg.model,
        dataset_meta=dataset_meta,
    )

    callbacks = build_callbacks(
        runtime_cfg,
        disabled_names=("learning_rate_monitor",) if runtime_cfg.exp else (),
    )
    loggers = build_loggers(runtime_cfg, enabled=not runtime_cfg.exp)

    trainer_cfg = build_trainer_config(runtime_cfg, apply_mps_guard=True)
    log.info("Instantiating trainer <%s>.", runtime_cfg.trainer._target_)
    trainer: Trainer = hydra.utils.instantiate(
        trainer_cfg,
        callbacks=callbacks,
        logger=loggers,
        _convert_="partial",
        inference_mode=_supports_inference_mode(model),
    )

    datamodule.device = model.device
    object_dict = {
        "cfg": runtime_cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }

    log.info("Logging hyperparameters.")
    log_hyperparameters(config=runtime_cfg, model=model, trainer=trainer)

    if runtime_cfg.get("train"):
        log.info("Starting training.")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=runtime_cfg.get("ckpt_path"),
        )

    optimized_metric = runtime_cfg.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise KeyError(
            "Metric for hyperparameter optimization was not logged. "
            "Verify the `optimized_metric` setting in the Hydra config."
        )

    train_metrics = trainer.callback_metrics

    if runtime_cfg.get("test"):
        ckpt_path = None
        if runtime_cfg.get("train") and not runtime_cfg.trainer.get("fast_dev_run"):
            ckpt_path = trainer.checkpoint_callback.best_model_path
        elif runtime_cfg.get("ckpt_path"):
            ckpt_path = runtime_cfg.ckpt_path

        log.info("Starting testing.")
        trainer.test(
            model=model, datamodule=datamodule, ckpt_path=ckpt_path, weights_only=False
        )

    if not runtime_cfg.trainer.get("fast_dev_run") and runtime_cfg.get("train"):
        log.info(
            "Best model checkpoint: %s", trainer.checkpoint_callback.best_model_path
        )

    metric_dict = {**train_metrics, **trainer.callback_metrics}
    return metric_dict, object_dict


@task_wrapper
def test(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate a checkpoint or freshly instantiated model on the configured test set."""
    runtime_cfg = clone_config(cfg)
    configure_matmul_precision(runtime_cfg, default="high")
    seed_from_config(runtime_cfg)

    model: LightningModule | None = None
    if runtime_cfg.get("checkpoint"):
        model = GotenModel.from_pretrained(runtime_cfg.checkpoint)
        if runtime_cfg.get("label", -1) == -1 and model.label is not None:
            runtime_cfg.label = model.label

    log.info("Instantiating datamodule <%s>.", runtime_cfg.datamodule._target_)
    datamodule: LightningDataModule = hydra.utils.instantiate(runtime_cfg.datamodule)

    runtime_cfg = populate_run_metadata(runtime_cfg, datamodule)
    dataset_meta = get_dataset_metadata(datamodule, runtime_cfg.label_str)

    if model is None:
        log.info("Instantiating model <%s>.", runtime_cfg.model._target_)
        model = hydra.utils.instantiate(runtime_cfg.model, dataset_meta=dataset_meta)

    callbacks = build_callbacks(
        runtime_cfg,
        allowed_names=("model_summary", "rich_progress_bar"),
    )
    loggers = build_loggers(runtime_cfg, enabled=True)

    trainer_cfg = build_trainer_config(runtime_cfg, apply_mps_guard=False)
    log.info("Instantiating trainer <%s>.", runtime_cfg.trainer._target_)
    trainer: Trainer = hydra.utils.instantiate(
        trainer_cfg,
        logger=loggers,
        callbacks=callbacks,
        inference_mode=_supports_inference_mode(model),
    )

    if trainer.logger:
        trainer.logger.log_hyperparams({"ckpt_path": runtime_cfg.ckpt_path})

    log.info("Starting testing.")
    trainer.test(
        model=model,
        datamodule=datamodule,
        ckpt_path=runtime_cfg.get("ckpt_path"),
        weights_only=False,
    )

    object_dict = {
        "cfg": runtime_cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }
    return dict(trainer.callback_metrics), object_dict
