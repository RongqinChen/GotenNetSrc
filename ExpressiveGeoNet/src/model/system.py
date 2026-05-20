"""GotenModel — LightningModule combining representation, task head, and optimizers."""

from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as opt
from omegaconf import DictConfig

from src.common.logging import get_logger
from src.common.project import get_function_name
from src.model.tasks import TASK_DICT

log = get_logger(__name__)

# ───────────────────────────────────────────────────────────────────────
# Hydra helper
# ───────────────────────────────────────────────────────────────────────


def _hydra_instantiate(config):
    """Recursively convert ``__target__`` → ``_target_`` and instantiate via Hydra."""
    if isinstance(config, (dict, DictConfig)):
        for key, value in config.items():
            if key == "__target__":
                log.info(f"Lazy instantiation of {value} with hydra.utils.instantiate")
                config["_target_"] = config.pop("__target__")
            elif isinstance(value, (dict, DictConfig)):
                _hydra_instantiate(value)
    return config


# ───────────────────────────────────────────────────────────────────────
# Main model
# ───────────────────────────────────────────────────────────────────────

VALID_PHASES = ("train", "validation", "test")


def _phase_metric_map(instance, phase: str):
    """Return ``(meta, metric_modules)`` tuple for the given phase."""
    registry = {
        "train": (instance.train_meta, instance.train_metrics),
        "validation": (instance.val_meta, instance.val_metrics),
        "test": (instance.test_meta, instance.test_metrics),
    }
    if phase not in registry:
        raise NotImplementedError(
            f"Unknown phase {phase!r}. Expected one of {VALID_PHASES}"
        )
    return registry[phase]


class GotenModel(pl.LightningModule):
    """Atomistic model for molecular property prediction.

    Combines a representation encoder with task-specific output modules to
    predict energies, forces, dipole moments, or other quantum-chemical properties.
    """

    # ── Initialisation ────────────────────────────────────────────────

    def __init__(  # noqa: PLR0913, PLR0915
        self,
        label: int,
        representation: nn.Module,
        task: str = "QM9",
        lr: float = 5e-4,
        lr_decay: float = 0.5,
        lr_patience: int = 100,
        lr_minlr: float = 1e-6,
        lr_monitor: str = "validation/val_loss",
        weight_decay: float = 0.01,
        cutoff: float = 5.0,
        dataset_meta: Optional[dict] = None,
        output: Optional[dict] = None,
        scheduler: Optional[dict] = None,  # passed as kwargs to CosineAnnealingLR
        save_predictions: Optional[bool] = None,
        task_config: Optional[dict] = None,
        lr_warmup_steps: int = 0,
        use_ema: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        dataset_meta = self._sanitize_dataset_meta(dataset_meta)

        # ── Store hyper-parameters for checkpointing ──────────────────
        # Logger-side hparams are emitted separately via src.common.logging,
        # so disable Lightning's automatic hparams logging here to avoid
        # serializing runtime tensors such as dataset_meta.atomref into
        # TensorBoard/CSV hparams.yaml files.
        self.save_hyperparameters(logger=False)

        # ── Training hyper-parameters ─────────────────────────────────
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_patience = lr_patience
        self.lr_minlr = lr_minlr
        self.lr_monitor = lr_monitor
        self.weight_decay = weight_decay
        self.lr_warmup_steps = lr_warmup_steps

        # ── General configuration ─────────────────────────────────────
        self.task = task
        self.label = label
        self.cutoff = cutoff
        self.use_ema = use_ema
        self.save_predictions = save_predictions
        self.scheduler = scheduler  # dict forwarded to CosineAnnealingLR

        # Store lightweight dataset metadata used by task heads.
        self.dataset_meta = dataset_meta

        # ── Representation encoder ────────────────────────────────────
        self.representation = self._maybe_hydra_instantiate(representation)

        # ── Task handler ──────────────────────────────────────────────
        # The task handler provides losses, metrics, output heads, and optionally an evaluator.
        if self.task in TASK_DICT:
            self.task_handler = TASK_DICT[self.task](
                representation=self.representation,
                label_key=label,
                dataset_meta=dataset_meta,
                task_config=task_config,
            )
        else:
            self.task_handler = None

        self.evaluator = (
            self.task_handler.get_evaluator() if self.task_handler else None
        )

        # ── Metrics ───────────────────────────────────────────────────
        # Separate metric modules for train/validation/test (train is empty by default).
        self.train_meta: list = []
        self.train_metrics = nn.ModuleList()

        self.val_meta = self.get_metrics()
        self.val_metrics = nn.ModuleList([meta["metric"]() for meta in self.val_meta])

        self.test_meta = self.get_metrics()
        self.test_metrics = nn.ModuleList([meta["metric"]() for meta in self.test_meta])

        # ── Output heads ──────────────────────────────────────────────
        self.output_modules = self.get_output(output or {})

        # ── Loss functions ────────────────────────────────────────────
        self.loss_meta = self._prepare_loss_meta(self.get_losses())
        self.loss_modules = nn.ModuleList([loss["metric"]() for loss in self.loss_meta])

        # ── EMA tracking ──────────────────────────────────────────────
        # Initialise EMA state for each loss across all phases.
        self.ema: dict[str, Optional[torch.Tensor]] = {}
        for loss_cfg in self.loss_meta:
            for phase in VALID_PHASES:
                self.ema[f"{phase}_{loss_cfg['target']}"] = None

        # ── Derivative requirements ───────────────────────────────────
        # Flag: does any output head require force derivatives?
        self.requires_force_derivatives = any(
            output_module.derivative for output_module in self.output_modules
        )

    # ── Class methods ─────────────────────────────────────────────────

    @classmethod
    def from_pretrained(cls, checkpoint_url: str) -> "GotenModel":
        """Load a pretrained model from a remote checkpoint URL."""
        from src.runtime.checkpoints import download_checkpoint

        checkpoint_path = download_checkpoint(checkpoint_url)
        return cls.load_from_checkpoint(checkpoint_path)

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _maybe_hydra_instantiate(representation: nn.Module) -> nn.Module:
        """If *representation* is a Hydra ``DictConfig``, instantiate it."""
        if isinstance(representation, DictConfig) and (
            "__target__" in representation or "_target_" in representation
        ):
            import hydra

            _hydra_instantiate(representation)
            return hydra.utils.instantiate(representation)
        return representation

    @staticmethod
    def _prepare_loss_meta(loss_configs: list) -> list:
        """Fill default ``ema_stages`` for any loss that defines an ``ema_rate``."""
        for cfg in loss_configs:
            if "ema_rate" in cfg and "ema_stages" not in cfg:
                cfg["ema_stages"] = ["train", "validation"]
        return loss_configs

    @staticmethod
    def _sanitize_dataset_meta(dataset_meta: Optional[dict]) -> Optional[dict]:
        """Strip heavy runtime-only objects before hparams are serialized."""
        if dataset_meta is None:
            return None

        sanitized = dict(dataset_meta)
        sanitized.pop("dataset", None)
        return sanitized

    @staticmethod
    def _get_num_graphs(batch) -> int:
        """Extract the number of graphs from a PyG batch (or tuple thereof)."""
        if isinstance(batch, (list, tuple)):
            batch = batch[0]
        return batch.num_graphs

    # ── Task delegation ───────────────────────────────────────────────

    def get_losses(self) -> list:
        """Return loss configurations from the task handler."""
        if self.task_handler is not None:
            return self.task_handler.get_losses()
        raise NotImplementedError(
            "No task handler configured — cannot retrieve losses."
        )

    def get_metrics(self) -> list:
        """Return metric configurations from the task handler."""
        if self.task_handler is not None:
            return self.task_handler.get_metrics()
        raise NotImplementedError(f"Task not implemented: {self.task}")

    def get_output(self, output_config: Optional[dict] = None) -> list:
        """Return output head modules from the task handler."""
        if self.task_handler is not None:
            return self.task_handler.get_output(output_config)
        raise NotImplementedError(f"Task not implemented: {self.task}")

    # ── Forward passes ─────────────────────────────────────────────────

    def _encode(self, batch) -> Any:
        """Run the representation encoder on *batch* and attach results."""
        batch.representation, batch.vector_representation = self.representation(batch)
        return batch

    def _enable_grads_on_positions(self, batch) -> None:
        """Enable gradient tracking on atomic positions if force derivatives are needed."""
        if self.requires_force_derivatives:
            batch.pos.requires_grad_()

    def _apply_output_heads(self, batch) -> dict:
        """Compute all output head predictions for *batch*."""
        result = {}
        for output_module in self.output_modules:
            result.update(output_module(batch))
        return result

    def _forward_pipeline(self, batch) -> dict:
        """Run the full forward pass for the current autograd context."""
        self._enable_grads_on_positions(batch)
        self._encode(batch)
        return self._apply_output_heads(batch)

    def _needs_grad_for_phase(self, phase: str) -> bool:
        """Return whether *phase* requires autograd-enabled execution."""
        return phase == "train" or self.requires_force_derivatives

    def _run_phase(self, batch, phase: str) -> dict:
        """Run a forward pass using the appropriate grad mode for *phase*."""
        with torch.set_grad_enabled(self._needs_grad_for_phase(phase)):
            return self._forward_pipeline(batch)

    def forward(self, batch) -> dict:
        """Full forward pass that respects the caller's current grad mode."""
        return self._forward_pipeline(batch)

    def encode(self, batch) -> Any:
        """Encode *batch* only (no output heads), respecting caller grad mode."""
        self._enable_grads_on_positions(batch)
        return self._encode(batch)

    # ── Training / validation / test steps ─────────────────────────────

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:  # type: ignore[override]
        """Single training step: forward pass → loss → return scalar loss."""
        result = self._run_phase(batch, "train")
        return self.calculate_loss(batch, result, name="train")

    def validation_step(  # type: ignore[override]
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict:
        """Single validation step: forward pass → loss → metrics → evaluation outputs."""
        result = self._run_phase(batch, "validation")

        # Loss
        val_loss = self.calculate_loss(batch, result, name="validation").detach().item()
        self.log_metrics(batch, result, "validation")

        logged: dict = {"val_loss": val_loss}
        self.log(
            "validation/val_loss",
            val_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self._get_num_graphs(batch),
        )

        # Evaluation outputs (e.g., for external evaluators)
        if self.evaluator is not None:
            eval_keys = self.task_handler.get_evaluation_keys()
            logged["outputs"] = {
                "y_pred": result[eval_keys["pred"]].detach().cpu(),
                "y_true": batch[eval_keys["target"]].detach().cpu(),
            }
        return logged

    def test_step(  # type: ignore[override]
        self,
        batch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict:
        """Single test step: forward pass → loss → metrics → evaluation outputs."""
        result = self._run_phase(batch, "test")

        self.calculate_loss(batch, result).detach().item()
        self.log_metrics(batch, result, "test")

        # Collect per-output predictions for analysis
        logged = {
            loss_cfg["prediction"]: result[loss_cfg["prediction"]].cpu()
            for loss_cfg in self.loss_meta
        }
        if self.evaluator is not None:
            eval_keys = self.task_handler.get_evaluation_keys()
            logged["outputs"] = {
                "y_pred": result[eval_keys["pred"]].detach().cpu(),
                "y_true": batch[eval_keys["target"]].detach().cpu(),
            }
        return logged

    # ── Metrics ───────────────────────────────────────────────────────

    def log_metrics(self, batch, result: dict, mode: str) -> None:
        """Compute and log metrics for the given phase (*mode*)."""
        meta_list, metric_modules = _phase_metric_map(self, mode)

        for idx, (meta, metric_fn) in enumerate(zip(meta_list, metric_modules)):
            # Compute the metric value
            if "target" in meta:
                # Supervised metric: compare predictions to ground-truth targets
                predictions, targets = self.task_handler.process_metric_outputs(
                    batch,
                    result,
                    meta,
                    idx,
                )
                if meta["prediction"] == "force":
                    predictions = predictions[:, :]
                metric_value = metric_fn(predictions, targets).detach().item()
            else:
                # Unsupervised / scalar metric (e.g., uncertainty estimate)
                metric_value = metric_fn(result[meta["prediction"]]).detach().item()

            # Log with a human-readable name
            metric_name = get_function_name(metric_fn)
            variable_name = (
                self.task_handler.get_metric_names(meta, idx)
                if self.task_handler is not None
                else ""
            )
            self.log(
                f"{mode}/{metric_name}_{variable_name}",
                metric_value,
                on_step=False,
                on_epoch=True,
                batch_size=self._get_num_graphs(batch),
            )

    # ── Loss ──────────────────────────────────────────────────────────

    @staticmethod
    def _ema_total_log_key(phase_name: str) -> str:
        """Return the aggregate EMA log key for a given phase."""
        if phase_name == "validation":
            return "validation/ema_val_loss"
        return f"{phase_name}/ema_loss"

    def calculate_loss(
        self,
        batch,
        result: dict,
        name: Optional[str] = None,
    ) -> torch.Tensor:
        """Compute the total raw loss and optionally log detached EMA diagnostics.

        Parameters
        ----------
        batch : Data
            Input batch (provides labels).
        result : dict
            Output of ``_apply_output_heads``.
        name : str or None
            Phase name (``"train"``, ``"validation"``, or ``None`` for no logging).

        Returns
        -------
        torch.Tensor
            Raw aggregated loss (scalar) used for optimization.
        """
        device = self.device
        dtype = self.dtype

        total_loss = torch.tensor(0.0, device=device, dtype=dtype)
        ema_total = (
            torch.tensor(0.0, device=device, dtype=dtype)
            if self.use_ema and name is not None
            else None
        )

        for loss_idx, loss_cfg in enumerate(self.loss_meta):
            loss_fn = self.loss_modules[loss_idx]

            # 1. Compute individual loss
            if "target" in loss_cfg:
                predictions, targets = self.task_handler.process_loss_outputs(
                    batch,
                    result,
                    loss_cfg,
                    loss_idx,
                )
                raw_loss = loss_fn(predictions, targets)
            else:
                raw_loss = loss_fn(result[loss_cfg["prediction"]])

            # 2. Accumulate the raw loss for optimization.
            total_loss = total_loss + loss_cfg["loss_weight"] * raw_loss

            # 3. Update detached EMA diagnostics strictly for logging.
            ema_loss = None
            if ema_total is not None:
                ema_loss = self._update_ema_and_get_smoothed(
                    loss_cfg,
                    raw_loss,
                    name,
                )
                ema_total = ema_total + loss_cfg["loss_weight"] * ema_loss

            # 4. Log the individual loss component(s).
            if name is not None:
                log_key = f"{name}/{loss_cfg['prediction']}_loss"
                self.log(
                    log_key,
                    raw_loss,
                    on_step=(name == "train"),
                    on_epoch=True,
                    prog_bar=(name == "train"),
                    batch_size=self._get_num_graphs(batch),
                )
                if ema_loss is not None:
                    self.log(
                        f"{name}/{loss_cfg['prediction']}_ema_loss",
                        ema_loss,
                        on_step=(name == "train"),
                        on_epoch=True,
                        prog_bar=False,
                        batch_size=self._get_num_graphs(batch),
                    )

        if ema_total is not None:
            self.log(
                self._ema_total_log_key(name),
                ema_total,
                on_step=(name == "train"),
                on_epoch=True,
                batch_size=self._get_num_graphs(batch),
            )

        return total_loss

    def _update_ema_and_get_smoothed(
        self,
        loss_cfg: dict,
        current_loss: torch.Tensor,
        phase_name: Optional[str],
    ) -> torch.Tensor:
        """Update and return detached EMA-smoothed diagnostics for *current_loss*.

        EMA is used only for logging/monitoring and never changes the raw
        optimization objective returned by :meth:`calculate_loss`.
        """
        ema_rate = loss_cfg.get("ema_rate")
        ema_stages = loss_cfg.get("ema_stages", [])
        if (
            not self.use_ema
            or ema_rate is None
            or phase_name is None
            or phase_name not in ema_stages
            or not (0.0 < ema_rate < 1.0)
        ):
            return current_loss

        ema_key = f"{phase_name}_{loss_cfg['target']}"
        smoothed = self.ema[ema_key]

        if smoothed is None:
            self.ema[ema_key] = current_loss.detach()
        else:
            smoothed = ema_rate * current_loss + (1.0 - ema_rate) * smoothed
            self.ema[ema_key] = smoothed.detach()
            current_loss = smoothed

        return current_loss

    # ── Optimizer ─────────────────────────────────────────────────────

    def configure_optimizers(self) -> tuple:
        """Set up AdamW optimizer and learning-rate scheduler."""
        optimizer = opt.AdamW(
            self.trainer.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            eps=1e-7,
        )

        if self.scheduler:
            # Custom scheduler configuration (typically CosineAnnealingLR params)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                **self.scheduler,
            )
        else:
            # Default: ReduceLROnPlateau with stored hyper-parameters
            scheduler = opt.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.lr_decay,
                patience=self.lr_patience,
                min_lr=self.lr_minlr,
            )

        return [optimizer], [
            {
                "scheduler": scheduler,
                "monitor": self.lr_monitor,
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        ]

    def optimizer_step(self, *args, **kwargs) -> None:  # type: ignore[override]
        """Override optimisation step to support linear LR warmup.

        During the first ``lr_warmup_steps`` iterations the learning rate is
        linearly scaled from 0 up to the configured ``lr``.
        """
        # Extract the optimizer from positional or keyword arguments
        optimizer = kwargs.get("optimizer")
        if optimizer is None and len(args) > 2:
            optimizer = args[2]

        # Linear warmup: scale LR up over the warmup period
        if self.trainer.global_step < self.lr_warmup_steps:
            progress = float(self.trainer.global_step + 1) / float(
                max(1, self.lr_warmup_steps)
            )
            scale = min(1.0, progress)
            for param_group in optimizer.param_groups:
                param_group["lr"] = scale * self.lr

        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()
