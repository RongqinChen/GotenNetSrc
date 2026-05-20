"""Base class for all tasks in the project."""

from __future__ import absolute_import, division, print_function

import torch

from src.common.logging import get_logger

log = get_logger(__name__)


class Task:
    """Base interface for tasks — subclasses define losses, metrics, and output heads."""

    name = None

    def __init__(
        self,
        representation,
        label_key,
        dataset_meta,
        task_config=None,
        task_defaults=None,
        **kwargs,
    ):
        if task_config is None:
            task_config = {}
        if task_defaults is None:
            task_defaults = {}

        self.task_config = task_config
        self.config = {**task_defaults, **task_config}
        log.info(f"Task config: {self.config}")
        self.representation = representation
        self.label_key = label_key
        self.dataset_meta = dataset_meta
        # Keep optimization losses in the model's native dtype by default, but
        # allow opting into fp64 for experiments that truly need it.
        self.cast_to_float64 = self.config.get("cast_to_float64", False)
        # Metrics are cheap enough to normalize to fp32 for more consistent
        # logging, especially if a run later enables reduced-precision compute.
        self.cast_metrics_to_float32 = self.config.get(
            "cast_metrics_to_float32", True
        )

    def _select_outputs(self, batch, result, metric_meta, metric_idx):
        """Select and reshape predictions + targets for supervised comparisons."""
        pred = result[metric_meta["prediction"]]
        targets = batch[metric_meta["target"]]
        pred = pred.reshape(targets.shape)
        return pred, targets

    def process_loss_outputs(self, batch, result, metric_meta, metric_idx):
        """Prepare supervised tensors for loss computation."""
        pred, targets = self._select_outputs(batch, result, metric_meta, metric_idx)

        if self.cast_to_float64:
            targets = targets.type(torch.float64)
            pred = pred.type(torch.float64)

        return pred, targets

    def process_metric_outputs(self, batch, result, metric_meta, metric_idx):
        """Prepare supervised tensors for metric computation."""
        pred, targets = self._select_outputs(batch, result, metric_meta, metric_idx)

        if self.cast_to_float64:
            targets = targets.type(torch.float64)
            pred = pred.type(torch.float64)
        elif self.cast_metrics_to_float32:
            targets = targets.float()
            pred = pred.float()

        return pred, targets

    def process_outputs(self, batch, result, metric_meta, metric_idx):
        """Backward-compatible alias for metric preprocessing."""
        return self.process_metric_outputs(batch, result, metric_meta, metric_idx)

    def get_metric_names(self, metric_meta, metric_idx=0):
        return f"{metric_meta['prediction']}"

    def get_losses(self):
        raise NotImplementedError("get_losses() is not implemented")

    def get_metrics(self):
        raise NotImplementedError("get_metrics() is not implemented")

    def get_output(self, output_config=None):
        raise NotImplementedError("get_output() is not implemented")

    def get_evaluator(self):
        return None

    def get_dataloader_map(self):
        return ["test"]
