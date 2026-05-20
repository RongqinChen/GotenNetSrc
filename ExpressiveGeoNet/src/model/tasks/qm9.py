"""QM9 task — quantum chemistry property prediction for 133k small molecules."""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
import torchmetrics
from torch.nn import L1Loss

from src.data.datasets.qm9 import QM9
from src.model.heads import Atomwise, Dipole, ElectronicSpatialExtentV2
from src.model.tasks.base import Task


class QM9Task(Task):
    """Task for the QM9 dataset (12 quantum-chemical properties)."""

    name = "QM9"

    def __init__(self, representation, label_key, dataset_meta, task_config=None, **kwargs):
        super().__init__(representation, label_key, dataset_meta, task_config, **kwargs)

        if isinstance(label_key, str):
            self.label_key = QM9.available_properties.index(label_key)
        self.num_classes = 1
        self.task_loss = self.task_config.get("task_loss", "L1Loss")

    def _select_outputs(self, batch, result, metric_meta, metric_idx):
        """Extract predictions and targets, handling multi-column ``y``."""
        pred = result[metric_meta["prediction"]]
        if batch.y.shape[1] == 1:
            targets = batch.y
        else:
            targets = batch.y[:, metric_meta["target"]]
        pred = pred.reshape(targets.shape)
        return pred, targets

    def get_metric_names(self, metric_meta, metric_idx=0):
        if metric_meta["prediction"] == "property":
            return QM9.available_properties[metric_meta["target"]]
        return super().get_metric_names(metric_meta, metric_idx)

    def get_losses(self):
        LossClass = L1Loss if self.task_loss == "L1Loss" else torch.nn.MSELoss
        return [{"metric": LossClass, "prediction": "property", "target": self.label_key, "loss_weight": 1.0}]

    def get_metrics(self):
        return [
            {"metric": torchmetrics.MeanSquaredError, "prediction": "property", "target": self.label_key},
            {"metric": torchmetrics.MeanAbsoluteError, "prediction": "property", "target": self.label_key},
        ]

    def get_output(self, output_config=None) -> torch.nn.ModuleList:
        label_name = QM9.available_properties[self.label_key]
        output_config = output_config or {}

        if label_name == "mu":
            outputs = Dipole(
                n_in=self.representation.hidden_dim,
                predict_magnitude=True,
                property="property",
                mean=self.dataset_meta.get("mean"),
                stddev=self.dataset_meta.get("std"),
                **output_config,
            )
        elif label_name == "r2":
            outputs = ElectronicSpatialExtentV2(
                n_in=self.representation.hidden_dim,
                property="property",
                **output_config,
            )
        else:
            outputs = Atomwise(
                n_in=self.representation.hidden_dim,
                mean=self.dataset_meta.get("mean"),
                stddev=self.dataset_meta.get("std"),
                atomref=self.dataset_meta.get("atomref"),
                property="property",
                activation=F.silu,
                **output_config,
            )
        return torch.nn.ModuleList([outputs])

    def get_evaluator(self):
        return None

    def get_dataloader_map(self):
        return ["test"]
