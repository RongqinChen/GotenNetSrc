"""Molecule3D task — 7 DFT properties for ~3.9M molecules."""

from __future__ import absolute_import, division, print_function

import torch
import torch.nn.functional as F
import torchmetrics
from torch.nn import L1Loss

from src.data.datasets.molecule3d import Molecule3D
from src.model.heads import Atomwise
from src.model.tasks.base import Task


class Molecule3DTask(Task):
    """Task for the Molecule3D dataset (HOMO, LUMO, gap, SCF energy, dipole components)."""

    name = "Molecule3D"

    def __init__(self, representation, label_key, dataset_meta, task_config=None, **kwargs):
        super().__init__(representation, label_key, dataset_meta, task_config, **kwargs)

        if isinstance(label_key, str):
            self.label_key = Molecule3D.available_properties.index(label_key)
        self.num_classes = 1
        self.task_loss = self.task_config.get("task_loss", "L1Loss")

    def _select_outputs(self, batch, result, metric_meta, metric_idx):
        pred = result[metric_meta["prediction"]]
        if batch.y.shape[1] == 1:
            targets = batch.y
        else:
            targets = batch.y[:, metric_meta["target"]]
        pred = pred.reshape(targets.shape)
        return pred, targets

    def get_metric_names(self, metric_meta, metric_idx=0):
        if metric_meta["prediction"] == "property":
            return Molecule3D.available_properties[metric_meta["target"]]
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
        output_config = output_config or {}
        return torch.nn.ModuleList([
            Atomwise(
                n_in=self.representation.hidden_dim,
                mean=self.dataset_meta.get("mean"),
                stddev=self.dataset_meta.get("std"),
                atomref=self.dataset_meta.get("atomref"),
                property="property",
                activation=F.silu,
                **output_config,
            )
        ])
