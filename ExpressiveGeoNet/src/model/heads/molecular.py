from typing import Optional

import ase
import torch
import torch.nn.functional as F
import torch_scatter
from torch import nn
from torch_geometric.utils import scatter

from src.model.heads.atomwise import Atomwise
from src.model.heads.blocks import GatedEquivariantBlock
from src.model.layers import shifted_softplus


class Dipole(nn.Module):
    """Output layer for dipole moment."""

    def __init__(
        self,
        n_in: int,
        n_hidden: Optional[int] = None,
        activation=F.silu,
        property: str = "dipole",
        predict_magnitude: bool = False,
        output_v: bool = True,
        mean: Optional[torch.Tensor] = None,
        stddev: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the Dipole module.

        Args:
            n_in (int): Input dimension of atomwise features.
            n_hidden (Optional[int]): Size of hidden layers.
            activation: Activation function.
            property (str): Name of property to be predicted.
            predict_magnitude (bool): If true, calculate magnitude of dipole.
            output_v (bool): If true, output vector representation.
            mean (Optional[torch.Tensor]): Mean of the property for standardization.
            stddev (Optional[torch.Tensor]): Standard deviation for standardization.
        """
        super().__init__()

        self.stddev = stddev
        self.mean = mean
        self.output_v = output_v
        if n_hidden is None:
            n_hidden = n_in

        self.property = property
        self.derivative = None
        self.predict_magnitude = predict_magnitude

        self.equivariant_layers = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    n_sin=n_in,
                    n_vin=n_in,
                    n_sout=n_hidden,
                    n_vout=n_hidden,
                    n_hidden=n_hidden,
                    activation=activation,
                    sactivation=activation,
                ),
                GatedEquivariantBlock(
                    n_sin=n_hidden,
                    n_vin=n_hidden,
                    n_sout=1,
                    n_vout=1,
                    n_hidden=n_hidden,
                    activation=activation,
                ),
            ]
        )
        self.requires_dr = False
        self.requires_stress = False
        self.aggregation_mode = "sum"

    def forward(self, inputs):
        """
        Predicts dipole moment.

        Args:
            inputs: Input data containing atomic representations.

        Returns:
            dict: Dictionary with predicted dipole properties.
        """
        positions = inputs.pos
        l0 = inputs.representation
        l1 = inputs.vector_representation[:, :3, :]

        for eqlayer in self.equivariant_layers:
            l0, l1 = eqlayer(l0, l1)

        if self.stddev is not None:
            l0 = self.stddev * l0 + self.mean

        atomic_dipoles = torch.squeeze(l1, -1)
        charges = l0
        dipole_offsets = positions * charges

        y = atomic_dipoles + dipole_offsets
        # y = torch.sum(y, dim=1)
        y = torch_scatter.scatter(y, inputs.batch, dim=0, reduce=self.aggregation_mode)
        if self.output_v:
            y_vector = torch_scatter.scatter(
                l1, inputs.batch, dim=0, reduce=self.aggregation_mode
            )

        if self.predict_magnitude:
            y = torch.norm(y, dim=1, keepdim=True)

        result = {self.property: y}
        if self.output_v:
            result[self.property + "_vector"] = y_vector
        return result


class ElectronicSpatialExtentV2(Atomwise):
    """Electronic spatial extent prediction module."""

    def __init__(
        self,
        n_in: int,
        n_layers: int = 2,
        n_hidden: Optional[int] = None,
        activation=shifted_softplus,
        property: str = "y",
        contributions: Optional[str] = None,
        mean: Optional[torch.Tensor] = None,
        stddev: Optional[torch.Tensor] = None,
        outnet: Optional[nn.Module] = None,
    ):
        """
        Initialize the ElectronicSpatialExtentV2 module.

        Args:
            n_in (int): Input dimension of atomwise features.
            n_layers (int): Number of layers in the output network.
            n_hidden (Optional[int]): Size of hidden layers.
            activation: Activation function.
            property (str): Name of the target property.
            contributions (Optional[str]): Name of the atomic contributions.
            mean (Optional[torch.Tensor]): Mean of the property for standardization.
            stddev (Optional[torch.Tensor]): Standard deviation for standardization.
            outnet (Optional[nn.Module]): Network for property prediction.
        """
        super(ElectronicSpatialExtentV2, self).__init__(
            n_in,
            1,
            "sum",
            n_layers,
            n_hidden,
            activation=activation,
            mean=mean,
            stddev=stddev,
            outnet=outnet,
            property=property,
            contributions=contributions,
        )
        atomic_mass = torch.from_numpy(ase.data.atomic_masses).float()
        self.register_buffer("atomic_mass", atomic_mass)

    def forward(self, inputs):
        """
        Predicts the electronic spatial extent.

        Args:
            inputs: Input data containing atomic representations and positions.

        Returns:
            dict: Dictionary with predicted electronic spatial extent properties.
        """
        positions = inputs.pos
        x = self.out_net(inputs)
        mass = self.atomic_mass[inputs.z].view(-1, 1)
        c = scatter(mass * positions, inputs.batch, dim=0) / scatter(
            mass, inputs.batch, dim=0
        )

        yi = torch.norm(positions - c[inputs.batch], dim=1, keepdim=True)
        yi = yi**2 * x

        y = torch_scatter.scatter(yi, inputs.batch, dim=0, reduce=self.aggregation_mode)

        # collect results
        result = {self.property: y}

        if self.contributions:
            result[self.contributions] = x

        return result
