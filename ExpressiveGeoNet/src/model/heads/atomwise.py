from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch_scatter
from torch import nn
from torch.autograd import grad

from src.common.logging import get_logger
from src.model.layers import (
    GetItem,
    ScaleShift,
    SchnetMLP,
    shifted_softplus,
    str2act,
)

log = get_logger(__name__)


class AtomwiseV3(nn.Module):
    """
    Atomwise prediction module V3 for predicting atomic properties.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        aggregation_mode: Optional[str] = "sum",
        n_layers: int = 2,
        n_hidden: Optional[int] = None,
        activation=shifted_softplus,
        property: str = "y",
        contributions: Optional[str] = None,
        derivative: Optional[str] = None,
        negative_dr: bool = True,
        create_graph: bool = True,
        mean: Optional[Union[float, torch.Tensor]] = None,
        stddev: Optional[Union[float, torch.Tensor]] = None,
        atomref: Optional[torch.Tensor] = None,
        outnet: Optional[nn.Module] = None,
        return_vector: Optional[str] = None,
        standardize: bool = True,
    ):
        """
        Initialize the AtomwiseV3 module.

        Args:
            n_in (int): Input dimension of atomwise features.
            n_out (int): Output dimension of target property.
            aggregation_mode (Optional[str]): Aggregation method for atomic contributions.
            n_layers (int): Number of layers in the output network.
            n_hidden (Optional[int]): Size of hidden layers.
            activation: Activation function.
            property (str): Name of the target property.
            contributions (Optional[str]): Name of the atomic contributions.
            derivative (Optional[str]): Name of the property derivative.
            negative_dr (bool): If True, negative derivative of the energy.
            create_graph (bool): If True, create computational graph for derivatives.
            mean (Optional[Union[float, torch.Tensor]]): Mean of the property for standardization.
            stddev (Optional[Union[float, torch.Tensor]]): Standard deviation for standardization.
            atomref (Optional[torch.Tensor]): Reference single-atom properties.
            outnet (Optional[nn.Module]): Network for property prediction.
            return_vector (Optional[str]): Name of the vector property to return.
            standardize (bool): If True, standardize the output property.
        """
        super(AtomwiseV3, self).__init__()

        self.return_vector = return_vector
        self.n_layers = n_layers
        self.create_graph = create_graph
        self.property = property
        self.contributions = contributions
        self.derivative = derivative
        self.negative_dr = negative_dr
        self.standardize = standardize

        mean = 0.0 if mean is None else mean
        stddev = 1.0 if stddev is None else stddev
        self.mean = mean
        self.stddev = stddev

        if type(activation) is str:
            activation = str2act(activation)

        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(atomref.type(torch.float32))
        else:
            self.atomref = None

        if outnet is None:
            self.out_net = nn.Sequential(
                GetItem("representation"),
                SchnetMLP(n_in, n_out, n_hidden, n_layers, activation),
            )
        else:
            self.out_net = outnet

        # build standardization layer
        if self.standardize and (mean is not None and stddev is not None):
            self.standardize = ScaleShift(mean, stddev)
        else:
            self.standardize = nn.Identity()

        self.aggregation_mode = aggregation_mode

    def _derivative_graph_flags(self) -> tuple[bool, bool]:
        """Return ``(create_graph, retain_graph)`` for derivative predictions.

        Training force losses need a higher-order graph so gradients can flow
        through predicted forces back to model parameters. During validation,
        test, and inference we only need first-order force values, so we avoid
        building or retaining that extra graph.
        """
        create_graph = self.create_graph and self.training and torch.is_grad_enabled()
        retain_graph = create_graph
        return create_graph, retain_graph

    def forward(self, inputs):
        """
        Predicts atomwise property.

        Args:
            inputs: Input data containing atomic representations.

        Returns:
            dict: Dictionary with predicted properties.
        """
        atomic_numbers = inputs.z
        result = {}
        yi = self.out_net(inputs)
        yi = yi * self.stddev

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0

        if self.aggregation_mode is not None:
            y = torch_scatter.scatter(
                yi, inputs.batch, dim=0, reduce=self.aggregation_mode
            )
        else:
            y = yi

        y = y + self.mean

        # collect results
        result[self.property] = y

        if self.contributions:
            result[self.contributions] = yi
        if self.derivative:
            sign = -1.0 if self.negative_dr else 1.0
            create_graph, retain_graph = self._derivative_graph_flags()
            dy = grad(
                outputs=result[self.property],
                inputs=[inputs.pos],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=create_graph,
                retain_graph=retain_graph,
            )[0]

            dy = sign * dy
            result[self.derivative] = dy
        return result


class Atomwise(nn.Module):
    """
    Atomwise prediction module for predicting atomic properties.
    """

    def __init__(
        self,
        n_in: int,
        n_out: int = 1,
        aggregation_mode: Optional[str] = "sum",
        n_layers: int = 2,
        n_hidden: Optional[int] = None,
        activation=shifted_softplus,
        property: str = "y",
        contributions: Optional[str] = None,
        derivative: Optional[str] = None,
        negative_dr: bool = True,
        create_graph: bool = True,
        mean: Optional[torch.Tensor] = None,
        stddev: Optional[torch.Tensor] = None,
        atomref: Optional[torch.Tensor] = None,
        outnet: Optional[nn.Module] = None,
        return_vector: Optional[str] = None,
        standardize: bool = True,
    ):
        """
        Initialize the Atomwise module.

        Args:
            n_in (int): Input dimension of atomwise features.
            n_out (int): Output dimension of target property.
            aggregation_mode (Optional[str]): Aggregation method for atomic contributions.
            n_layers (int): Number of layers in the output network.
            n_hidden (Optional[int]): Size of hidden layers.
            activation: Activation function.
            property (str): Name of the target property.
            contributions (Optional[str]): Name of the atomic contributions.
            derivative (Optional[str]): Name of the property derivative.
            negative_dr (bool): If True, negative derivative of the energy.
            create_graph (bool): If True, create computational graph for derivatives.
            mean (Optional[torch.Tensor]): Mean of the property for standardization.
            stddev (Optional[torch.Tensor]): Standard deviation for standardization.
            atomref (Optional[torch.Tensor]): Reference single-atom properties.
            outnet (Optional[nn.Module]): Network for property prediction.
            return_vector (Optional[str]): Name of the vector property to return.
            standardize (bool): If True, standardize the output property.
        """
        super(Atomwise, self).__init__()

        self.return_vector = return_vector
        self.n_layers = n_layers
        self.create_graph = create_graph
        self.property = property
        self.contributions = contributions
        self.derivative = derivative
        self.negative_dr = negative_dr
        self.standardize = standardize

        if mean is None:
            mean = torch.tensor([0.0], dtype=torch.float32)
        elif isinstance(mean, float):
            mean = torch.tensor([mean], dtype=torch.float32)
        else:
            mean = mean.detach().clone().float()

        if stddev is None:
            stddev = torch.tensor([1.0], dtype=torch.float32)
        elif isinstance(stddev, float):
            stddev = torch.tensor([stddev], dtype=torch.float32)
        else:
            stddev = stddev.detach().clone().float()

        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)

        if type(activation) is str:
            activation = str2act(activation)

        # initialize single atom energies
        if atomref is not None:
            self.atomref = nn.Embedding.from_pretrained(atomref.type(torch.float32))
        else:
            self.atomref = None

        self.equivariant = False
        # build output network
        if outnet is None:
            self.out_net = nn.Sequential(
                GetItem("representation"),
                SchnetMLP(n_in, n_out, n_hidden, n_layers, activation),
            )
        else:
            self.out_net = outnet

        if self.standardize:
            log.info(
                "Using graph-level standardization with mean %s and stddev %s",
                self.mean,
                self.stddev,
            )

        self.aggregation_mode = aggregation_mode

    def _derivative_graph_flags(self) -> tuple[bool, bool]:
        """Return ``(create_graph, retain_graph)`` for derivative predictions.

        Keep higher-order graphs only during training, where force losses need
        to backpropagate through predicted derivatives.
        """
        create_graph = self.create_graph and self.training and torch.is_grad_enabled()
        retain_graph = create_graph
        return create_graph, retain_graph

    def forward(self, inputs):
        """
        Predicts atomwise property.

        Args:
            inputs: Input data containing atomic representations.

        Returns:
            dict: Dictionary with predicted properties.
        """
        atomic_numbers = inputs.z
        result = {}

        if self.equivariant:
            l0 = inputs.representation
            l1 = inputs.vector_representation
            for eqlayer in self.out_net:
                l0, l1 = eqlayer(l0, l1)

            if self.return_vector:
                result[self.return_vector] = l1
            yi = l0
        else:
            yi = self.out_net(inputs)

        if self.standardize:
            yi = yi * self.stddev

        if self.atomref is not None:
            y0 = self.atomref(atomic_numbers)
            yi = yi + y0

        if self.aggregation_mode is not None:
            y = torch_scatter.scatter(
                yi, inputs.batch, dim=0, reduce=self.aggregation_mode
            )
            if self.standardize:
                # Dataset statistics are graph-level, so only shift once after
                # aggregating atomic contributions into a graph prediction.
                y = y + self.mean
        else:
            y = yi + self.mean if self.standardize else yi

        # collect results
        result[self.property] = y

        if self.contributions:
            result[self.contributions] = yi

        if self.derivative:
            sign = -1.0 if self.negative_dr else 1.0
            create_graph, retain_graph = self._derivative_graph_flags()
            dy = grad(
                outputs=result[self.property],
                inputs=[inputs.pos],
                grad_outputs=torch.ones_like(result[self.property]),
                create_graph=create_graph,
                retain_graph=retain_graph,
            )[0]

            result[self.derivative] = sign * dy
        return result
