import math

import torch
from torch import nn as nn
from torch import Tensor

from src.model.layers.activations import normalize_string
from src.model.layers.cutoff import CosineCutoff


def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    """
    Compute Gaussian radial basis functions.

    Args:
        inputs (torch.Tensor): Input distances. Shape: [..., 1]
        offsets (torch.Tensor): Centers of the Gaussian functions. Shape: [n_rbf]
        widths (torch.Tensor): Widths of the Gaussian functions. Shape: [n_rbf]

    Returns:
        torch.Tensor: Gaussian RBF values. Shape: [..., n_rbf]
    """
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianRBF(nn.Module):
    """
    Gaussian radial basis functions module.

    Expands distances using Gaussian functions centered at different offsets.

    Args:
        n_rbf (int): Total number of Gaussian functions.
        cutoff (float): Center of the last Gaussian function (maximum distance).
        start (float, optional): Center of the first Gaussian function. Defaults to 0.0.
        trainable (bool, optional): If True, widths and offsets are learnable parameters. Defaults to False.
    """

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)


class BesselBasis(nn.Module):
    """
    Sine for radial basis expansion with coulomb decay (0th order Bessel functions).

    Used in DimeNet. Reference: https://arxiv.org/abs/2003.03123

    Args:
        cutoff (float, optional): Radial cutoff distance. Defaults to 5.0.
        n_rbf (int, optional): Number of basis functions. Defaults to None.
        trainable (bool, optional): Kept for compatibility, but parameters are not learnable. Defaults to False.
    """

    def __init__(self, cutoff=5.0, n_rbf=None, trainable=False):
        super(BesselBasis, self).__init__()
        if n_rbf is None:
            raise ValueError("n_rbf must be specified for BesselBasis")
        self.n_rbf = n_rbf
        # compute offset and width of Gaussian functions
        freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
        self.register_buffer("freqs", freqs)
        self.register_buffer("norm1", torch.tensor(1.0))

    def forward(self, inputs):
        a = self.freqs[None, :]
        inputs = inputs[..., None]
        ax = inputs * a
        sinax = torch.sin(ax)

        norm = torch.where(inputs == 0, self.norm1, inputs)
        y = sinax / norm

        return y


class ExpNormalSmearing(nn.Module):
    """
    Exponential Normal Smearing for radial basis functions.

    Uses exponentially spaced means and Gaussian functions for smearing distances.

    Args:
        cutoff (float, optional): Cutoff distance. Defaults to 5.0.
        n_rbf (int, optional): Number of radial basis functions. Defaults to 50.
        trainable (bool, optional): If True, means and betas are learnable parameters. Defaults to False.
    """

    def __init__(self, cutoff=5.0, n_rbf=50, trainable=False):
        super(ExpNormalSmearing, self).__init__()
        if isinstance(cutoff, torch.Tensor):
            cutoff = cutoff.item()
        self.cutoff = cutoff
        self.n_rbf = n_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.n_rbf)
        betas = torch.tensor([(2 / self.n_rbf * (1 - start_value)) ** -2] * self.n_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        # Match the paper's notation where the radial basis φ(r) and cutoff ϕ(r)
        # are applied as separate factors in the interaction equations.
        return torch.exp(
            -self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2
        )


def str2basis(input_str):
    """
    Convert a radial basis function name string to the corresponding class.

    Args:
        input_str (Union[str, Callable]): Name of the basis function ('BesselBasis', 'GaussianRBF', 'expnorm')
                                          or already a callable class.

    Returns:
        Callable: The radial basis function class.

    Raises:
        ValueError: If the input string is unknown.
    """
    if not isinstance(input_str, str):
        return input_str  # Assume it's already a callable class

    normalized_input = normalize_string(input_str)

    if normalized_input == "besselbasis":
        radial_basis = BesselBasis
    elif input_str == "GaussianRBF":
        radial_basis = GaussianRBF
    elif input_str.lower() == "expnorm":
        radial_basis = ExpNormalSmearing
    else:
        raise ValueError("Unknown radial basis: {}".format(input_str))

    return radial_basis
