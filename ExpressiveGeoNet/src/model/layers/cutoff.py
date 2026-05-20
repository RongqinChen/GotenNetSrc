import math

import torch
from torch import Tensor
from torch import nn as nn

from src.common.logging import get_logger

log = get_logger(__name__)


class PolynomialCutoff(nn.Module):
    """
    Polynomial cutoff function, as proposed in DimeNet.

    Smoothly reduces values to zero based on a cutoff radius using a polynomial decay.
    Reference: https://arxiv.org/abs/2003.03123

    Args:
        cutoff (float): Cutoff radius.
        p (int, optional): Exponent for the polynomial decay. Defaults to 6.
    """

    def __init__(self, cutoff, p: int = 6):
        super(PolynomialCutoff, self).__init__()
        self.cutoff = cutoff
        self.p = p

    @staticmethod
    def polynomial_cutoff(r: Tensor, rcut: float, p: float = 6.0) -> Tensor:
        """
        Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        """
        if not p >= 2.0:
            raise ValueError(f"Exponent p={p} has to be >= 2.")

        rscaled = r / rcut

        out = 1.0
        out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(rscaled, p))
        out = out + (p * (p + 2.0) * torch.pow(rscaled, p + 1.0))
        out = out - ((p * (p + 1.0) / 2) * torch.pow(rscaled, p + 2.0))

        return out * (rscaled < 1.0).float()

    def forward(self, r):
        return self.polynomial_cutoff(r=r, rcut=self.cutoff, p=self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff}, p={self.p})"


class CosineCutoff(nn.Module):
    """
    Cosine cutoff function.

    Smoothly reduces values to zero based on a cutoff radius using a cosine function.

    Args:
        cutoff (float): Cutoff radius.
    """

    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()

        if isinstance(cutoff, torch.Tensor):
            cutoff = cutoff.item()
        self.cutoff = cutoff

    def forward(self, distances):
        cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff) + 1.0)
        cutoffs = cutoffs * (distances < self.cutoff).float()
        return cutoffs


@torch.jit.script
def safe_norm(x: Tensor, dim: int = -2, eps: float = 1e-8, keepdim: bool = False):
    """
    Compute the norm of a tensor safely, avoiding division by zero.

    Args:
        x (Tensor): Input tensor.
        dim (int, optional): Dimension along which to compute the norm. Defaults to -2.
        eps (float, optional): Small epsilon value for numerical stability. Defaults to 1e-8.
        keepdim (bool, optional): Whether the output tensor has `dim` retained or not. Defaults to False.

    Returns:
        Tensor: The norm of the input tensor.
    """
    return torch.sqrt(torch.sum(x**2, dim=dim, keepdim=keepdim)) + eps
