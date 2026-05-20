import torch
import torch.nn.functional as F
from torch import nn

from src.model.layers import Dense


class GatedEquivariantBlock(nn.Module):
    """
    The gated equivariant block is used to obtain rotationally invariant and equivariant features to be used
    for tensorial prop.
    """

    def __init__(
        self,
        n_sin: int,
        n_vin: int,
        n_sout: int,
        n_vout: int,
        n_hidden: int,
        activation=F.silu,
        sactivation=None,
    ):
        """
        Initialize the GatedEquivariantBlock.

        Args:
            n_sin (int): Input dimension of scalar features.
            n_vin (int): Input dimension of vectorial features.
            n_sout (int): Output dimension of scalar features.
            n_vout (int): Output dimension of vectorial features.
            n_hidden (int): Size of hidden layers.
            activation: Activation of hidden layers.
            sactivation: Final activation to scalar features.
        """
        super().__init__()
        self.n_sin = n_sin
        self.n_vin = n_vin
        self.n_sout = n_sout
        self.n_vout = n_vout
        self.n_hidden = n_hidden
        self.mix_vectors = Dense(n_vin, 2 * n_vout, activation=None, bias=False)
        self.scalar_net = nn.Sequential(
            Dense(n_sin + n_vout, n_hidden, activation=activation),
            Dense(n_hidden, n_sout + n_vout, activation=None),
        )
        self.sactivation = sactivation

    def forward(self, scalars: torch.Tensor, vectors: torch.Tensor):
        """
        Forward pass of the GatedEquivariantBlock.

        Args:
            scalars (torch.Tensor): Scalar input features.
            vectors (torch.Tensor): Vector input features.

        Returns:
            tuple: Tuple containing:
                - torch.Tensor: Output scalar features.
                - torch.Tensor: Output vector features.
        """
        vmix = self.mix_vectors(vectors)
        vectors_V, vectors_W = torch.split(vmix, self.n_vout, dim=-1)
        vectors_Vn = torch.norm(vectors_V, dim=-2)

        ctx = torch.cat([scalars, vectors_Vn], dim=-1)
        x = self.scalar_net(ctx)
        s_out, x = torch.split(x, [self.n_sout, self.n_vout], dim=-1)
        v_out = x.unsqueeze(-2) * vectors_W

        if self.sactivation:
            s_out = self.sactivation(s_out)

        return s_out, v_out
