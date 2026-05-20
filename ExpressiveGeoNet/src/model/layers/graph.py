from typing import List

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing

from src.model.layers.common import MLP
from src.model.layers.cutoff import CosineCutoff


def get_split_sizes_from_lmax(lmax):
    """
    Return split sizes for torch.split based on lmax.

    Calculates the dimensions of spherical harmonic components for each
    angular momentum value from 1 to lmax.

    Args:
        lmax: Maximum angular momentum value.

    Returns:
        List[int]: List of split sizes for torch.split.
    """
    return [2 * l + 1 for l in range(1, lmax + 1)]


class TensorLayerNorm(nn.Module):
    """
    Layer normalization for high-degree steerable features (tensors).

    Applies normalization independently to each degree component of the tensor features.
    Uses max-min normalization within each degree component.

    Args:
        hidden_channels (int): Dimension of the feature channels.
        trainable (bool): Whether the scaling weight is learnable.
        lmax (int, optional): Maximum degree (lmax) of the tensor features. Defaults to 1.
    """

    def __init__(self, hidden_channels, trainable, lmax=1, **kwargs):
        super(TensorLayerNorm, self).__init__()

        self.hidden_channels = hidden_channels
        self.eps = 1e-12
        self.lmax = lmax

        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)

        self.reset_parameters()

    def reset_parameters(self):
        weight = torch.ones(self.hidden_channels)
        self.weight.data.copy_(weight)

    def max_min_norm(self, tensor):
        # Based on VisNet (https://www.nature.com/articles/s41467-023-43720-2)
        dist = torch.norm(tensor, dim=1, keepdim=True)

        if (dist == 0).all():
            return torch.zeros_like(tensor)

        dist = dist.clamp(min=self.eps)
        direct = tensor / dist

        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)

        return F.relu(dist) * direct

    def forward(self, tensor):
        try:
            split_sizes = get_split_sizes_from_lmax(self.lmax)
        except ValueError as e:
            raise ValueError(
                f"TensorLayerNorm received unsupported feature dimension {tensor.shape[1]}: {str(e)}"
            ) from e

        # Split the vector into parts
        vec_parts = torch.split(tensor, split_sizes, dim=1)

        # Normalize each part separately
        normalized_parts = [self.max_min_norm(part) for part in vec_parts]

        # Concatenate the normalized parts
        normalized_vec = torch.cat(normalized_parts, dim=1)

        # Apply weight
        return normalized_vec * self.weight.unsqueeze(0).unsqueeze(0)


class Distance(nn.Module):
    """
    Compute edge distances and vectors between nodes within a cutoff radius.

    Uses torch_cluster.radius_graph to find neighbors.

    Args:
        cutoff (float): Cutoff distance for finding neighbors.
        max_num_neighbors (int, optional): Maximum number of neighbors to consider for each node. Defaults to 32.
        loop (bool, optional): Whether to include self-loops in the graph. Defaults to True.
        direction (str, optional): Direction of edge vectors ('source_to_target' or 'target_to_source').
                                   Defaults to "source_to_target".
    """

    def __init__(
        self, cutoff, max_num_neighbors=32, loop=True, direction="source_to_target"
    ):
        super(Distance, self).__init__()
        if direction not in ["source_to_target", "target_to_source"]:
            raise ValueError(
                f"Unknown direction '{direction}'. Choose 'source_to_target' or 'target_to_source'."
            )
        self.direction = direction
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.loop = loop

    def forward(self, pos, batch):
        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
        )
        if self.direction == "source_to_target":
            # keep as is
            edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
        else:
            edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.loop:
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(
                edge_vec.size(0), device=edge_vec.device, dtype=edge_vec.dtype
            )
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        return edge_index, edge_weight, edge_vec


class NodeInit(MessagePassing):
    """
    Node initialization layer for message passing networks.

    Initializes scalar node features based on atom types and their local environment
    using message passing. Implements Eq. 1 and 2 from the GotenNet paper.

    Args:
        hidden_channels (Union[int, List[int]]): Dimension of hidden channels. If a list, defines MLP layers.
        num_rbf (int): Number of radial basis functions used for edge features.
        cutoff (float): Cutoff distance for interactions.
        max_z (int, optional): Maximum atomic number for embedding lookup. Defaults to 100.
        activation (Callable, optional): Activation function. Defaults to F.silu.
        proj_ln (str, optional): Type of layer normalization for projection ('layer' or ''). Defaults to ''.
        weight_init (Callable, optional): Weight initialization function. Defaults to nn.init.xavier_uniform_.
        bias_init (Callable, optional): Bias initialization function. Defaults to nn.init.zeros_.
    """

    def __init__(
        self,
        hidden_channels,
        num_rbf,
        cutoff,
        max_z=100,
        activation=F.silu,
        proj_ln="",
        weight_init=nn.init.xavier_uniform_,
        bias_init=nn.init.zeros_,
    ):
        super(NodeInit, self).__init__(aggr="add")
        if type(hidden_channels) == int:
            hidden_channels = [hidden_channels]

        last_channel = hidden_channels[-1]
        self.A_nbr = nn.Embedding(max_z, last_channel)
        self.W_ndp = MLP(
            [num_rbf] + [last_channel],
            activation=None,
            norm="",
            weight_init=weight_init,
            bias_init=bias_init,
            last_activation=None,
        )

        self.W_nrd_nru = MLP(
            [2 * last_channel] + hidden_channels,
            activation=activation,
            norm=proj_ln,
            weight_init=weight_init,
            bias_init=bias_init,
            last_activation=None,
        )
        self.cutoff = CosineCutoff(cutoff)
        self.reset_parameters()

    def reset_parameters(self):
        self.A_nbr.reset_parameters()
        self.W_ndp.reset_parameters()
        self.W_nrd_nru.reset_parameters()

    def forward(self, z, h, edge_index, r0_ij, varphi_r0_ij):
        # remove self loops
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            r0_ij = r0_ij[mask]
            varphi_r0_ij = varphi_r0_ij[mask]

        h_src = self.A_nbr(z)
        phi_r0_ij = self.cutoff(r0_ij)
        r0_ij_feat = self.W_ndp(varphi_r0_ij) * phi_r0_ij.view(-1, 1)

        # propagate_type: (h_src: Tensor, r0_ij_feat:Tensor)
        m_i = self.propagate(edge_index, h_src=h_src, r0_ij_feat=r0_ij_feat, size=None)
        return self.W_nrd_nru(torch.cat([h, m_i], dim=1))

    def message(self, h_src_j, r0_ij_feat):
        return h_src_j * r0_ij_feat


class EdgeInit(MessagePassing):
    """
    Edge initialization layer for message passing networks.

    Initializes scalar edge features based on connected node features and radial basis functions.
    Implements Eq. 3 from the GotenNet paper.

    Args:
        num_rbf (int): Number of radial basis functions.
        hidden_channels (int): Dimension of hidden channels (must match node features).
        activation (Callable, optional): Activation function (currently unused). Defaults to None.
    """

    def __init__(self, num_rbf, hidden_channels, activation=None):
        super(EdgeInit, self).__init__(aggr=None)
        self.W_erp = nn.Linear(num_rbf, hidden_channels)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_erp.weight)
        self.W_erp.bias.data.fill_(0)

    def forward(self, edge_index, phi_r0_ij, h):
        # propagate_type: (h: Tensor, phi_r0_ij: Tensor)
        out = self.propagate(edge_index, h=h, phi_r0_ij=phi_r0_ij)
        return out

    def message(self, h_i, h_j, phi_r0_ij):
        return (h_i + h_j) * self.W_erp(phi_r0_ij)

    def aggregate(self, features, index):
        # no aggregate
        return features
