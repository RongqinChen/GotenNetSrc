import inspect
from functools import partial
from typing import List

import torch
from torch import nn as nn
from torch.nn.init import constant_, xavier_uniform_
from torch_geometric.nn.inits import glorot_orthogonal

from src.common.logging import get_logger
from src.model.layers.activations import shifted_softplus

zeros_initializer = partial(constant_, val=0.0)
log = get_logger(__name__)


class ScaleShift(nn.Module):
    """
    Scale and shift layer for standardization.

    Applies `y = x * stddev + mean`. Useful for normalizing outputs.

    Args:
        mean (torch.Tensor or float): Mean value (`mu`).
        stddev (torch.Tensor or float): Standard deviation value (`sigma`).
    """

    def __init__(self, mean, stddev):
        super(ScaleShift, self).__init__()
        if isinstance(mean, float):
            mean = torch.FloatTensor([mean])
        if isinstance(stddev, float):
            stddev = torch.FloatTensor([stddev])
        self.register_buffer("mean", mean)
        self.register_buffer("stddev", stddev)

    def forward(self, input):
        """Compute layer output.

        Args:
            input (torch.Tensor): input data.

        Returns:
            torch.Tensor: layer output.

        """
        y = input * self.stddev + self.mean
        return y


class GetItem(nn.Module):
    """
    Extraction layer to get an item from a dictionary-like input.

    Args:
        key (str): Key of the item to be extracted from the input dictionary.
    """

    def __init__(self, key):
        super(GetItem, self).__init__()
        self.key = key

    def forward(self, inputs):
        """Compute layer output.
        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.
        Returns:
            torch.Tensor: layer output.
        """
        return inputs[self.key]


class SchnetMLP(nn.Module):
    """
    Multiple layer fully connected perceptron neural network, based on SchNet.

    Args:
        n_in (int): Number of input features.
        n_out (int): Number of output features.
        n_hidden (list of int or int, optional): Number of hidden layer nodes.
            If an integer, uses the same number for all hidden layers.
            If None, creates a pyramidal network where layer size is halved. Defaults to None.
        n_layers (int, optional): Total number of layers (including input and output). Defaults to 2.
        activation (callable, optional): Activation function for hidden layers. Defaults to shifted_softplus.
    """

    def __init__(
        self, n_in, n_out, n_hidden=None, n_layers=2, activation=shifted_softplus
    ):
        super(SchnetMLP, self).__init__()
        # get list of number of nodes in input, hidden & output layers
        if n_hidden is None:
            c_neurons = n_in
            self.n_neurons = []
            for _i in range(n_layers):
                self.n_neurons.append(c_neurons)
                c_neurons = c_neurons // 2
            self.n_neurons.append(n_out)
        else:
            # get list of number of nodes hidden layers
            if type(n_hidden) is int:
                n_hidden = [n_hidden] * (n_layers - 1)
            self.n_neurons = [n_in] + n_hidden + [n_out]

        # assign a Dense layer (with activation function) to each hidden layer
        layers = [
            Dense(self.n_neurons[i], self.n_neurons[i + 1], activation=activation)
            for i in range(n_layers - 1)
        ]
        # assign a Dense layer (without activation function) to the output layer
        layers.append(Dense(self.n_neurons[-2], self.n_neurons[-1], activation=None))
        # put all layers together to make the network
        self.out_net = nn.Sequential(*layers)

    def forward(self, inputs):
        """Compute neural network output.
        Args:
            inputs (torch.Tensor): network input.
        Returns:
            torch.Tensor: network output.
        """
        return self.out_net(inputs)


def glorot_orthogonal_wrapper_(tensor, scale=2.0):
    """
    Wrapper for glorot_orthogonal initialization.

    Args:
        tensor (Tensor): Tensor to initialize.
        scale (float, optional): Scaling factor. Defaults to 2.0.

    Returns:
        Tensor: Initialized tensor.
    """
    return glorot_orthogonal(tensor, scale=scale)


def _standardize(kernel):
    """
    Standardize a kernel tensor to have zero mean and unit variance.

    Ensures Var(W) = 1 and E[W] = 0.

    Args:
        kernel (Tensor): The kernel tensor to standardize.

    Returns:
        Tensor: The standardized kernel tensor.
    """
    eps = 1e-6

    if len(kernel.shape) == 3:
        axis = [0, 1]  # last dimension is output dimension
    else:
        axis = 1

    var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel


def he_orthogonal_init(tensor):
    """
    Initialize weights using He initialization with an orthogonal basis.

    Combines He initialization variance scaling with an orthogonal matrix,
    aiming for better decorrelation of features.

    Args:
        tensor (Tensor): The weight tensor to initialize.

    Returns:
        Tensor: The initialized weight tensor.
    """
    tensor = torch.nn.init.orthogonal_(tensor)

    if len(tensor.shape) == 3:
        fan_in = tensor.shape[:-1].numel()
    else:
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor


def get_weight_init_by_string(init_str):
    """
    Get a weight initialization function based on its string name.

    Args:
        init_str (str): Name of the initialization method (e.g., 'zeros', 'xavier_uniform').

    Returns:
        Callable: The corresponding weight initialization function.

    Raises:
        ValueError: If the initialization string is unknown.
    """
    if init_str == "":
        # No-op
        return lambda x: x
    elif init_str == "zeros":
        return torch.nn.init.zeros_
    elif init_str == "xavier_uniform":
        return torch.nn.init.xavier_uniform_
    elif init_str == "glo_orthogonal":
        return glorot_orthogonal_wrapper_
    elif init_str == "he_orthogonal":
        return he_orthogonal_init
    else:
        raise ValueError(f"Unknown initialization {init_str}")


# train.py -m label=mu,alpha,homo,lumo,r2,zpve,U0,U,H,G,Cv name='${label_str}_int6_glo-ort_3090' hydra.sweeper.n_jobs=1 model.representation.n_interactions=6 model.representation.weight_init=glo_orthogonal


class Dense(nn.Linear):
    """
    Fully connected linear layer with optional activation and normalization.
    Applies a linear transformation followed by optional normalization and activation.
    Borrowed from https://github.com/atomistic-machine-learning/schnetpack/blob/master/src/schnetpack/nn/base.py

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If False, the layer will not adapt bias. Defaults to True.
        activation (callable, optional): Activation function. If None, no activation is used. Defaults to None.
        weight_init (callable, optional): Weight initializer. Defaults to xavier_uniform_.
        bias_init (callable, optional): Bias initializer. Defaults to zeros_initializer.
        norm (str, optional): Normalization type ('layer', 'batch', 'instance', or None). Defaults to None.
        gain (float, optional): Gain for weight initialization if applicable. Defaults to None.
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
        norm=None,
        gain=None,
    ):
        # initialize linear layer y = xW^T + b
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.gain = gain
        super(Dense, self).__init__(in_features, out_features, bias)
        # Initialize activation function
        if inspect.isclass(activation):
            self.activation = activation()
        self.activation = activation

        if norm == "layer":
            self.norm = nn.LayerNorm(out_features)
        elif norm == "batch":
            self.norm = nn.BatchNorm1d(out_features)
        elif norm == "instance":
            self.norm = nn.InstanceNorm1d(out_features)
        else:
            self.norm = None

    def reset_parameters(self):
        """Reinitialize model weight and bias values."""
        if self.gain:
            self.weight_init(self.weight, gain=self.gain)
        else:
            self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, inputs):
        """Compute layer output.

        Args:
            inputs (dict of torch.Tensor): batch of input values.

        Returns:
            torch.Tensor: layer output.

        """
        # compute linear layer y = xW^T + b
        y = super(Dense, self).forward(inputs)
        if self.norm is not None:
            y = self.norm(y)
        # add activation function
        if self.activation:
            y = self.activation(y)
        return y


class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable hidden dimensions and activations.

    Args:
        hidden_dims (List[int]): List defining the dimensions of each layer,
            including input and output (e.g., [in_dim, hid1_dim, ..., out_dim]).
        bias (bool, optional): Whether to use bias in linear layers. Defaults to True.
        activation (callable, optional): Activation function for hidden layers. Defaults to None.
        last_activation (callable, optional): Activation function for the output layer. Defaults to None.
        weight_init (callable, optional): Weight initialization function. Defaults to xavier_uniform_.
        bias_init (callable, optional): Bias initialization function. Defaults to zeros_initializer.
        norm (str, optional): Normalization type ('layer', 'batch', 'instance', or ''). Defaults to ''.
    """

    def __init__(
        self,
        hidden_dims: List[int],
        bias=True,
        activation=None,
        last_activation=None,
        weight_init=xavier_uniform_,
        bias_init=zeros_initializer,
        norm="",
    ):
        super().__init__()

        # hidden_dims = [hidden, half, hidden]

        dims = hidden_dims
        n_layers = len(dims)

        DenseMLP = partial(
            Dense, bias=bias, weight_init=weight_init, bias_init=bias_init
        )

        self.dense_layers = nn.ModuleList(
            [
                DenseMLP(dims[i], dims[i + 1], activation=activation, norm=norm)
                for i in range(n_layers - 2)
            ]
            + [DenseMLP(dims[-2], dims[-1], activation=last_activation)]
        )

        self.layers = nn.Sequential(*self.dense_layers)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.dense_layers:
            m.reset_parameters()

    def forward(self, x):
        return self.layers(x)
