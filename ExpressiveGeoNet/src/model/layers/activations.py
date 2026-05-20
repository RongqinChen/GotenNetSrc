import inspect
import math

import torch
import torch.nn.functional as F
from torch import nn as nn


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


class ShiftedSoftplus(nn.Module):
    """
    Shifted Softplus activation function.

    Computes `softplus(x) - log(2)`.
    """

    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class Swish(nn.Module):
    """
    Swish activation function.

    Computes `x * sigmoid(x)`. Also known as SiLU.
    """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


act_class_mapping = {
    "ssp": ShiftedSoftplus,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "swish": Swish,
}


def shifted_softplus(x: torch.Tensor):
    """
    Compute shifted soft-plus activation function.

    Computes `ln(1 + exp(x)) - ln(2)`.

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Shifted soft-plus of input.
    """
    return F.softplus(x) - math.log(2.0)


def normalize_string(s: str) -> str:
    """
    Normalize a string by converting to lowercase and removing dashes, underscores, and spaces.

    Args:
        s (str): Input string.

    Returns:
        str: Normalized string.
    """
    return s.lower().replace("-", "").replace("_", "").replace(" ", "")


def get_activations(optional=False, *args, **kwargs):
    """
    Get a dictionary mapping normalized activation function names to their classes/functions.

    Includes common activations from torch.nn and custom ones like shifted_softplus.
    Reference: https://github.com/sunglasses-ai/classy/blob/3e74cba1fdf1b9f9f2ba1cfcfa6c2017aa59fc04/classy/optim/factories.py#L14

    Args:
        optional (bool, optional): If True, include an empty string key mapping to None. Defaults to False.

    Returns:
        Dict[str, Optional[Callable]]: Dictionary mapping names to activation functions/classes.
    """
    activations = {
        normalize_string(act.__name__): act
        for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, torch.nn.Module)
    }
    activations.update(
        {
            "relu": torch.nn.ReLU,
            "elu": torch.nn.ELU,
            "sigmoid": torch.nn.Sigmoid,
            "silu": torch.nn.SiLU,
            "mish": torch.nn.Mish,
            "swish": torch.nn.SiLU,
            "selu": torch.nn.SELU,
            "softplus": shifted_softplus,
        }
    )

    if optional:
        activations[""] = None

    return activations


def get_activations_none(optional=False, *args, **kwargs):
    """
    Get a dictionary mapping normalized activation function names to their classes/functions,
    excluding softplus-based activations.

    Args:
        optional (bool, optional): If True, include an empty string and None key mapping to None. Defaults to False.

    Returns:
        Dict[str, Optional[Callable]]: Dictionary mapping names to activation functions/classes.
    """
    activations = {
        normalize_string(act.__name__): act
        for act in vars(torch.nn.modules.activation).values()
        if isinstance(act, type) and issubclass(act, torch.nn.Module)
    }
    activations.update(
        {
            "relu": torch.nn.ReLU,
            "elu": torch.nn.ELU,
            "sigmoid": torch.nn.Sigmoid,
            "silu": torch.nn.SiLU,
            "selu": torch.nn.SELU,
        }
    )

    if optional:
        activations[""] = None
        activations[None] = None

    return activations


def dictionary_to_option(options, selected):
    """
    Select an option from a dictionary based on a key, handling potential class instantiation.

    Args:
        options (Dict): Dictionary of options (e.g., activation functions).
        selected (Optional[str]): The key of the selected option.

    Returns:
        Optional[Callable]: The selected option (possibly instantiated if it's a class).

    Raises:
        ValueError: If the selected key is not in the options dictionary.
    """
    if selected not in options:
        raise ValueError(
            f'Invalid choice "{selected}", choose one from {", ".join(list(options.keys()))}'
        )

    activation = options[selected]
    if inspect.isclass(activation):
        activation = activation()
    return activation


def str2act(input_str, *args, **kwargs):
    """
    Convert an activation function name string to the corresponding function/class instance.

    Args:
        input_str (Optional[str]): Name of the activation function (case-insensitive, ignores '-', '_', ' ').
                                   If None or "", returns None.

    Returns:
        Optional[Callable]: The instantiated activation function or None.
    """
    if not input_str:  # Handles None and ""
        return None

    act = get_activations(*args, optional=True, **kwargs)
    out = dictionary_to_option(act, input_str)
    return out
