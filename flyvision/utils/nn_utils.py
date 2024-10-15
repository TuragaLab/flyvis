"""Utility functions for operations on torch.nn.Modules."""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator

from torch import nn


@contextmanager
def simulation(network: nn.Module) -> Generator[None, None, None]:
    """
    Context manager to turn off training mode and require_grad for a network.

    Args:
        network: The neural network module to simulate.

    Yields:
        None

    Example:
        ```python
        model = MyNeuralNetwork()
        with simulation(model):
            # Perform inference or evaluation
            output = model(input_data)
        ```

    Note:
        This context manager temporarily disables gradient computation and sets
        the network to evaluation mode. It restores the original state after
        exiting the context.
    """
    _training = network.training
    network.training = False
    params_require_grad = {}
    for name, p in network.named_parameters():
        params_require_grad[name] = p.requires_grad
        p.requires_grad = False
    try:
        yield
    finally:
        network.training = _training
        for name, p in network.named_parameters():
            p.requires_grad = params_require_grad[name]


@dataclass
class NumberOfParams:
    """
    Dataclass to store the number of free and fixed parameters.

    Attributes:
        free: The number of trainable parameters.
        fixed: The number of non-trainable parameters.
    """

    free: int
    fixed: int


def n_params(nnmodule: nn.Module) -> NumberOfParams:
    """
    Returns the numbers of free and fixed parameters in a PyTorch module.

    Args:
        nnmodule: The PyTorch module to analyze.

    Returns:
        A NumberOfParams object containing the count of free and fixed parameters.

    Example:
        ```python
        model = MyNeuralNetwork()
        param_count = n_params(model)
        print(f"Free parameters: {param_count.free}")
        print(f"Fixed parameters: {param_count.fixed}")
        ```
    """
    n_free = 0
    n_fixed = 0
    for param in nnmodule.parameters():
        if param.requires_grad:
            n_free += param.nelement()
        else:
            n_fixed += param.nelement()
    return NumberOfParams(n_free, n_fixed)
