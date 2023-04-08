"""Utility functions for operations on torch.nn.Modules."""
from dataclasses import dataclass
from contextlib import contextmanager


@contextmanager
def simulation(network):
    """Context manager to turn off training mode and require_grad for a network."""
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
    free: int
    fixed: int


def n_params(nnmodule):
    """Returns the numbers of free and fixed parameters in a pytorch module."""
    n_free = 0
    n_fixed = 0
    for param in nnmodule.parameters():
        if param.requires_grad:
            n_free += param.nelement()
        else:
            n_fixed += param.nelement()
    return NumberOfParams(n_free, n_fixed)
