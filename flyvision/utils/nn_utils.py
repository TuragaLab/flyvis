"""Utility functions for operations on torch.nn.Modules."""
from contextlib import contextmanager


@contextmanager
def simulation(network):
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


def n_params(nnmodule):
    """Returns the numbers of parameters in a module that require_grad."""
    n_tot = 0
    for param in nnmodule.parameters():
        if param.requires_grad:
            n_tot += param.nelement()
    return n_tot
