import pytest

from flyvis.network import dynamics
from flyvis.network.dynamics import NetworkDynamics, activation_fns
from flyvis.utils.class_utils import forward_subclass


@pytest.fixture(
    scope="module",
    params=[
        "NetworkDynamics",
        *[sc.__name__ for sc in dynamics.NetworkDynamics.__subclasses__()],
    ],
)
def type(request):
    return request.param


@pytest.fixture(scope="module", params=list(activation_fns.keys()))
def activation(request):
    return dict(type=request.param)


@pytest.fixture(scope="module")
def dynamics(type, activation):
    return forward_subclass(NetworkDynamics, {"type": type, "activation": activation})


def test_methods_exist(dynamics):
    def has_method(obj, name):
        return hasattr(obj, name) and callable(getattr(obj, name))

    assert has_method(dynamics, "write_derived_params")
    assert has_method(dynamics, "write_initial_state")
    assert has_method(dynamics, "write_state_velocity")
    assert has_method(dynamics, "currents")
