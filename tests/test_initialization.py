import torch
import numpy as np
from datamate import Namespace
import pytest

from flyvision import connectome_file
from flyvision.initialization import InitialDistribution, Parameter
from flyvision.connectome import ConnectomeDir

# -- Fixtures ------------------------------------------------------------------


@pytest.fixture(scope="session")
def connectome(tmp_path_factory):
    return ConnectomeDir(
        tmp_path_factory.mktemp("tmp") / "test",
        dict(file=connectome_file, extent=1, n_syn_fill=1),
    )


@pytest.fixture(
    scope="module",
    params=[
        *[sc.__name__ for sc in InitialDistribution.__subclasses__()],
    ],
)
def initial_dist_type(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        *[sc.__name__ for sc in Parameter.__subclasses__()],
    ],
)
def parameter_type(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=["mean", "sample"],
)
def mode(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[True, False],
)
def grad(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[(1, 1, 1), (1, (1, 1), (1, 1)), ((1, 1), 1, 1)],
)
def value_mean_param(request):
    return request.param


@pytest.fixture(scope="module")
def initial_dist_config(initial_dist_type, grad, mode, value_mean_param):
    value, mean, std = value_mean_param
    return Namespace(
        initial_dist=initial_dist_type,
        value=value,
        mean=mean,
        std=std,
        requires_grad=grad,
        mode=mode,
    )


@pytest.fixture(scope="module")
def param_config(parameter_type, initial_dist_config):
    config = initial_dist_config
    config.update(type=parameter_type)
    return config


# -- InitialDistribution ------------------------------------------------------


def test_initial_distributions(initial_dist_config):
    def isclose(x, y, atol=1e-8, rtol=1e-5):
        return torch.allclose(x, torch.tensor(y, dtype=x.dtype), atol=atol, rtol=rtol)

    initial_dist = InitialDistribution(initial_dist_config)

    assert hasattr(initial_dist, "raw_values")

    assert initial_dist.raw_values.requires_grad == initial_dist_config.requires_grad

    if initial_dist_config.initial_dist == "Value":
        assert isclose(initial_dist.raw_values, initial_dist_config.value)
        assert initial_dist.raw_values.nelement() == np.count_nonzero(
            initial_dist_config.value
        )

    if initial_dist_config.initial_dist in ["Normal", "Lognormal"]:
        if initial_dist_config.mode == "mean":
            assert isclose(initial_dist.raw_values, initial_dist_config.mean)
        elif initial_dist_config.mode == "sample":
            assert not isclose(initial_dist.raw_values, initial_dist_config.mean, 0, 0)
        assert initial_dist.raw_values.nelement() == np.count_nonzero(
            initial_dist_config.mean
        )

    if initial_dist_config.initial_dist == "Lognormal":
        assert torch.equal(
            initial_dist.semantic_values, torch.exp(initial_dist.raw_values)
        )


# -- Parameter -----------------------------------------------------------------


def test_parameter(param_config, connectome):
    if param_config.type == "SynapseCount" and param_config.mode == "sample":
        pytest.skip("SynapseCount does not support sampling")

    param = Parameter(param_config, connectome)

    assert hasattr(param, "parameter") and isinstance(
        param.parameter, InitialDistribution
    )
    assert hasattr(param, "indices") and isinstance(param.indices, torch.Tensor)
    assert hasattr(param, "symmetry_masks") and isinstance(param.symmetry_masks, list)
    assert hasattr(param, "keys") and isinstance(param.keys, list)
    assert param.parameter.raw_values.requires_grad == param_config.requires_grad
    assert param[param.keys[0]]
