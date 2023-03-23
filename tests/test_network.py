import pytest
import torch
from datamate import Namespace

import flyvision
from flyvision import Network


@pytest.fixture(scope="module")
def network() -> Network:
    network = Network(
        connectome=Namespace(
            type="ConnectomeDir", file="fib25-fib19_v2.2.json", extent=15, n_syn_fill=1
        ),
        dynamics=Namespace(
            type="PPNeuronIGRSynapses", activation=Namespace(type="relu")
        ),
        node_config=Namespace(
            bias=Namespace(
                type="RestingPotential",
                keys=["type"],
                initial_dist="Normal",
                mode="sample",
                requires_grad=True,
                mean=0.5,
                std=0.05,
                penalize=Namespace(activity=True),
                seed=0,
            ),
            time_const=Namespace(
                type="TimeConstant",
                keys=["type"],
                initial_dist="Value",
                value=0.05,
                requires_grad=True,
            ),
        ),
        edge_config=Namespace(
            sign=Namespace(
                type="SynapseSign", initial_dist="Value", requires_grad=False
            ),
            syn_count=Namespace(
                type="SynapseCount",
                initial_dist="Lognormal",
                mode="mean",
                requires_grad=False,
                std=1.0,
            ),
            syn_strength=Namespace(
                type="SynapseCountScaling",
                initial_dist="Value",
                requires_grad=True,
                scale_elec=0.01,
                scale_chem=0.01,
                clamp="non_negative",
            ),
        ),
    )
    return network


def test_init(network):
    assert isinstance(network, Network)
    assert hasattr(network, "connectome")
    assert hasattr(network, "dynamics")
    assert hasattr(network, "node_params")
    assert hasattr(network, "edge_params")
    assert hasattr(network, "config")
    assert hasattr(network, "symmetry_config")
    assert hasattr(network, "clamp_config")
    assert hasattr(network, "_elec_indices")
    assert hasattr(network, "_chem_indices")
    assert hasattr(network, "_state_hooks")
    assert hasattr(network, "num_parameters")
    assert hasattr(network, "stimulus")


def test_param_api(network):

    param_api = network._param_api()
    assert len(param_api) == 4
    assert param_api.get("nodes", None) is not None
    assert param_api.get("edges", None) is not None
    assert param_api.get("sources", None) is not None
    assert param_api.get("targets", None) is not None
    assert isinstance(
        param_api.get("nodes").get(list(param_api.get("nodes").keys())[0], None),
        flyvision.utils.tensor_utils.RefTensor,
    )
    assert len(param_api.sources.bias) == network.n_edges
    assert len(param_api.targets.bias) == network.n_edges


def test_target_sum(network):

    x = torch.Tensor(2, network.n_edges)
    y = network.target_sum(x)
    assert y.shape == (2, network.n_nodes)


def test_initial_state(network):

    param_api = network._param_api()
    state = network._initial_state(param_api, 2)
    assert state.get("nodes", None) is not None
    assert state.get("edges", None) is not None
    assert state.get("sources", None) is not None
    assert state.get("targets", None) is not None
    assert state.nodes.activity.shape == (2, network.n_nodes)


def test_next_state(network):

    param_api = network._param_api()
    state = network._initial_state(param_api, 2)
    x_t = torch.Tensor(2, network.n_nodes)
    dt = 0.02
    state = network._next_state(param_api, state, x_t, dt)
    assert state.get("nodes", None) is not None
    assert state.get("edges", None) is not None
    assert state.get("sources", None) is not None
    assert state.get("targets", None) is not None
    assert state.sources.activity.shape == (2, network.n_edges)


def test_state_hook(network):
    assert not network._state_hooks
    assert not hasattr(network, "times_hooked")

    param_api = network._param_api()

    def record_hook(state):
        assert state.get("nodes", None) is not None
        assert state.get("edges", None) is not None
        assert state.nodes.activity.shape == (2, network.n_nodes)
        if not hasattr(network, "times_hooked"):
            network.times_hooked = 0
        network.times_hooked += 1

    network.register_state_hook(record_hook)
    state = network._initial_state(param_api, 2)
    assert network.times_hooked == 1
    x_t = torch.Tensor(2, network.n_nodes)
    dt = 0.02
    state = network._next_state(param_api, state, x_t, dt)
    assert network.times_hooked == 2

    def modify_state_hook(state, value):
        activity = state.nodes.activity.clone()
        activity *= value
        state.nodes.update(activity=activity)
        return state

    _activity = state.nodes.activity.clone()
    network.register_state_hook(modify_state_hook, value=2)
    modified_state = network._state_api(state)
    assert (modified_state.nodes.activity == 2 * _activity).all()
    assert network.times_hooked == 3


def test_forward(network):

    x = torch.Tensor(2, 20, network.n_nodes).random_(2)
    activity = network.forward(x, 1 / 50)
    assert activity.shape == (2, 20, network.n_nodes)
    assert activity.grad_fn.name() == "StackBackward0"


def test_simulate(network):
    x = torch.Tensor(2, 20, 1, 721).random_(2)
    activity = network.simulate(x, 1 / 50)
    assert activity.shape == (2, 20, network.n_nodes)

    with pytest.raises(ValueError):
        network.simulate(torch.Tensor(20, 1, 721).random_(2), 1 / 50)

    with pytest.raises(ValueError):
        network.simulate(x, 1 / 49)


def test_simulate_clamp(network):

    x = torch.Tensor(2, 20, 1, 721).random_(2)
    activity = network.simulate_clamp(
        x, 1 / 50, None, "T4c", mode="central", substitute=0.0
    )
    assert (activity[:, :, network.stimulus.central_cells_index["T4c"]] == 0).all()
    assert not network._state_hooks

    activity = network.simulate_clamp(
        x, 1 / 50, None, "R1", mode="layer", substitute="resting"
    )
    assert (
        activity[:, :, network.stimulus.layer_index["R1"]]
        == network.node_params.bias["R1"].data.expand(
            *activity.shape[:2], len(network.stimulus.layer_index["R1"])
        )
    ).all()
    assert not network._state_hooks

    activity = network.simulate_clamp(
        x, 1 / 50, None, ["R1", "R2", "R3"], mode="layer", substitute=1.0
    )
    assert all(
        [
            torch.eq(activity[:, :, network.stimulus.layer_index[cell_type]], 1.0)
            .all()
            .item()
            for cell_type in ["R1", "R2", "R3"]
        ]
    )
    assert not network._state_hooks

    with pytest.raises(ValueError):
        network.simulate_clamp(x, 1 / 50, None, "test", mode="layer", substitute=0.5)


def test_steady_state(network):
    pass


def test_fade_in_state(network):
    pass
