"""Sparse aggregation parity, including the real FlyVis recurrent runtime."""

import pytest
import torch

from flyvis import Network
from flyvis.network.dynamics import activation_fns
from flyvis.network.sparse import SparseCurrent

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def defaults():
    device, dtype = torch.get_default_device(), torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)


def make_network(backend, activation="relu"):
    return Network(
        connectome=dict(
            type="ConnectomeFromAvgFilters",
            file="fib25-fib19_v2.2.json",
            extent=1,
            n_syn_fill=1,
        ),
        dynamics=dict(type="PPNeuronIGRSynapses", activation=dict(type=activation)),
        node_config=dict(
            bias=dict(
                type="RestingPotential",
                groupby=["type"],
                initial_dist="Normal",
                mode="mean",
                requires_grad=True,
                mean=0.5,
                std=0.05,
            ),
            time_const=dict(
                type="TimeConstant",
                groupby=["type"],
                initial_dist="Value",
                value=0.05,
                requires_grad=True,
            ),
        ),
        backend=backend,
    ).to(dtype=torch.get_default_dtype())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "train_values,train_state", [(True, True), (True, False), (False, True)]
)
def test_unsorted_parallel_edges_and_gradients(device, train_values, train_state):
    torch.set_default_device(device)
    source = torch.tensor([2, 0, 2, 4, 1, 0])
    target = torch.tensor([1, 3, 1, 0, 4, 0])
    op = SparseCurrent(source, target, 6)  # Also includes an isolated node.
    w0 = torch.tensor([0.0, 1.0, -0.5, 0.3, -0.2, 0.1])
    h0 = torch.randn(3, 12)[:, ::2]  # Noncontiguous input.
    results = []
    for sparse in (False, True):
        w = w0.clone().requires_grad_(train_values)
        h = h0.detach().requires_grad_(train_state)
        if sparse:
            y = op(h, op.prepare(w))
        else:
            y = torch.zeros(3, 6).scatter_add(
                -1, target.expand(3, -1), w * h.index_select(-1, source)
            )
        y.square().sum().backward()
        results.append((y.detach(), w.grad, h.grad))
    for reference, actual in zip(*results):
        if reference is not None:
            torch.testing.assert_close(actual, reference, atol=1e-10, rtol=1e-9)


def test_finite_difference_gradcheck():
    torch.set_default_device("cpu")
    op = SparseCurrent(torch.tensor([2, 0, 2]), torch.tensor([1, 2, 1]), 4)
    w = torch.randn(3, requires_grad=True)
    h = torch.randn(2, 4, requires_grad=True)
    assert torch.autograd.gradcheck(lambda w, h: op(h, op.prepare(w)), (w, h))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("activation", list(activation_fns))
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_real_network_outputs_and_all_gradients(device, activation, dtype):
    torch.set_default_device(device)
    torch.set_default_dtype(dtype)
    reference = make_network("scatter", activation)
    actual = make_network("sparse_trainable", activation)
    actual.load_state_dict(reference.state_dict(), strict=True)
    x0 = torch.randn(2, 5, reference.n_nodes) * 0.2
    outputs, inputs, gradients = [], [], []
    for model in (reference, actual):
        x = x0.clone().requires_grad_(True)
        y = model(x, dt=0.01)
        y.square().mean().backward()
        outputs.append(y.detach())
        inputs.append(x.grad)
        gradients.append({
            n: p.grad for n, p in model.named_parameters() if p.requires_grad
        })
    tolerances = (
        dict(atol=1e-6, rtol=1e-4)
        if dtype == torch.float32
        else dict(atol=1e-9, rtol=1e-8)
    )
    torch.testing.assert_close(outputs[1], outputs[0], **tolerances)
    torch.testing.assert_close(inputs[1], inputs[0], **tolerances)
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        assert gradients[0][name] is not None, name
        torch.testing.assert_close(gradients[1][name], gradients[0][name], **tolerances)


@pytest.mark.parametrize("device", DEVICES)
def test_hooks_initial_state_currents_and_optimizer_updates(device):
    torch.set_default_device(device)
    reference, actual = make_network("scatter"), make_network("sparse_trainable")
    actual.load_state_dict(reference.state_dict())
    optimizers = [torch.optim.Adam(m.parameters(), lr=1e-4) for m in (reference, actual)]

    def hook(state):
        state.nodes.activity = state.nodes.activity * 0.9 + 0.02
        return state

    for model in (reference, actual):
        model.register_state_hook(hook)
    x = torch.randn(2, 4, reference.n_nodes) * 0.2
    for _ in range(3):
        results = []
        for model, optimizer in zip((reference, actual), optimizers):
            optimizer.zero_grad(set_to_none=True)
            params = model._param_api()
            state = model._initial_state(params, 2)
            h = torch.full((2, model.n_nodes), 0.3, requires_grad=True)
            state.nodes.activity = h
            states = model(x, dt=0.01, state=model._state_api(state), as_states=True)
            y = torch.stack([s.nodes.activity for s in states], dim=1)
            # Per-edge inspection remains in original connectome order.
            currents = model.dynamics.currents(states[-1], params)
            loss = y.square().mean() + currents.square().mean()
            loss.backward()
            results.append((y.detach(), h.grad.clone(), currents.detach()))
            optimizer.step()
        for got, expected in zip(results[1], results[0]):
            torch.testing.assert_close(got, expected, atol=1e-9, rtol=1e-7)
    rebuilt = Network(**actual.config)
    rebuilt.load_state_dict(actual.state_dict(), strict=True)
    assert rebuilt.backend == "sparse_trainable"
    assert set(actual.state_dict()) == set(reference.state_dict())


def test_no_edge_batch_activations_saved():
    torch.set_default_device("cpu")
    model = make_network("sparse_trainable")
    saved_shapes = []

    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        model(torch.randn(3, 5, model.n_nodes), dt=0.01).square().mean().backward()
    assert (3, model.n_edges) not in saved_shapes
    assert (model.n_edges, 3) not in saved_shapes
    assert (model.n_nodes, model.n_nodes) not in saved_shapes


@pytest.mark.parametrize("backend", ["scatter", "sparse_trainable"])
def test_forward_inference_then_training(backend):
    torch.set_default_device("cpu")
    model = make_network(backend)
    with torch.inference_mode():
        model(torch.randn(2, 3, model.n_nodes), dt=0.01)
    model(torch.randn(2, 3, model.n_nodes), dt=0.01).sum().backward()
    assert model.edges_syn_strength.grad is not None


def test_unsupported_dynamics_and_inplace_activation_rejected():
    torch.set_default_device("cpu")
    for dynamics in (
        dict(type="NetworkDynamics", activation=dict(type="relu")),
        dict(type="PPNeuronIGRSynapses", activation=dict(type="relu", inplace=True)),
    ):
        with pytest.raises(ValueError, match="requires PPNeuronIGRSynapses"):
            Network(
                connectome=dict(type="ConnectomeFromAvgFilters", extent=1),
                dynamics=dynamics,
                backend="sparse_trainable",
            )


@pytest.mark.parametrize("device", DEVICES)
def test_empty_edges(device):
    torch.set_default_device(device)
    indices = torch.empty(0, dtype=torch.int64)
    op = SparseCurrent(indices, indices, 4)
    values = torch.empty(0, requires_grad=True)
    h = torch.randn(2, 4, requires_grad=True)
    y = op(h, op.prepare(values))
    y.sum().backward()
    torch.testing.assert_close(y, torch.zeros_like(y))
    torch.testing.assert_close(h.grad, torch.zeros_like(h))
    assert values.grad is not None and values.grad.numel() == 0


@pytest.mark.parametrize("source,target", [([4], [0]), ([-1], [0]), ([0], [4])])
def test_invalid_edge_coordinates_rejected(source, target):
    torch.set_default_device("cpu")
    with pytest.raises(ValueError, match="out of bounds"):
        SparseCurrent(torch.tensor(source), torch.tensor(target), 4)


def test_sparse_layout_moves_with_module_and_stays_out_of_checkpoint():
    torch.set_default_device("cpu")
    op = SparseCurrent(torch.tensor([2, 0]), torch.tensor([1, 2]), 3)
    assert not op.state_dict()
    for device in DEVICES:
        op.to(device)
        h = torch.randn(2, 3, device=device, requires_grad=True)
        w = torch.randn(2, device=device, requires_grad=True)
        op(h, op.prepare(w)).sum().backward()
        assert all(t.device.type == device for t in op.buffers())
        assert h.grad is not None and w.grad is not None
