from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from datamate import Directory, Namespace

from flyvis import results_dir
from flyvis.network.ensemble import Ensemble, TaskError
from flyvis.network.ensemble_view import EnsembleView
from flyvis.network.network import IntegrationWarning, Network

pytestmark = pytest.mark.require_download


@pytest.fixture(scope="function")
def ensemble() -> Ensemble:
    models = [results_dir / f"flow/0000/{i:03}" for i in range(4)]
    ensemble = Ensemble(
        models,
        best_checkpoint_fn_kwargs={
            "validation_subdir": "validation",
            "loss_file_name": "loss",
        },
    )
    return ensemble


def test_ensemble_init(ensemble):
    assert isinstance(ensemble, Ensemble)
    assert len(ensemble) == 4
    assert hasattr(ensemble, "names")
    assert hasattr(ensemble, "name")
    assert hasattr(ensemble, "path")
    assert hasattr(ensemble, "model_paths")
    assert hasattr(ensemble, "dir")


def test_yield_networks(ensemble):
    networks = list(ensemble.yield_networks())
    assert len(networks) == len(ensemble)
    assert isinstance(networks[0], Network)


def test_simulate(ensemble: Ensemble):
    x = torch.ones(1, 2, 1, 721).random_(2)
    network = next(ensemble.yield_networks())
    with pytest.warns(IntegrationWarning):
        activity = np.array(list(ensemble.simulate(x, 1)))
    assert activity.shape == (len(ensemble), 1, 2, network.n_nodes)

    with pytest.raises(ValueError):
        activity = np.array(list(ensemble.simulate(torch.ones(1, 2, 721).random_(2), 1)))


def test_validation_losses(ensemble):
    losses = ensemble.validation_losses()
    assert len(losses) == len(ensemble)


def test_rank_by_validation_error(ensemble):
    sorted_names = deepcopy(ensemble.names)

    # destroy current task sorting in place
    np.random.shuffle(ensemble.names)

    random_names = deepcopy(ensemble.names)

    with ensemble.rank_by_validation_error():
        assert ensemble.names == sorted_names
        assert ensemble.names != random_names

    with ensemble.rank_by_validation_error(reverse=True):
        assert ensemble.names == sorted_names[::-1]

    assert ensemble.names == random_names
    ensemble.names = sorted_names


def test_ratio(ensemble):
    names = deepcopy(ensemble.names)

    with ensemble.ratio(best=0.5):
        assert len(ensemble) == int(0.5 * len(names))
        assert ensemble.names == names[: int(0.5 * len(names))]

    with ensemble.ratio(worst=0.5):
        assert len(ensemble) == int(0.5 * len(names))
        assert ensemble.names == names[int(0.5 * len(names)) :]

    with ensemble.ratio(best=1 / 3, worst=1 / 3):
        assert len(ensemble) == int(2 / 3 * len(names))
        assert ensemble.names == [
            *names[: int(len(names) * 1 / 3)],
            *names[len(names) - int(len(names) * 1 / 3) :],
        ]

    assert ensemble.names == names


def test_task_error(ensemble):
    task_error = ensemble.task_error()
    assert isinstance(task_error, TaskError)
    assert len(task_error.values) == len(ensemble)
    assert len(task_error.values) == len(task_error.colors)


@pytest.mark.slow
def test_cluster_indices():
    ensemble = Ensemble(
        "flow/0000",
        best_checkpoint_fn_kwargs={
            "validation_subdir": "validation",
            "loss_file_name": "loss",
        },
    )
    for cell_type in ensemble.connectome.unique_cell_types[:].astype(str):
        cluster_indices = ensemble.cluster_indices(cell_type)
        assert isinstance(cluster_indices, dict)
        assert type(cluster_indices[0]) is np.ndarray
        assert cluster_indices[0].dtype is np.dtype("int64")
        assert set(
            sorted(np.concatenate(list(cluster_indices.values()), axis=0))
        ).issubset(set(np.arange(len(ensemble)).tolist()))


def test_loss_histogram(ensemble: Ensemble):
    ensemble_view = EnsembleView(
        ensemble,
        best_checkpoint_fn_kwargs={
            "validation_subdir": "validation",
            "loss_file_name": "loss",
        },
    )
    fig, ax = ensemble_view.task_error_histogram()
    fig.show()
    plt.close(fig)


@pytest.mark.slow
def test_responses():
    ensemble = Ensemble(
        results_dir / "flow/0000",
        best_checkpoint_fn_kwargs={
            "validation_subdir": "validation",
            "loss_file_name": "loss",
        },
    )

    test_data_dir = Directory(Path(__file__).parent / "data")
    assert "responses" in test_data_dir
    test_responses = test_data_dir.responses[:]
    assert test_responses.shape == (50, 12, 1, 65)
    config = test_data_dir.config
    assert config == Namespace(
        method="steady_state",
        t_pre=0.25,
        dt=1 / 50,
        batch_size=1,
        value=0.5,
        state=None,
        grad=False,
        return_last=False,
    )
    method = config.pop("method")
    ensemble_responses = []
    central_cells_index = None
    for network in ensemble.yield_networks():
        if central_cells_index is None:
            central_cells_index = network.connectome.central_cells_index[:]
        if isinstance(method, str):
            method = getattr(network, method)
        states = method(**config)
        activity = np.array([s.nodes.activity.cpu().numpy() for s in states])[
            :, :, central_cells_index
        ]
        ensemble_responses.append(activity)
    ensemble_responses = np.array(ensemble_responses)
    assert ensemble_responses.shape == test_responses.shape
    assert np.allclose(ensemble_responses, test_responses, atol=1e-4)
