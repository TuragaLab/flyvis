import numpy as np
import pytest
import torch
from datamate import set_root_context

from flyvis.network.directories import NetworkDir
from flyvis.solver import MultiTaskSolver
from flyvis.utils.config_utils import get_default_config


@pytest.fixture(scope="module")
def solver(mock_sintel_data, tmp_path_factory) -> MultiTaskSolver:
    config = get_default_config(
        path="../../flyvis/config/solver.yaml",
        overrides=[
            "task_name=flow",
            "ensemble_and_network_id=0",
            "task.n_iters=50",
            f"+task.dataset.sintel_path={str(mock_sintel_data)}",
            "task.original_split=false",
            "task.dataset.boxfilter.extent=1",
            "task.dataset.n_frames=4",
            "task.dataset.dt=0.041",
            "task.batch_size=2",
            "network.connectome.extent=1",
        ],
    )
    with set_root_context(str(tmp_path_factory.mktemp("tmp"))):
        return MultiTaskSolver("test", config)


def test_solver_config():
    config = get_default_config(
        path="../../flyvis/config/solver.yaml",
        overrides=[
            "task_name=flow",
            "ensemble_and_network_id=0",
        ],
    )
    assert config.task_name == "flow"
    assert config.ensemble_and_network_id == 0


@pytest.mark.slow
def test_solver_init(solver):
    assert isinstance(solver, MultiTaskSolver)
    assert solver.dir.path.exists()
    assert solver.network
    assert solver.task
    assert solver.decoder
    assert solver.optimizer
    assert solver.penalty
    assert solver.scheduler


def test_solver_overfit(solver):
    solver.train(overfit=True)
    loss = solver.dir.loss[:]
    assert loss[-1] < loss[0]


def small_config(mock_sintel_data, n_iters):
    return get_default_config(
        path="../../flyvis/config/solver.yaml",
        overrides=[
            "task_name=flow",
            "ensemble_and_network_id=0",
            f"task.n_iters={n_iters}",
            f"+task.dataset.sintel_path={str(mock_sintel_data)}",
            "task.original_split=false",
            "task.dataset.boxfilter.extent=1",
            "task.dataset.n_frames=4",
            "task.dataset.dt=0.041",
            "task.batch_size=2",
            "network.connectome.extent=1",
            "scheduler.chkpt_every_epoch=1",
        ],
    )


def test_solver_continue_training_keeps_history(mock_sintel_data, tmp_path):
    with set_root_context(str(tmp_path)):
        solver = MultiTaskSolver("test", small_config(mock_sintel_data, n_iters=2))
        solver.train()
        loss = solver.dir.loss[:]
        assert len(loss) == solver.iteration == 2

        solver.task.n_iters = 4
        solver.train()
        assert len(solver.dir.loss[:]) == solver.iteration == 4
        np.testing.assert_array_equal(solver.dir.loss[:2], loss)


def test_solver_recover(mock_sintel_data, tmp_path):
    config = small_config(mock_sintel_data, n_iters=2)
    with set_root_context(str(tmp_path)):
        solver = MultiTaskSolver("test", config)
        solver.train()
        state = {k: v.clone() for k, v in solver.network.state_dict().items()}

        resumed = MultiTaskSolver("test", config)
        resumed.recover(checkpoint=-1)
        assert resumed.iteration == solver.iteration
        assert resumed._last_chkpt_ind == solver._last_chkpt_ind
        for key, value in resumed.network.state_dict().items():
            torch.testing.assert_close(value, state[key])

        # the initial checkpoint was made before the first iteration
        resumed.recover(checkpoint=0)
        assert resumed.iteration == 0

        resumed.recover(checkpoint="best")
        assert resumed._curr_chkpt_ind in solver.checkpoints

        resumed.task.n_iters = 4
        resumed.train()
        assert len(resumed.dir.loss[:]) == resumed.iteration == 4

        fresh = MultiTaskSolver("test", config, delete_if_exists=True)
        assert fresh.checkpoints == []
        assert "loss" not in fresh.dir


def test_solver_recover_older_checkpoint_then_checkpoint(mock_sintel_data, tmp_path):
    with set_root_context(str(tmp_path)):
        solver = MultiTaskSolver("test", small_config(mock_sintel_data, n_iters=2))
        solver.train()
        assert solver.checkpoints == [0, 1]

        solver.recover(checkpoint=0)
        solver.checkpoint()
        assert solver._last_chkpt_ind == solver._curr_chkpt_ind == 2


def test_solver_opens_directory_with_stored_delete_flag(mock_sintel_data, tmp_path):
    # directories created before the fix store delete_if_exists in their config
    config = {**small_config(mock_sintel_data, n_iters=2), "delete_if_exists": False}
    with set_root_context(str(tmp_path)):
        legacy = NetworkDir("test", config)

        solver = MultiTaskSolver("test", config)
        assert solver.dir.path == legacy.path

        with pytest.raises(FileExistsError):
            MultiTaskSolver("test", small_config(mock_sintel_data, n_iters=3))
