from unittest.mock import patch

import pytest
from datamate import set_root_context

from flyvis.solver import MultiTaskSolver
from flyvis.utils.config_utils import get_default_config


@pytest.fixture(scope="module")
def solver(mock_sintel_data, tmp_path_factory) -> MultiTaskSolver:
    with patch('flyvis.datasets.sintel_utils.download_sintel') as mock_download:
        mock_download.return_value = mock_sintel_data

        config = get_default_config(
            path="../../config/solver.yaml",
            overrides=[
                "task_name=flow",
                "ensemble_and_network_id=0",
                "task.n_iters=50",
            ],
        )

        with set_root_context(str(tmp_path_factory.mktemp("tmp"))):
            solver = MultiTaskSolver("test", config)
        return solver


def test_solver_config():
    config = get_default_config(
        path="../../config/solver.yaml",
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
