import pytest
from datamate import set_root_context

from flyvision.solver import MultiTaskSolver
from flyvision.utils.config_utils import get_default_config

# add large_download mark to deselect this test in CI
pytestmark = pytest.mark.require_large_download


@pytest.fixture(scope="module")
def solver(tmp_path_factory) -> MultiTaskSolver:
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
