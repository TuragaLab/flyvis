import pytest

from datamate import set_root_context

from flyvision.utils.config_utils import get_default_config
from flyvision.solver import MultiTaskSolver


@pytest.fixture(scope="module")
def solver(tmp_path_factory) -> MultiTaskSolver:
    config = get_default_config(
        config_name="solver",
        overrides=[
            "task_name=flow",
            "network_id=0",
            "task.n_iters=50",
        ],
    )
    with set_root_context(str(tmp_path_factory.mktemp("tmp"))):
        solver = MultiTaskSolver("test", config)
    return solver


def test_solver_config():
    config = get_default_config(
        config_name="solver",
        overrides=[
            "task_name=flow",
            "network_id=0",
        ],
    )
    assert config.task_name == "flow"
    assert config.network_id == 0


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
