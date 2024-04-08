import pytest
import torch

from flyvision.utils.class_utils import forward_subclass
from flyvision.objectives import Objective


@pytest.mark.parametrize("type", ["L2Norm", "EPE"])
def test_objective(type):
    objective = forward_subclass(Objective, dict(type=type))
    a = torch.Tensor(4, 3, 2, 5).random_(to=5)
    b = torch.Tensor(4, 3, 2, 5).random_(to=5)

    assert objective(a, b) > 0
    assert objective(a, a) == 0
