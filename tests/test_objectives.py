import pytest
import torch

from flyvision.objectives import Loss
from flyvision.utils.class_utils import forward_subclass


@pytest.mark.parametrize("type", ["l2norm", "epe"])
def test_objective(type):
    objective = forward_subclass(Loss, dict(type=type))
    a = torch.ones(4, 3, 2, 5).random_(to=5)
    b = torch.ones(4, 3, 2, 5).random_(to=5)

    assert objective(a, b) > 0
    assert objective(a, a) == 0
