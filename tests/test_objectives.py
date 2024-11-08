import pytest
import torch

from flyvis.task.objectives import epe, l2norm


@pytest.mark.parametrize("objective", [l2norm, epe])
def test_objective(objective):
    a = torch.ones(4, 3, 2, 5).random_(to=5)
    b = torch.ones(4, 3, 2, 5).random_(to=5)

    assert objective(a, b) > 0
    assert objective(a, a) == 0
