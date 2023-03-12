from pathlib import Path

import torch
import numpy as np
import pytest
from dvs.datasets.sintel import (
    MultiTaskSintel,
    AugmentedSintel,
    AugmentedSintelLum,
)
from dvs.datasets.dots import Dots
from dvs.datasets.flicker import RectangularFlicker
from dvs.datasets.moving_bar import Movingbar
from dvs.datasets.flashes import Flashes, BarFlashes, TwoBarFlashes
from dvs.datasets.oriented_bar import OrientedBar
from dvs.datasets.gratings import Gratings
from dvs.datasets.gratings_v2 import Gratings_v2
from dvs.analysis.time_constants import WhiteNoise


@pytest.mark.parametrize(
    "dataset_class", [MultiTaskSintel, AugmentedSintel, AugmentedSintelLum]
)
def test_sintel(dataset_class):
    dataset = dataset_class()
    dataset.dt = 1 / 10
    data = dataset[0]

    if isinstance(data, dict):
        assert "lum" in data and "flow" in data
        assert isinstance(data["lum"], torch.Tensor) and isinstance(
            data["flow"], torch.Tensor
        )
    else:
        assert isinstance(data, torch.Tensor)


@pytest.mark.parametrize(
    "dataset_class",
    [
        Flashes,
        Movingbar,
        Dots,
        RectangularFlicker,
        BarFlashes,
        TwoBarFlashes,
        OrientedBar,
        Gratings,
        # Gratings_v2,
        # WhiteNoise,
    ],
)
def test_stimulus(dataset_class):
    dataset = dataset_class()
    dataset.dt = 1 / 10
    data = dataset[0]
    assert isinstance(data, torch.Tensor)
