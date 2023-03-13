from pathlib import Path

import torch
import numpy as np
import pytest

import flyvision
from flyvision.datasets.sintel import (
    MultiTaskSintel,
    AugmentedSintel,
    AugmentedSintelLum,
)
from flyvision.networks import Network

default_network_config = flyvision.get_default_config().solver.network


def test_network_forward_steady_state():
    network = Network(**default_network_config)
    steady_state = network.steady_state(1.0, 1 / 10, 1)
    assert list(steady_state.keys()) == ["nodes", "edges", "sources", "targets"]
    assert len(steady_state.nodes.activity) == 1


@pytest.mark.parametrize(
    "dataset_class", [MultiTaskSintel, AugmentedSintel, AugmentedSintelLum]
)
def test_network_stimulus_response(dataset_class):
    dataset = dataset_class()
    network = Network(**default_network_config)
    stimulus_responses = list(
        network.stimulus_response(
            dataset,
            1 / 10,
            indices=[0, 1],
            t_fade_in=0.1,
            t_pre=0.1,
            grad=False,
        )
    )
    assert len(stimulus_responses) == 2
