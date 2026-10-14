import numpy as np
import pytest
import xarray as xr

from flyvis.analysis.moving_bar_responses import (
    angular_distance_to_known,
    correlation_to_known_tuning_curves,
    peak_responses,
    preferred_direction,
)
from flyvis.utils import groundtruth_utils

CELL_TYPES = ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]


def moving_edge_dataset(offsets=(-10, 11), speeds=(19.0, 25.0), dt=1 / 200):
    """Central T4/T5 responses that follow the known tuning curves during the edge."""
    angles = np.arange(0, 360, 30)
    t_stim = {
        s: len(range(*offsets)) * np.radians(2.25) / (s * np.radians(5.8)) for s in speeds
    }
    rows = [(a, 80, i, t_stim[s], s) for a in angles for i in (0, 1) for s in speeds]
    angle, width, intensity, stim_time, speed = map(np.array, zip(*rows))
    time = np.arange(-1.0, max(t_stim.values()) + 1.0, dt)
    responses = np.zeros((1, len(rows), len(time), len(CELL_TYPES)))
    for n, cell_type in enumerate(CELL_TYPES):
        tuning = groundtruth_utils.tuning_curves[cell_type]
        polarity = 1 if cell_type.startswith("T4") else 0
        for k in range(len(rows)):
            if intensity[k] == polarity:
                window = (time > 0.05) & (time < stim_time[k] - 0.05)
                responses[0, k, window, n] = tuning[k // 4]
    return xr.Dataset(
        {"responses": (["network_id", "sample", "frame", "neuron"], responses)},
        coords={
            "network_id": [0],
            "sample": np.arange(len(rows)),
            "angle": ("sample", angle),
            "width": ("sample", width),
            "intensity": ("sample", intensity),
            "t_stim": ("sample", stim_time),
            "speed": ("sample", speed),
            "frame": np.arange(len(time)),
            "time": ("frame", time),
            "neuron": np.arange(len(CELL_TYPES)),
            "cell_type": ("neuron", CELL_TYPES),
        },
        attrs={"config": {"offsets": list(offsets), "dt": dt, "t_pre": 1.0}},
    )


def by_cell_type(da):
    return dict(zip(da.cell_type.values.tolist(), da.values.squeeze().tolist()))


@pytest.fixture(scope="module")
def dataset():
    return moving_edge_dataset()


@pytest.fixture(scope="module")
def reordered(dataset):
    # same data with T4a and T4b swapped along the neuron dimension
    return dataset.isel(neuron=[1, 0, 2, 3, 4, 5, 6, 7])


def test_correlation_to_known_tuning_curves_matches_by_cell_type(dataset, reordered):
    for data in (dataset, reordered):
        correlation = correlation_to_known_tuning_curves(data)
        for i, cell_type in enumerate(correlation.cell_type.values):
            polarity = 1 if cell_type.startswith("T4") else 0
            value = correlation.isel(neuron=i).sel(intensity=polarity).item()
            assert value == pytest.approx(1.0), cell_type


def test_angular_distance_to_known_matches_by_cell_type(dataset, reordered):
    expected = by_cell_type(angular_distance_to_known(preferred_direction(dataset)))
    result = by_cell_type(angular_distance_to_known(preferred_direction(reordered)))
    assert result == pytest.approx(expected)
    assert max(expected.values()) < np.pi / 4


@pytest.mark.parametrize("offsets", [(-10, 12), (-11, 12), (-12, 16)])
def test_peak_responses_offsets(offsets):
    peak = peak_responses(moving_edge_dataset(offsets=offsets))
    assert peak.sizes["neuron"] == len(CELL_TYPES)
