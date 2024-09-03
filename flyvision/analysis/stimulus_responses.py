"""Stimulus response generators."""

# pylint: disable=dangerous-default-value
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, Optional, Tuple

import numpy as np
import pandas as pd
from datamate import Namespace

import flyvision
from flyvision.datasets.datasets import StimulusDataset
from flyvision.datasets.dots import CentralImpulses, SpatialImpulses
from flyvision.datasets.flashes import Flashes
from flyvision.datasets.moving_bar import MovingBar, MovingEdge
from flyvision.datasets.sintel import AugmentedSintel

from . import optimal_stimuli


@dataclass
class Result:
    """Dataclass for stimulus response results."""

    config: dict
    stim_args: pd.Series | pd.DataFrame
    stimulus: np.ndarray
    responses: np.ndarray


def flash_responses_generator(
    network_view: flyvision.NetworkView,
    dataset: Optional[Flashes] = None,
    radius=[-1, 6],
    dt=1 / 200,
    batch_size=1,
) -> Generator[Result, None, None]:
    """Yield flash responses."""
    network_view.init_network()
    if dataset is None:
        flashes = Namespace(
            dynamic_range=[0, 1],
            t_stim=1,
            t_pre=1.0,
            dt=dt,
            radius=radius,
            alternations=(0, 1, 0),
        )
        dataset = Flashes(**flashes)
    for i, (stimulus, responses) in enumerate(
        network_view.network.stimulus_response(
            dataset, dataset.dt, t_pre=1.0, t_fade_in=0.0, batch_size=batch_size
        )
    ):
        yield Result(
            dataset.config,
            dataset.arg_df.iloc[i * batch_size : (i + 1) * batch_size],
            stimulus,
            responses,
        )


def movingedge_responses_generator(
    network_view: flyvision.NetworkView,
    dataset: Optional[MovingEdge] = None,
    speeds=[2.4, 4.8, 9.7, 13, 19, 25],
    offsets=(-10, 11),
    dt=1 / 200,
    batch_size=1,
) -> Generator[Result, None, None]:
    """Yield moving edge responses."""
    network_view.init_network()
    if dataset is None:
        dataset = MovingEdge(
            offsets=offsets,
            intensities=[0, 1],
            speeds=speeds,
            height=80,
            post_pad_mode="continue",
            dt=dt,
            device="cuda",
            t_pre=1.0,
            t_post=1.0,
        )
    for i, (stimulus, responses) in enumerate(
        network_view.network.stimulus_response(
            dataset, dataset.dt, t_pre=1.0, t_fade_in=0.0, batch_size=batch_size
        )
    ):
        yield Result(
            dataset.config,
            dataset.arg_df.iloc[i * batch_size : (i + 1) * batch_size],
            stimulus,
            responses,
        )


def movingbar_responses_generator(
    network_view: flyvision.NetworkView,
    dataset: Optional[MovingBar] = None,
    dt=1 / 200,
    batch_size=1,
) -> Generator[Result, None, None]:
    """Yield moving bar responses."""
    network_view.init_network()
    if dataset is None:
        dataset = MovingBar(
            widths=[1, 2, 4],
            offsets=(-10, 11),
            intensities=[0, 1],
            speeds=[2.4, 4.8, 9.7, 13, 19, 25],
            height=9,
            post_pad_mode="continue",
            dt=dt,
            t_pre=1.0,
            t_post=1.0,
            device="cuda",
        )
    for i, (stimulus, responses) in enumerate(
        network_view.network.stimulus_response(
            dataset, dataset.dt, t_pre=1.0, t_fade_in=0.0, batch_size=batch_size
        )
    ):
        yield Result(
            dataset.config,
            dataset.arg_df.iloc[i * batch_size : (i + 1) * batch_size],
            stimulus,
            responses,
        )


def naturalistic_stimuli_responses_generator(
    network_view: flyvision.NetworkView,
    dataset: Optional[AugmentedSintel] = None,
    dt=1 / 100,
    batch_size=1,
) -> Generator[Result, None, None]:
    """Yield naturalistic stimuli responses."""
    network_view.init_network()
    if dataset is None:
        config = Namespace(
            tasks=["flow"],
            interpolate=False,
            boxfilter=dict(extent=15, kernel_size=13),
            temporal_split=True,
            dt=dt,
        )
        dataset = AugmentedSintel(**config)

    for i, (stimulus, responses) in enumerate(
        network_view.network.stimulus_response(
            dataset, dataset.dt, t_pre=0.0, t_fade_in=2.0, batch_size=batch_size
        )
    ):
        yield Result(
            dataset.config,
            dataset.arg_df.iloc[i * batch_size : (i + 1) * batch_size],
            stimulus,
            responses,
        )


def central_impulses_responses_generator(
    network_view: flyvision.NetworkView,
    dataset: Optional[CentralImpulses] = None,
    intensity=1,
    bg_intensity=0.5,
    impulse_durations=[5e-3, 20e-3, 50e-3, 100e-3, 200e-3, 300e-3],
    dt=1 / 200,
    batch_size=1,
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    """Yield central ommatidium impulses responses."""
    network_view.init_network()
    if dataset is None:
        config = Namespace(
            impulse_durations=impulse_durations,
            dot_column_radius=0,
            bg_intensity=bg_intensity,
            t_stim=2,
            dt=dt,
            n_ommatidia=721,
            t_pre=1.0,
            t_post=0,
            intensity=intensity,
            mode="impulse",
            device="cuda",
        )
        dataset = CentralImpulses(**config)

    for i, (stimulus, responses) in enumerate(
        network_view.network.stimulus_response(
            dataset, dataset.dt, t_pre=4.0, t_fade_in=0.0, batch_size=batch_size
        )
    ):
        yield Result(
            dataset.config,
            dataset.arg_df.iloc[i * batch_size : (i + 1) * batch_size],
            stimulus,
            responses,
        )


# pylint: disable=dangerous-default-value
def spatial_impulses_responses_generator(
    network_view: flyvision.NetworkView,
    dataset: Optional[SpatialImpulses] = None,
    intensity=1,
    bg_intensity=0.5,
    impulse_durations=[5e-3, 20e-3],
    max_extent=4,
    dt=1 / 200,
    batch_size=1,
) -> Generator[Result, None, None]:
    """Yield spatial ommatidium impulses responses."""
    network_view.init_network()
    if dataset is None:
        config = Namespace(
            impulse_durations=impulse_durations,
            max_extent=max_extent,
            dot_column_radius=0,
            bg_intensity=bg_intensity,
            t_stim=2,
            dt=dt,
            n_ommatidia=721,
            t_pre=1.0,
            t_post=0,
            intensity=intensity,
            mode="impulse",
            device="cuda",
        )
        dataset = SpatialImpulses(**config)
    for i, (stimulus, responses) in enumerate(
        network_view.network.stimulus_response(
            dataset, dataset.dt, t_pre=4.0, t_fade_in=0.0, batch_size=batch_size
        )
    ):
        yield Result(
            dataset.config,
            dataset.arg_df.iloc[i * batch_size : (i + 1) * batch_size],
            stimulus,
            responses,
        )


def optimal_stimulus_responses_generator(
    network_view: flyvision.NetworkView, dataset: Optional[StimulusDataset] = None
) -> Generator[Dict[str, np.ndarray | str], None, None]:
    """Yield optimal stimuli responses."""
    network_view.init_network()
    if dataset is None:
        findoptstim = optimal_stimuli.FindOptimalStimuli(network_view, stimuli="default")
    else:
        findoptstim = optimal_stimuli.FindOptimalStimuli(network_view, stimuli=dataset)

    for cell_type in network_view.cell_types_sorted:
        optstim = findoptstim.regularized_optimal_stimuli(
            cell_type,
        )
        yield {
            "cell_type": cell_type,
            "stimulus": optstim.stimulus.stimulus.cpu().numpy(),
            "response": optstim.stimulus.response.cpu().numpy(),
            "regularized_stimulus": optstim.regularized_stimulus,
            "regularized_response": optstim.response,
            "central_predicted_activity": optstim.central_predicted_response,
            "central_target_activity": optstim.central_target_response,
            "losses": optstim.losses,
        }
