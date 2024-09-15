"""Stimulus response functions with caching using xarray."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr

import flyvision
from flyvision.analysis import optimal_stimuli
from flyvision.datasets.datasets import StimulusDataset
from flyvision.datasets.dots import CentralImpulses, SpatialImpulses
from flyvision.datasets.flashes import Flashes
from flyvision.datasets.moving_bar import MovingBar, MovingEdge
from flyvision.datasets.sintel import AugmentedSintel

# --------------------- Helper Function ---------------------


@dataclass
class Result:
    stimuli: np.ndarray
    responses: np.ndarray
    arg_df: pd.DataFrame


def compute_responses(
    network: "flyvision.network.CheckpointedNetwork",
    dataset_class: type,
    dataset_config: Dict,
    batch_size: int,
    t_pre: float,
    t_fade_in: float,
) -> Result:
    """Compute responses and return as xarray Dataset.

    This function is compatible with joblib caching.
    """
    # Reconstruct the network
    checkpointed_network = network
    network = checkpointed_network.network

    # Initialize dataset
    dataset = dataset_class(**dataset_config)

    # Prepare lists to collect data
    stimuli_list = []
    responses_list = []

    # Compute responses
    for stimulus, responses in network.stimulus_response(
        dataset,
        dataset.dt,
        t_pre=t_pre,
        t_fade_in=t_fade_in,
        batch_size=batch_size,
    ):
        stimuli_list.append(stimulus)
        responses_list.append(responses)

    # Concatenate data
    stimuli_array = np.concatenate(stimuli_list, axis=0)
    responses_array = np.concatenate(responses_list, axis=0)
    return Result(stimuli_array, responses_array, dataset.arg_df.reset_index(drop=True))


def generic_responses(
    network_view_or_ensemble: Union["flyvision.NetworkView", "flyvision.EnsembleView"],
    dataset: Optional[Any],
    default_dataset_config: Dict,
    default_dataset_class: type,
    t_pre: float,
    t_fade_in: float,
    batch_size: int,
    cell_index: Optional[np.ndarray | str] = "central",
) -> xr.Dataset:
    """Return responses for a given dataset as an xarray Dataset."""
    # Handle both single and multiple NetworkViews
    if isinstance(network_view_or_ensemble, flyvision.network.NetworkView):
        network_views = [network_view_or_ensemble]
    else:
        network_views = list(network_view_or_ensemble.values())

    # Prepare dataset configuration
    if dataset is None:
        dataset_config = default_dataset_config
        dataset_class = default_dataset_class
    else:
        dataset_config = dataset.config.to_dict()
        dataset_class = type(dataset)

    # Prepare list to collect datasets
    results = []
    checkpoints = []

    def handle_network(idx, network_view, network):
        if idx == 0:
            # Initialize network
            checkpointed_network = network_view.network(checkpoint="best")
            checkpointed_network.init()
        else:
            # Pass initialized network over to next network view to avoid
            # reinitializing the network
            checkpointed_network = network_view.network(
                checkpoint="best", network=network
            )

        # use the memory cache from this network_view
        joblib_cache = network_view.memory.cache

        # Call the cached compute_responses function
        # This is cached, but the checkpointed_network is not pickled and hashed
        # because it implements it's own has
        results.append(
            joblib_cache(compute_responses)(
                checkpointed_network,
                dataset_class,
                dataset_config,
                batch_size,
                t_pre,
                t_fade_in,
            )  # type: Result
        )
        checkpoints.append(checkpointed_network.checkpoint)
        return checkpointed_network.network

    network = handle_network(0, network_views[0], None)

    for idx, network_view in enumerate(network_views[1:], 1):
        handle_network(idx, network_view, network)

    # Stack results
    result = Result(
        results[0].stimuli,
        np.stack([r.responses for r in results], axis=0),
        results[0].arg_df,
    )

    # Create xarray Dataset with time coordinate
    cell_types = network_views[0].connectome.dir.nodes.type[:].astype(str)
    u_coords = network_views[0].connectome.dir.nodes.u[:]
    v_coords = network_views[0].connectome.dir.nodes.v[:]
    if cell_index is not None:
        # Prepare central cells index
        cell_index = network_views[0].connectome.dir.central_cells_index[:]
        result.responses = result.responses[:, :, :, cell_index]
        cell_types = cell_types[cell_index]
        u_coords = u_coords[cell_index]
        v_coords = v_coords[cell_index]

    # Assuming dataset.dt is available
    dt = dataset_config.get('dt')
    t_pre = dataset_config.get('t_pre', 0.0)
    n_frames = result.stimuli.shape[1]
    time = np.arange(n_frames).astype(float) * dt - t_pre

    # Create xarray Dataset with time coordinate
    responses = xr.Dataset(
        {
            'stimulus': (['sample', 'frame', 'channel', 'hex_pixel'], result.stimuli),
            'responses': (['network_id', 'sample', 'frame', 'neuron'], result.responses),
        },
        coords={
            'network_id': np.arange(len(network_views)),
            'sample': np.arange(result.stimuli.shape[0]),
            'frame': np.arange(n_frames),
            'time': ('frame', time),  # Add time coordinate
            'channel': np.arange(result.stimuli.shape[2]),
            'hex_pixel': np.arange(result.stimuli.shape[3]),
            'neuron': np.arange(result.responses.shape[3]),
            'cell_type': ('neuron', cell_types),
            'u': ('neuron', u_coords),
            'v': ('neuron', v_coords),
            **{
                col: ('sample', result.arg_df[col].values)
                for col in result.arg_df.columns
            },
            'checkpoints': ('network_id', checkpoints),
        },
        attrs={
            'config': dataset_config,
            'network_config': network_views[0].dir.config.network.to_dict(),
        },
    )

    return responses


# --------------------- Flash Responses ---------------------


def flash_responses(
    network_view_or_ensemble: Union["flyvision.NetworkView", "flyvision.EnsembleView"],
    dataset: Optional[Flashes] = None,
    radius=(-1, 6),
    dt=1 / 200,
    batch_size=1,
) -> xr.Dataset:
    default_dataset_config = {
        'dynamic_range': [0, 1],
        't_stim': 1,
        't_pre': 1.0,
        'dt': dt,
        'radius': radius,
        'alternations': (0, 1, 0),
    }
    return generic_responses(
        network_view_or_ensemble,
        dataset,
        default_dataset_config,
        Flashes,
        t_pre=1.0,
        t_fade_in=0.0,
        batch_size=batch_size,
    )


# --------------------- Moving Edge Responses ---------------------


def moving_edge_responses(
    network_view_or_ensemble: Union["flyvision.NetworkView", "flyvision.EnsembleView"],
    dataset: Optional[MovingEdge] = None,
    speeds=(2.4, 4.8, 9.7, 13, 19, 25),
    offsets=(-10, 11),
    dt=1 / 200,
    batch_size=1,
) -> xr.Dataset:
    default_dataset_config = {
        'offsets': offsets,
        'intensities': [0, 1],
        'speeds': speeds,
        'height': 80,
        'post_pad_mode': "continue",
        'dt': dt,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
        't_pre': 1.0,
        't_post': 1.0,
    }
    return generic_responses(
        network_view_or_ensemble,
        dataset,
        default_dataset_config,
        MovingEdge,
        t_pre=1.0,
        t_fade_in=0.0,
        batch_size=batch_size,
    )


# --------------------- Moving Bar Responses ---------------------


def moving_bar_responses(
    network_view_or_ensemble: Union["flyvision.NetworkView", "flyvision.EnsembleView"],
    dataset: Optional[MovingBar] = None,
    dt=1 / 200,
    batch_size=1,
) -> xr.Dataset:
    default_dataset_config = {
        'widths': [1, 2, 4],
        'offsets': (-10, 11),
        'intensities': [0, 1],
        'speeds': [2.4, 4.8, 9.7, 13, 19, 25],
        'height': 9,
        'post_pad_mode': "continue",
        'dt': dt,
        't_pre': 1.0,
        't_post': 1.0,
        'device': "cuda" if torch.cuda.is_available() else "cpu",
    }
    return generic_responses(
        network_view_or_ensemble,
        dataset,
        default_dataset_config,
        MovingBar,
        t_pre=1.0,
        t_fade_in=0.0,
        batch_size=batch_size,
    )


# --------------------- Naturalistic Stimuli Responses ---------------------


def naturalistic_stimuli_responses(
    network_view_or_ensemble: Union["flyvision.NetworkView", "flyvision.EnsembleView"],
    dataset: Optional[AugmentedSintel] = None,
    dt=1 / 100,
    batch_size=1,
) -> xr.Dataset:
    default_dataset_config = {
        'tasks': ["flow"],
        'interpolate': False,
        'boxfilter': {'extent': 15, 'kernel_size': 13},
        'temporal_split': True,
        'dt': dt,
    }
    return generic_responses(
        network_view_or_ensemble,
        dataset,
        default_dataset_config,
        AugmentedSintel,
        t_pre=0.0,
        t_fade_in=2.0,
        batch_size=batch_size,
    )


# --------------------- Central Impulses Responses ---------------------


def central_impulses_responses(
    network_view_or_ensemble: Union["flyvision.NetworkView", "flyvision.EnsembleView"],
    dataset: Optional[CentralImpulses] = None,
    intensity=1,
    bg_intensity=0.5,
    impulse_durations=(5e-3, 20e-3, 50e-3, 100e-3, 200e-3, 300e-3),
    dt=1 / 200,
    batch_size=1,
) -> xr.Dataset:
    default_dataset_config = {
        'impulse_durations': impulse_durations,
        'dot_column_radius': 0,
        'bg_intensity': bg_intensity,
        't_stim': 2,
        'dt': dt,
        'n_ommatidia': 721,
        't_pre': 1.0,
        't_post': 0,
        'intensity': intensity,
        'mode': "impulse",
        'device': "cuda" if torch.cuda.is_available() else "cpu",
    }
    return generic_responses(
        network_view_or_ensemble,
        dataset,
        default_dataset_config,
        CentralImpulses,
        t_pre=4.0,
        t_fade_in=0.0,
        batch_size=batch_size,
    )


# --------------------- Spatial Impulses Responses ---------------------


def spatial_impulses_responses(
    network_view_or_ensemble: Union["flyvision.NetworkView", "flyvision.EnsembleView"],
    dataset: Optional[SpatialImpulses] = None,
    intensity=1,
    bg_intensity=0.5,
    impulse_durations=(5e-3, 20e-3),
    max_extent=4,
    dt=1 / 200,
    batch_size=1,
) -> xr.Dataset:
    default_dataset_config = {
        'impulse_durations': impulse_durations,
        'max_extent': max_extent,
        'dot_column_radius': 0,
        'bg_intensity': bg_intensity,
        't_stim': 2,
        'dt': dt,
        'n_ommatidia': 721,
        't_pre': 1.0,
        't_post': 0,
        'intensity': intensity,
        'mode': "impulse",
        'device': "cuda" if torch.cuda.is_available() else "cpu",
    }
    return generic_responses(
        network_view_or_ensemble,
        dataset,
        default_dataset_config,
        SpatialImpulses,
        t_pre=4.0,
        t_fade_in=0.0,
        batch_size=batch_size,
    )


# --------------------- Optimal Stimulus Responses ---------------------


def _compute_optimal_stimulus_responses(
    network_config: Dict,
    checkpoint_path: str,
    dataset_config: Dict,
) -> xr.Dataset:
    """Compute optimal stimuli responses and return as xarray Dataset.

    This function is compatible with joblib caching.
    """
    # Reconstruct the network
    network = flyvision.Network(**network_config)
    flyvision.recover_network(network, checkpoint_path)
    network.eval()

    # Create a dummy NetworkView for FindOptimalStimuli
    network_view = flyvision.NetworkView(network_dir=None)
    network_view.network = network
    network_view.init_network()

    if dataset_config == "default":
        findoptstim = optimal_stimuli.FindOptimalStimuli(network_view, stimuli="default")
    else:
        stimuli_dataset = StimulusDataset(**dataset_config)
        findoptstim = optimal_stimuli.FindOptimalStimuli(
            network_view, stimuli=stimuli_dataset
        )

    # Prepare lists to collect data
    data_list = []

    for cell_type in network_view.cell_types_sorted:
        optstim = findoptstim.regularized_optimal_stimuli(cell_type)
        data_list.append({
            'cell_type': cell_type,
            'stimulus': optstim.stimulus.stimulus.cpu().numpy(),
            'response': optstim.stimulus.response.cpu().numpy(),
            'regularized_stimulus': optstim.regularized_stimulus,
            'regularized_response': optstim.response,
            'central_predicted_activity': optstim.central_predicted_response,
            'central_target_activity': optstim.central_target_response,
            'losses': optstim.losses,
        })

    # Convert list of dicts to xarray Dataset
    ds = xr.Dataset.from_dict({'data': data_list})

    # Add attrs
    ds.attrs.update({
        'network_config': network_config,
        'checkpoint_path': checkpoint_path,
    })

    return ds


def optimal_stimulus_responses(
    network_view: "flyvision.NetworkView",
    dataset: Optional[StimulusDataset] = None,
) -> xr.Dataset:
    """Return optimal stimuli responses as xarray Dataset."""
    network_view.init_network()

    # Extract network configuration and checkpoint path
    network_config = network_view.dir.config.network.to_dict()
    checkpoint_path = str(network_view.checkpoints.path)

    # Prepare dataset configuration
    dataset_config = "default" if dataset is None else dataset.config.to_dict()

    # Call the cached helper function
    ds = network_view.memory.cache(_compute_optimal_stimulus_responses)(
        network_config,
        checkpoint_path,
        dataset_config,
    )
    return ds


if __name__ == '__main__':
    # Example usage
    import time

    # nv = flyvision.NetworkView("flow/0000/000")
    # start = time.time()
    # x = flash_responses(nv)
    # print(x)
    # print(f"Elapsed time: {time.time() - start:.2f} seconds")

    ensemble = flyvision.EnsembleView("flow/0000")  # ["flow/0000/000", "flow/0000/001"])
    start = time.time()
    x = flash_responses(ensemble)
    print(x)
    print(f"Elapsed time: {time.time() - start:.2f} seconds")
