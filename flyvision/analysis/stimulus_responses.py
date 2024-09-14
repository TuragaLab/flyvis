"""Stimulus response functions with caching using xarray."""

from __future__ import annotations

from typing import Dict, Optional

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


# @dataclass
# class CheckpointedNetwork:
#     network: Any
#     config: dict
#     checkpoint: Path | str

#     def init(self, eval: bool = True):
#         network = self.network(**self.config)
#         recover_network(network, self.checkpoint)
#         if eval:
#             network.eval()
#         return network

compute_responses = None
# def compute_responses(
#     network: CheckpointedNetwork,
#     dataset_class: type,
#     dataset_config: Dict,
#     batch_size: int,
#     t_pre: float,
#     t_fade_in: float,
# ) -> xr.Dataset:
#     """Compute responses and return as xarray Dataset.

#     This function is compatible with joblib caching.
#     """
#     # Reconstruct the network
#     network = network.init()
#     # Initialize dataset
#     dataset = dataset_class(**dataset_config)

#     # Prepare lists to collect data
#     stimuli_list = []
#     responses_list = []

#     # Compute responses
#     for stimulus, responses in network.stimulus_response(
#         dataset,
#         dataset.dt,
#         t_pre=t_pre,
#         t_fade_in=t_fade_in,
#         batch_size=batch_size,
#     ):
#         stimuli_list.append(stimulus)
#         responses_list.append(responses)

#     # Concatenate data
#     stimuli_array = np.concatenate(stimuli_list, axis=0)
#     responses_array = np.concatenate(responses_list, axis=0)
#     stim_args_df = dataset.arg_df.reset_index(drop=True)

#     # Assuming dataset.dt is available
#     dt = dataset.dt
#     n_frames = stimuli_array.shape[1]
#     time = np.arange(n_frames) * dt - dataset.t_pre

#     # Create xarray Dataset with time coordinate
#     ds = xr.Dataset(
#         {
#             'stimulus': (['sample', 'frame', 'channel', 'hex_pixel'], stimuli_array),
#             'responses': (['sample', 'frame', 'neuron'], responses_array),
#         },
#         coords={
#             'sample': np.arange(stimuli_array.shape[0]),
#             'frame': np.arange(n_frames),
#             'time': ('frame', time),  # Add time coordinate
#             'channel': np.arange(stimuli_array.shape[2]),
#             'hex_pixel': np.arange(stimuli_array.shape[3]),
#             'neuron': np.arange(responses_array.shape[2]),
#             'cell_type': ('neuron', network.connectome.nodes.type[:].astype(str)),
#             'u': ('neuron', network.connectome.nodes.u[:]),
#             'v': ('neuron', network.connectome.nodes.v[:]),
#             **{col: ('sample', stim_args_df[col].values)
# for col in stim_args_df.columns},
#         },
#         attrs={
#             'config': dataset.config.to_dict(),
#             'network_config': network.config,
#             'checkpoint_path': checkpoint_path,
#         },
#     )

#     # Add stim_args as coordinates
#     # ds = ds.assign_coords(**{
#     #     col: ('sample', stim_args_df[col].values) for col in stim_args_df.columns
#     # })
#     # ds.set_index(sample=['intensity', 'radius'], inplace=True)

#     return ds


# --------------------- Flash Responses ---------------------


def flash_responses(
    network_view: "flyvision.NetworkView",
    dataset: Optional[Flashes] = None,
    radius=(-1, 6),
    dt=1 / 200,
    batch_size=1,
) -> xr.Dataset:
    """Return flash responses as xarray Dataset."""
    network_view.init_network()

    # Extract network configuration and checkpoint path
    network_config = network_view.dir.config.network.to_dict()
    checkpoint_path = str(network_view.checkpoints.path)

    # Prepare dataset configuration
    if dataset is None:
        dataset_config = dict(
            dynamic_range=[0, 1],
            t_stim=1,
            t_pre=1.0,
            dt=dt,
            radius=radius,
            alternations=(0, 1, 0),
        )
        dataset_class = Flashes
    else:
        dataset_config = dataset.config.to_dict()
        dataset_class = type(dataset)

    t_pre = 1.0
    t_fade_in = 0.0

    # Call the cached helper function

    ds = network_view.memory.cache(compute_responses)(
        network_config,
        checkpoint_path,
        dataset_class,
        dataset_config,
        batch_size,
        t_pre,
        t_fade_in,
    )
    return ds


# --------------------- Moving Edge Responses ---------------------


def movingedge_responses(
    network_view: "flyvision.NetworkView",
    dataset: Optional[MovingEdge] = None,
    speeds=(2.4, 4.8, 9.7, 13, 19, 25),
    offsets=(-10, 11),
    dt=1 / 200,
    batch_size=1,
) -> xr.Dataset:
    """Return moving edge responses as xarray Dataset."""
    network_view.init_network()

    # Extract network configuration and checkpoint path
    network_config = network_view.dir.config.network.to_dict()
    checkpoint_path = str(network_view.checkpoints.path)

    # Prepare dataset configuration
    if dataset is None:
        dataset_config = dict(
            offsets=offsets,
            intensities=[0, 1],
            speeds=speeds,
            height=80,
            post_pad_mode="continue",
            dt=dt,
            device="cuda" if torch.cuda.is_available() else "cpu",
            t_pre=1.0,
            t_post=1.0,
        )
        dataset_class = MovingEdge
    else:
        dataset_config = dataset.config.to_dict()
        dataset_class = type(dataset)

    t_pre = 1.0
    t_fade_in = 0.0

    # Call the cached helper function
    ds = network_view.memory.cache(compute_responses)(
        network_config,
        checkpoint_path,
        dataset_class,
        dataset_config,
        batch_size,
        t_pre,
        t_fade_in,
    )
    return ds


# --------------------- Moving Bar Responses ---------------------


def movingbar_responses(
    network_view: "flyvision.NetworkView",
    dataset: Optional[MovingBar] = None,
    dt=1 / 200,
    batch_size=1,
) -> xr.Dataset:
    """Return moving bar responses as xarray Dataset."""
    network_view.init_network()

    # Extract network configuration and checkpoint path
    network_config = network_view.dir.config.network.to_dict()
    checkpoint_path = str(network_view.checkpoints.path)

    # Prepare dataset configuration
    if dataset is None:
        dataset_config = dict(
            widths=[1, 2, 4],
            offsets=(-10, 11),
            intensities=[0, 1],
            speeds=[2.4, 4.8, 9.7, 13, 19, 25],
            height=9,
            post_pad_mode="continue",
            dt=dt,
            t_pre=1.0,
            t_post=1.0,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        dataset_class = MovingBar
    else:
        dataset_config = dataset.config.to_dict()
        dataset_class = type(dataset)

    t_pre = 1.0
    t_fade_in = 0.0

    # Call the cached helper function
    ds = network_view.memory.cache(compute_responses)(
        network_config,
        checkpoint_path,
        dataset_class,
        dataset_config,
        batch_size,
        t_pre,
        t_fade_in,
    )
    return ds


# --------------------- Naturalistic Stimuli Responses ---------------------


def naturalistic_stimuli_responses(
    network_view: "flyvision.NetworkView",
    dataset: Optional[AugmentedSintel] = None,
    dt=1 / 100,
    batch_size=1,
) -> xr.Dataset:
    """Return naturalistic stimuli responses as xarray Dataset."""
    network_view.init_network()

    # Extract network configuration and checkpoint path
    network_config = network_view.dir.config.network.to_dict()
    checkpoint_path = str(network_view.checkpoints.path)

    # Prepare dataset configuration
    if dataset is None:
        dataset_config = dict(
            tasks=["flow"],
            interpolate=False,
            boxfilter=dict(extent=15, kernel_size=13),
            temporal_split=True,
            dt=dt,
        )
        dataset_class = AugmentedSintel
    else:
        dataset_config = dataset.config.to_dict()
        dataset_class = type(dataset)

    t_pre = 0.0
    t_fade_in = 2.0

    # Call the cached helper function
    ds = network_view.memory.cache(compute_responses)(
        network_config,
        checkpoint_path,
        dataset_class,
        dataset_config,
        batch_size,
        t_pre,
        t_fade_in,
    )
    return ds


# --------------------- Central Impulses Responses ---------------------


def central_impulses_responses(
    network_view: "flyvision.NetworkView",
    dataset: Optional[CentralImpulses] = None,
    intensity=1,
    bg_intensity=0.5,
    impulse_durations=(5e-3, 20e-3, 50e-3, 100e-3, 200e-3, 300e-3),
    dt=1 / 200,
    batch_size=1,
) -> xr.Dataset:
    """Return central ommatidium impulses responses as xarray Dataset."""
    network_view.init_network()

    # Extract network configuration and checkpoint path
    network_config = network_view.dir.config.network.to_dict()
    checkpoint_path = str(network_view.checkpoints.path)

    # Prepare dataset configuration
    if dataset is None:
        dataset_config = dict(
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
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        dataset_class = CentralImpulses
    else:
        dataset_config = dataset.config.to_dict()
        dataset_class = type(dataset)

    t_pre = 4.0
    t_fade_in = 0.0

    # Call the cached helper function
    ds = network_view.memory.cache(compute_responses)(
        network_config,
        checkpoint_path,
        dataset_class,
        dataset_config,
        batch_size,
        t_pre,
        t_fade_in,
    )
    return ds


# --------------------- Spatial Impulses Responses ---------------------


def spatial_impulses_responses(
    network_view: "flyvision.NetworkView",
    dataset: Optional[SpatialImpulses] = None,
    intensity=1,
    bg_intensity=0.5,
    impulse_durations=(5e-3, 20e-3),
    max_extent=4,
    dt=1 / 200,
    batch_size=1,
) -> xr.Dataset:
    """Return spatial ommatidium impulses responses as xarray Dataset."""
    network_view.init_network()

    # Extract network configuration and checkpoint path
    network_config = network_view.dir.config.network.to_dict()
    checkpoint_path = str(network_view.checkpoints.path)

    # Prepare dataset configuration
    if dataset is None:
        dataset_config = dict(
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
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        dataset_class = SpatialImpulses
    else:
        dataset_config = dataset.config.to_dict()
        dataset_class = type(dataset)

    t_pre = 4.0
    t_fade_in = 0.0

    # Call the cached helper function
    ds = network_view.memory.cache(compute_responses)(
        network_config,
        checkpoint_path,
        dataset_class,
        dataset_config,
        batch_size,
        t_pre,
        t_fade_in,
    )
    return ds


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

    network_view = flyvision.NetworkView("flow/0000/000")
    start = time.time()
    ds = flash_responses(network_view)
    print(ds)
    print(f"Elapsed time: {time.time() - start:.2f} seconds")
