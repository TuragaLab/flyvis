"""To store stimulus responses. Defaults from paper are called main."""

import logging

from datamate import Namespace

from flyvision import NetworkView
from flyvision.analysis.optimal_stimuli import FindOptimalStimuli
from flyvision.datasets.dots import CentralImpulses, SpatialImpulses
from flyvision.datasets.flashes import Flashes
from flyvision.datasets.moving_bar import MovingBar, MovingEdge
from flyvision.datasets.sintel import AugmentedSintel
from flyvision.utils.activity_utils import CellTypeArray

logging = logging.getLogger(__name__)

# TODO: make generators and create facade in network view


def flash_responses_main(
    network_view: NetworkView, subdir="flash_responses", radius=[-1, 6]
):
    """Store flash responses."""

    dt = 1 / 200

    flashes = Namespace(
        dynamic_range=[0, 1],
        t_stim=1,
        t_pre=1.0,
        dt=dt,
        radius=radius,
        alternations=(0, 1, 0),
    )
    dataset = Flashes(**flashes)
    for _, responses in network_view.network.stimulus_response(
        dataset, dt, t_pre=1.0, t_fade_in=0.0
    ):
        extend_stored_activity(network_view, responses, subdir)

    # store config to be able to reproduce the dataset
    network_view.dir[subdir].config = dataset.config

    logging.info("Stored flash responses.")


def movingedge_responses_main(
    network_view: NetworkView,
    subdir="movingedge_responses",
    speeds=[2.4, 4.8, 9.7, 13, 19, 25],
    offsets=(-10, 11),
):
    """Store moving edge responses."""

    dt = 1 / 200

    dataset = MovingEdge(
        offsets=offsets,  # in 1 * radians(2.25)
        intensities=[0, 1],
        # in 1 * radians(5.8) / s
        speeds=speeds,
        height=80,  # in 1 * radians(2.25)
        # matters for visualization but not for
        # calculation of indices because not taken into ccount here
        post_pad_mode="continue",
        dt=dt,
        device="cuda",
        t_pre=1.0,
        t_post=1.0,
    )

    for _, responses in network_view.network.stimulus_response(
        dataset, dt, t_pre=1.0, t_fade_in=0.0
    ):
        extend_stored_activity(network_view, responses, subdir)

    network_view.dir[subdir].config = dataset.config

    logging.info("Stored moving edge response.")


def movingbar_responses_main(network_view: NetworkView, subdir="movingbar"):
    dt = 1 / 200
    ### ---------------------------- moving bars si ---------------------------#
    dataset = MovingBar(
        widths=[1, 2, 4],  # in 1 * radians(2.25)
        offsets=(-10, 11),  # in 1 * radians(2.25)
        intensities=[0, 1],
        # in 1 * radians(5.8) / s
        speeds=[2.4, 4.8, 9.7, 13, 19, 25],
        height=9,  # in 1 * radians(2.25)
        post_pad_mode="continue",
        dt=dt,
        t_pre=1.0,
        t_post=1.0,
        device="cuda",
    )
    for _, responses in network_view.network.stimulus_response(
        dataset, dt, t_pre=1.0, t_fade_in=0.0
    ):
        extend_stored_activity(network_view, responses, subdir)

    network_view.dir[subdir].config = dataset.config

    logging.info("Stored moving bar response.")


def naturalistic_stimuli_responses_main(
    network_view: NetworkView, subdir="naturalistic_stimuli_responses"
):
    dt = 1 / 100

    config = Namespace(
        tasks=["flow"],
        interpolate=False,
        boxfilter=dict(extent=15, kernel_size=13),
        temporal_split=True,
        dt=dt,
    )

    dataset = AugmentedSintel(**config)
    for _, responses in network_view.network.stimulus_response(
        dataset, dt, t_pre=0.0, t_fade_in=2.0
    ):
        extend_stored_activity(network_view, responses, subdir)

    network_view.dir[subdir].config = dataset.config

    logging.info("Stored naturalistic stimuli response.")


def central_impulses_responses_main(
    network_view: NetworkView,
    dt=1 / 200,
    intensity=1,
    bg_intensity=0.5,
    impulse_durations=[5e-3, 20e-3, 50e-3, 100e-3, 200e-3, 300e-3],
    subdir="central_impulses_responses",
):
    """Central ommatidium impulses."""
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
    for _, responses in network_view.network.stimulus_response(
        dataset, dt, t_pre=4.0, t_fade_in=0.0
    ):
        extend_stored_activity(network_view, responses, subdir)

    network_view.dir[subdir].config = dataset.config

    logging.info("Stored central ommatidium flash response.")


def spatial_impulses_responses_main(
    network_view: NetworkView,
    dt=1 / 200,
    intensity=1,
    bg_intensity=0.5,
    impulse_durations=[5e-3, 20e-3],
    max_extent=4,
    subdir="spatial_impulses_responses",
):
    """Single ommatidium impulses across the eye."""

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
    for _, responses in network_view.network.stimulus_response(
        dataset, dt, t_pre=4.0, t_fade_in=0.0
    ):
        extend_stored_activity(network_view, responses, subdir)

    network_view.dir[subdir].config = dataset.config

    logging.info("Stored spatial ommatidium flash response.")


def optimal_stimulus_responses_main(
    network_view: NetworkView,
    subdir="naturalistic_stimuli_responses",
):
    findoptstim = FindOptimalStimuli(network_view)

    subdir_optstim = network_view.dir[subdir].optstims
    subdir_regularized_optstim = network_view.dir[subdir].regularized_optstims

    responses = network_view.dir[subdir].network_states.nodes.activity_central[:]
    responses = CellTypeArray(responses, network_view.connectome)

    for cell_type in network_view.cell_types_sorted:
        optstim = findoptstim.regularized_optimal_stimuli(cell_type, responses=responses)

        subdir_optstim[cell_type].stimulus = optstim.stimulus.stimulus.cpu().numpy()
        subdir_optstim[cell_type].response = optstim.stimulus.response.cpu().numpy()
        subdir_regularized_optstim[cell_type].stimulus = optstim.regularized_stimulus
        subdir_regularized_optstim[cell_type].response = optstim.response
        subdir_regularized_optstim[
            cell_type
        ].central_predicted_activity = optstim.central_predicted_response
        subdir_regularized_optstim[
            cell_type
        ].central_target_activity = optstim.central_target_response
        subdir_regularized_optstim[cell_type].losses = optstim.losses
        logging.info(f"Stored maximally excitatory stimuli for - {cell_type}.")


def extend_stored_activity(
    network_view,
    activity,
    subdir,
    all_cells=False,
    file_prefix="activity",
):
    """
    activity : n_samples, n_frames, n_neurons
    """
    central_cells_index = network_view.network.connectome.central_cells_index[:]

    if not all_cells:
        file_prefix += "_central"
        activity = activity[:, :, central_cells_index]

    network_view.dir[subdir].network_states.nodes.extend(
        file_prefix, [activity[:].squeeze()]
    )
