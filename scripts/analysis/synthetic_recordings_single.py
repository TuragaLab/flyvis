"""Script to store synthetic recordings in the way the paper does it.

Example usage:
python synthetic_recordings_single.py task_name=flow ensemble_and_network_id=0000/000
python synthetic_recordings_single.py task_name=flow ensemble_and_network_id=9998/000
--functions spatial_impulses_responses central_impulses_responses
"""

import logging
import shutil

from flyvision import NetworkView
from flyvision.utils.config_utils import HybridArgumentParser

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)
logging = logger = logging.getLogger(__name__)


def delete_if_exists(network_view, subdir, delete_if_exists=True):
    path = network_view.dir[subdir].path
    if path.exists() and delete_if_exists:
        shutil.rmtree(path)


def extend_stored_activity(
    network_view,
    activity,
    subdir,
    config,
    central=True,
    filename="activity_central",
    batch_size=1,
):
    """Stores activity in the directory of the network.

    Extends along first axis when called multiple times with updated activity tensors.

    Activity must be (n_samples, n_frames, n_neurons).
    """

    if central:
        filename.replace("_central", "")
        central_cells_index = network_view.network.connectome.central_cells_index[:]
        activity = activity[:, :, central_cells_index]

    # store activity, this stacks the activity along the first axis when called multiple
    # times which is a datamate feature.
    if batch_size == 1:
        network_view.dir[subdir].network_states.nodes.extend(
            filename, [activity[:].squeeze()]
        )
    else:
        network_view.dir[subdir].network_states.nodes.extend(
            filename, activity[:].squeeze()
        )

    # store config
    network_view.dir[subdir].config = config


def store_optimal_stimulus_results(
    network_view,
    optstim,
    subdir,
):
    subdir_optstim = network_view.dir[subdir].optstims
    subdir_regularized_optstim = network_view.dir[subdir].regularized_optstims

    subdir_optstim[optstim["cell_type"]].stimulus = optstim["stimulus"]
    subdir_optstim[optstim["cell_type"]].response = optstim["response"]
    subdir_regularized_optstim[optstim["cell_type"]].stimulus = optstim[
        "regularized_stimulus"
    ]
    subdir_regularized_optstim[optstim["cell_type"]].response = optstim[
        "regularized_response"
    ]
    subdir_regularized_optstim[optstim["cell_type"]].central_predicted_activity = optstim[
        "central_predicted_activity"
    ]
    subdir_regularized_optstim[optstim["cell_type"]].central_target_activity = optstim[
        "central_target_activity"
    ]
    subdir_regularized_optstim[optstim["cell_type"]].losses = optstim["losses"]


if __name__ == "__main__":
    parser = HybridArgumentParser(
        hybrid_args=["task_name", "ensemble_and_network_id"],
        description="Record synthetic responses.",
    )
    parser.add_argument(
        "--chkpt", type=str, default="best", help="checkpoint to evaluate."
    )
    parser.add_argument(
        "--validation_subdir",
        type=str,
        default="validation",
    )
    parser.add_argument(
        "--loss_file_name",
        type=str,
        default="EPE",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--delete_recordings", action="store_true")
    default_functions = [
        "flash_responses",
        "movingedge_responses",
        "movingbar_responses",
        "naturalistic_stimuli_responses",
        "optimal_stimulus_responses",
        "spatial_impulses_responses",
        "central_impulses_responses",
    ]

    parser.add_argument(
        "--functions",
        nargs="+",
        help="List of functions to run.",
        default=default_functions,
        choices=default_functions,
    )
    args = parser.parse_args()

    network_name = f"{args.task_name}/{args.ensemble_and_network_id}"
    network_view = NetworkView(
        network_name,
        checkpoint=args.chkpt,
        validation_subdir=args.validation_subdir,
        loss_file_name=args.loss_file_name,
    )
    network_view.init_network()

    if "flash_responses" in args.functions:
        subdir = f"flash_responses/{network_view.checkpoints.current_chkpt_key}"
        delete_if_exists(network_view, subdir, args.delete_recordings)
        for result in network_view.flash_responses(batch_size=args.batch_size):
            extend_stored_activity(
                network_view,
                result.responses,
                subdir=subdir,
                config=result.config,
                batch_size=args.batch_size,
            )
        logging.info("Stored flash responses.")

    if "movingedge_responses" in args.functions:
        subdir = f"movingedge_responses/{network_view.checkpoints.current_chkpt_key}"
        delete_if_exists(network_view, subdir, args.delete_recordings)
        for result in network_view.movingedge_responses(batch_size=args.batch_size):
            extend_stored_activity(
                network_view,
                result.responses,
                subdir=subdir,
                config=result.config,
                batch_size=args.batch_size,
            )
        logging.info("Stored moving edge responses.")

    if "movingbar_responses" in args.functions:
        subdir = f"movingbar_responses/{network_view.checkpoints.current_chkpt_key}"
        delete_if_exists(network_view, subdir, args.delete_recordings)
        for result in network_view.movingbar_responses(batch_size=args.batch_size):
            extend_stored_activity(
                network_view,
                result.responses,
                subdir=subdir,
                config=result.config,
                batch_size=args.batch_size,
            )
        logging.info("Stored moving bar responses.")

    if "naturalistic_stimuli_responses" in args.functions:
        subdir = (
            "naturalistic_stimuli_responses/"
            f"{network_view.checkpoints.current_chkpt_key}"
        )
        delete_if_exists(network_view, subdir, args.delete_recordings)
        for result in network_view.naturalistic_stimuli_responses(
            batch_size=args.batch_size
        ):
            extend_stored_activity(
                network_view,
                result.responses,
                subdir=subdir,
                config=result.config,
                batch_size=args.batch_size,
            )
        logging.info("Stored naturalistic stimuli responses.")

        if "optimal_stimulus_responses" in args.functions:
            delete_if_exists(network_view, f"{subdir}/optstims", args.delete_recordings)
            delete_if_exists(
                network_view, f"{subdir}/regularized_optstims", args.delete_recordings
            )
            for optstim in network_view.optimal_stimulus_responses():
                store_optimal_stimulus_results(
                    network_view,
                    optstim,
                    subdir,
                )
            logging.info("Stored maximally excitatory stimuli.")

    if "spatial_impulses_responses" in args.functions:
        subdir = (
            f"spatial_impulses_responses/{network_view.checkpoints.current_chkpt_key}"
        )
        delete_if_exists(network_view, subdir, args.delete_recordings)
        for result in network_view.spatial_impulses_responses(batch_size=args.batch_size):
            extend_stored_activity(
                network_view,
                result.responses,
                subdir=subdir,
                config=result.config,
                batch_size=args.batch_size,
            )
        logging.info("Stored spatial impulses responses.")

    if "central_impulses_responses" in args.functions:
        subdir = (
            f"central_impulses_responses/{network_view.checkpoints.current_chkpt_key}"
        )
        delete_if_exists(network_view, subdir, args.delete_recordings)
        for result in network_view.central_impulses_responses(batch_size=args.batch_size):
            extend_stored_activity(
                network_view,
                result.responses,
                subdir=subdir,
                config=result.config,
                batch_size=args.batch_size,
            )
        logging.info("Stored central impulses responses.")
