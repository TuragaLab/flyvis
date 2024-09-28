"""Script to store synthetic recordings in the way the paper does it.

Example usage:
python synthetic_recordings_single.py task_name=flow ensemble_and_network_id=0000/000
python synthetic_recordings_single.py task_name=flow ensemble_and_network_id=9998/000
--functions spatial_impulses_responses central_impulses_responses
"""

import logging

from flyvision import NetworkView
from flyvision.utils.config_utils import HybridArgumentParser

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)
logging = logger = logging.getLogger(__name__)


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
        default="epe",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--delete_recordings", action="store_true")
    default_functions = [
        "flash_responses",
        "movingedge_responses",
        "movingbar_responses",
        "naturalistic_stimuli_responses",
        # "optimal_stimulus_responses",
        "spatial_impulses_responses",
        "central_impulses_responses",
    ]

    parser.add_argument(
        "--functions",
        nargs="+",
        help="List of functions to run.",
        default=default_functions,
    )
    args = parser.parse_with_hybrid_args()

    network_name = f"{args.task_name}/{args.ensemble_and_network_id}"
    network_view = NetworkView(
        network_name,
        best_checkpoint_fn_kwargs={
            "validation_subdir": args.validation_subdir,
            "loss_file_name": args.loss_file_name,
        },
    )
    if args.delete_recordings:
        network_view._clear_memory()
    network_view.init_network()

    if "flash_responses" in args.functions:
        network_view.flash_responses(batch_size=args.batch_size)
        logging.info("Stored flash responses.")

    if "movingedge_responses" in args.functions:
        network_view.movingedge_responses(batch_size=args.batch_size)
        logging.info("Stored moving edge responses.")

    if "movingbar_responses" in args.functions:
        network_view.movingbar_responses(batch_size=args.batch_size)
        logging.info("Stored moving bar responses.")

    if "naturalistic_stimuli_responses" in args.functions:
        network_view.naturalistic_stimuli_responses(batch_size=args.batch_size)
        logging.info("Stored naturalistic stimuli responses.")

    # TODO: this implementation is currently inefficient as it reloads the cache
    # for each cell type, but it's also uneccesary to store all of them
    # these can be computed at runtime relatively eaily for single networks
    # if "optimal_stimulus_responses" in args.functions:
    #     for cell_type in network_view.connectome_view.cell_types_sorted:
    #         network_view.optimal_stimulus_responses(cell_type=cell_type)
    #     logging.info("Stored maximally excitatory stimuli.")

    if "spatial_impulses_responses" in args.functions:
        network_view.spatial_impulses_responses(batch_size=args.batch_size)
        logging.info("Stored spatial impulses responses.")

    if "central_impulses_responses" in args.functions:
        network_view.central_impulses_responses(batch_size=args.batch_size)
        logging.info("Stored central impulses responses.")
