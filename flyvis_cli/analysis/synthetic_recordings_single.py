"""Script for precomputing synthetic recordings for a single network."""

# pyright: reportCallIssue=false

import argparse
import logging

from flyvis import NetworkView
from flyvis.utils.config_utils import HybridArgumentParser

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)
logging = logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = HybridArgumentParser(
        hybrid_args={
            "task_name": {
                "required": True,
                "help": "Name of the task (e.g., 'flow', 'depth')",
            },
            "ensemble_and_network_id": {
                "required": True,
                "help": "ID in the format XXXX/YYY (ensemble/network)",
            },
        },
        description=(
            "This script generates and stores various types of synthetic responses for a "
            "given network, such as flash responses, moving edge responses, and impulse "
            "responses. The responses are automatically cached for later use in analysis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage=(
            "\nflyvis synthetic-recordings [-h] task_name=TASK "
            "ensemble_and_network_id=XXXX/YYY [options]\n"
            "       or\n"
            "%(prog)s [-h] task_name=TASK ensemble_and_network_id=XXXX/YYY [options]\n"
        ),
        epilog="""
Examples:
--------
1. Generate all default synthetic recordings:
   flyvis synthetic-recordings task_name=flow ensemble_and_network_id=0000/000

2. Generate only specific response types:
   flyvis synthetic-recordings task_name=flow ensemble_and_network_id=0000/000 \\
       --functions spatial_impulses_responses central_impulses_responses

3. Generate with custom batch size and clear existing recordings:
   flyvis synthetic-recordings task_name=flow ensemble_and_network_id=0000/000 \\
       --batch_size 16 --delete_recordings
""",
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
        "moving_edge_responses",
        "moving_edge_responses_currents",
        "moving_bar_responses",
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
    # TODO: add logic to pass different checkpoint selection methods
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

    if "moving_edge_responses" in args.functions:
        network_view.moving_edge_responses(batch_size=args.batch_size)
        logging.info("Stored moving edge responses.")

    if "moving_edge_responses_currents" in args.functions:
        network_view.moving_edge_currents(
            target_cell_types=[
                "T4a",
                "T4b",
                "T4c",
                "T4d",
                "T5a",
                "T5b",
                "T5c",
                "T5d",
                "TmY3",
            ]
        )
        logging.info("Stored moving edge currents.")

    if "moving_bar_responses" in args.functions:
        network_view.moving_bar_responses(batch_size=args.batch_size)
        logging.info("Stored moving bar responses.")

    if "naturalistic_stimuli_responses" in args.functions:
        network_view.naturalistic_stimuli_responses(batch_size=args.batch_size)
        logging.info("Stored naturalistic stimuli responses.")

    # TODO: this implementation is currently inefficient as it reloads the cache
    # for each cell type, but it's also uneccesary to store all of them because
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
