"""Script for precomputing synthetic recordings for a single network.

Example Usage:
--------------
1. Generate all default synthetic recordings for a specific network:
   ```bash
   python synthetic_recordings_single.py task_name=flow ensemble_and_network_id=0000/000
   ```

2. Generate only spatial and central impulse responses for a different network:
   ```bash
   python synthetic_recordings_single.py task_name=flow ensemble_and_network_id=9998/000 \
   --functions spatial_impulses_responses central_impulses_responses
   ```

3. Generate default recordings with a custom batch size and delete existing recordings:
   ```bash
   python synthetic_recordings_single.py task_name=flow ensemble_and_network_id=0000/000 \
   --batch_size 16 --delete_recordings
   ```

Available Functions:
--------------------
- flash_responses
- moving_edge_responses
- moving_edge_responses_currents
- moving_bar_responses
- naturalistic_stimuli_responses
- spatial_impulses_responses
- central_impulses_responses
"""

# pyright: reportCallIssue=false

import argparse
import logging

from flyvision import NetworkView
from flyvision.utils.config_utils import HybridArgumentParser

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)
logging = logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = HybridArgumentParser(
        hybrid_args={
            "task_name": {"required": True},
            "ensemble_and_network_id": {"required": True},
        },
        description="Record synthetic responses.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    parser.epilog = __doc__
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
