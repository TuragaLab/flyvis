"""Script to store synthetic recordings in the way the paper does it.

Example usage:
python synthetic_recordings_single.py task_name=flow ensemble_and_network_id=0000/000
--functions spatial_impulses_responses_main central_impulses_responses_main
"""

from flyvision import NetworkView
from flyvision.analysis.stimulus_responses import (
    central_impulses_responses_main,
    flash_responses_main,
    movingbar_responses_main,
    movingedge_responses_main,
    naturalistic_stimuli_responses_main,
    optimal_stimulus_responses_main,
    spatial_impulses_responses_main,
)
from flyvision.utils.config_utils import HybridArgumentParser

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
    default_functions = [
        "flash_responses_main",
        "movingedge_responses_main",
        "movingbar_responses_main",
        "naturalistic_stimuli_responses_main",
        "optimal_stimulus_responses_main",
        "spatial_impulses_responses_main",
        "central_impulses_responses_main",
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

    if args.chkpt == "best":
        subdirchild = f"chkpt_best_{args.validation_subdir}_{args.loss_file_name}"
    else:
        subdirchild = f"chkpt_{args.chkpt}"

    if "flash_responses_main" in args.functions:
        flash_responses_main(
            network_view,
            subdir=f"flash_responses/{subdirchild}",
            radius=[-1, 6],
        )

    if "movingedge_responses_main" in args.functions:
        movingedge_responses_main(
            network_view,
            subdir=f"movingedge_responses/{subdirchild}",
        )

    if "movingbar_responses_main" in args.functions:
        movingbar_responses_main(
            network_view,
            subdir=f"movingbar_responses/{subdirchild}",
        )

    if "naturalistic_stimuli_responses_main" in args.functions:
        naturalistic_stimuli_responses_main(
            network_view,
            subdir=f"naturalistic_stimuli_responses/{subdirchild}",
        )

        if "optimal_stimulus_responses_main" in args.functions:
            optimal_stimulus_responses_main(
                network_view,
                subdir=f"naturalistic_stimuli_responses/{subdirchild}",
            )

    if "spatial_impulses_responses_main" in args.functions:
        spatial_impulses_responses_main(
            network_view,
            subdir=f"spatial_impulses_responses/{subdirchild}",
        )

    if "central_impulses_responses_main" in args.functions:
        central_impulses_responses_main(
            network_view,
            subdir=f"central_impulses_responses/{subdirchild}",
        )
