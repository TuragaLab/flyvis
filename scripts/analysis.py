"""Script to run ensemble analysis."""

import argparse
import logging
from typing import List

from flyvision import script_dir
from flyvision.utils.compute_cloud_utils import launch_single

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def run_ensemble_analysis(args: argparse.Namespace, kwargs: List[str]) -> None:
    """
    Launch ensemble analysis job.

    Args:
        args: Command-line arguments.
        kwargs: Additional keyword arguments as a list of strings.
    """
    launch_single(
        args.ensemble_id,
        args.task_name,
        args.nP,
        args.gpu,
        args.q,
        args.ensemble_analysis_script,
        args.dry,
        kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ensemble analysis on the cluster.")
    parser.add_argument("--nP", type=int, default=4, help="Number of processors.")
    parser.add_argument("--gpu", type=str, default="num=1", help="Number of GPUs.")
    parser.add_argument("--q", type=str, default="gpu_l4", help="Queue.")
    parser.add_argument(
        "--ensemble_id",
        type=int,
        required=True,
        help="Id of the ensemble, e.g. 0045.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="Name given to the task, e.g., flow.",
    )
    parser.add_argument(
        "--ensemble_analysis_script",
        type=str,
        default=f"{str(script_dir)}/analysis/ensemble_analysis.py",
        help="Script to run for ensemble analysis.",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Perform a dry run without actually launching jobs.",
    )

    args, kwargs = parser.parse_known_intermixed_args()
    run_ensemble_analysis(args, kwargs)
