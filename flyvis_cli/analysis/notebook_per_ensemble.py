"""Script to run a jupyter notebook for an ensemble."""

import argparse
import logging
import sys
from typing import List

from flyvis import notebook_template, script_dir
from flyvis.utils.compute_cloud_utils import launch_single

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def run_ensemble_notebook(args: argparse.Namespace, kwargs: List[str]) -> None:
    """
    Launch ensemble notebook job.

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
        f"{str(script_dir)}/analysis/notebook.py",
        args.dry,
        ["--notebook_path", args.notebook_path] + ["per_ensemble:bool=true"] + kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ensemble notebook on the compute cloud.",
        usage=(
            "\nflyvis notebook-per-ensemble [-h] [...] --ensemble_id ENSEMBLE_ID "
            "--task_name TASK_NAME --notebook_path PATH\n"
            "       or\n"
            "%(prog)s [-h] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME "
            "--notebook_path PATH\n"
        ),
    )
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
        "--notebook_path",
        type=str,
        default=str(notebook_template),
        help="Path of the notebook to execute.",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Perform a dry run without actually launching jobs.",
    )

    args, kwargs = parser.parse_known_intermixed_args()
    run_ensemble_notebook(args, sys.argv[1:])
