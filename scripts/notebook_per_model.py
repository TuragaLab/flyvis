"""Script to run a notebook for each model of an ensemble."""

import argparse
import logging
from typing import List

from flyvision import script_dir
from flyvision.utils.config_utils import HybridArgumentParser
from flyvision.utils.lsf_utils import launch_range

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def run_notebook_ensemble(args: argparse.Namespace, kwargs: List[str]) -> None:
    """
    Launch notebook jobs for an ensemble of models.

    Args:
        args: Command-line arguments.
        kwargs: Additional keyword arguments as a list of strings.
    """
    launch_range(
        args.start,
        args.end,
        args.ensemble_id,
        args.task_name,
        args.nP,
        args.gpu,
        args.q,
        args.notebook_script,
        args.dry,
        kwargs,
    )


if __name__ == "__main__":
    parser = HybridArgumentParser(
        hybrid_args=[],
        description="Run a notebook for each model of an ensemble on the cluster.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start id of ensemble.")
    parser.add_argument("--end", type=int, default=50, help="End id of ensemble.")
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
        "--notebook_script",
        type=str,
        default=f"{str(script_dir)}/run_notebook.py",
        help="Script to run for executing notebooks.",
    )
    parser.add_argument(
        "--notebook_path",
        type=str,
        required=True,
        help="Path of the notebook to execute, e.g. /path/to/__main__.ipynb.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path for the output notebook.",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Perform a dry run without actually launching jobs.",
    )

    args, kwargs = parser.parse_known_intermixed_args()
    run_notebook_ensemble(args, kwargs)
