"""Script to run a notebook for each model of an ensemble."""

import argparse
import logging
from importlib import resources
from typing import List

from flyvis.utils.compute_cloud_utils import launch_range
from flyvis.utils.config_utils import HybridArgumentParser

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

NOTEBOOK_SCRIPT_DIR = resources.files("flyvis_cli") / "analysis/notebook.py"


def run_notebook_ensemble(args: argparse.Namespace, kwargs: List[str]) -> None:
    """
    Launch notebook jobs for an ensemble of models.

    Args:
        args: Command-line arguments.
        kwargs: Additional keyword arguments as a list of strings.
    """
    # remove conflicting argument
    launch_range(
        args.start,
        args.end,
        args.ensemble_id,
        args.task_name,
        args.nP,
        args.gpu,
        args.q,
        str(NOTEBOOK_SCRIPT_DIR),
        args.dry,
        # add hybrid argument
        ["--notebook_per_model_path", args.notebook_per_model_path]
        + ["notebook_per_model:bool=true"]
        + kwargs,
    )


if __name__ == "__main__":
    parser = HybridArgumentParser(
        hybrid_args=[],
        description="Run a notebook for each model of an ensemble on the compute cloud.",
        usage=(
            "\nflyvis notebook-per-model [-h] [...] --ensemble_id ENSEMBLE_ID "
            "--task_name TASK_NAME --notebook_per_model_path PATH\n"
            "       or\n"
            "%(prog)s [-h] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME "
            "--notebook_per_model_path PATH\n"
        ),
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
    DEFAULT_NOTEBOOK_PER_MODEL_PATH = str(
        resources.files("flyvis.analysis").joinpath("__main_per_model__.ipynb")
    )
    parser.add_argument(
        "--notebook_per_model_path",
        type=str,
        default=DEFAULT_NOTEBOOK_PER_MODEL_PATH,
        help=(
            f"Path of the notebook to execute. "
            f"Default: {DEFAULT_NOTEBOOK_PER_MODEL_PATH} "
        ),
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Perform a dry run without actually launching jobs.",
    )

    args, kwargs = parser.parse_known_intermixed_args()
    run_notebook_ensemble(args, kwargs)
