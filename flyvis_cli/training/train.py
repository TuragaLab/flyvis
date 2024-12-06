"""Script to train an ensemble of models."""

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


def train_models(args: argparse.Namespace, kwargs: List[str]) -> None:
    """
    Launch training jobs for an ensemble of models.

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
        args.train_script,
        args.dry,
        kwargs,
    )


if __name__ == "__main__":
    CONFIG_PATH = str(resources.files("flyvis") / "config" / "solver.yaml")

    parser = HybridArgumentParser(
        description=(
            "Train an ensemble of models. Launches a job for each model on "
            "the compute cloud."
        ),
        drop_disjoint_from=CONFIG_PATH,
        formatter_class=argparse.RawTextHelpFormatter,
        usage=(
            "\nflyvis train [-h] [--start START] [--end END] [...] "
            "--ensemble_id ENSEMBLE_ID --task_name TASK_NAME "
            "[hydra_options...]\n"
            "       or\n"
            "%(prog)s [-h] [--start START] [--end END] [...] "
            "--ensemble_id ENSEMBLE_ID --task_name TASK_NAME "
            "[hydra_options...]\n"
            "\n"
            "For a full list of hydra options and default arguments, run: "
            "flyvis train-single --help"
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
    DEFAULT_SCRIPT = str(resources.files("flyvis_cli") / "training" / "train_single.py")
    parser.add_argument(
        "--train_script",
        type=str,
        default=DEFAULT_SCRIPT,
        help=f"Script to run for training. Default: {DEFAULT_SCRIPT}",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Perform a dry run without actually launching jobs.",
    )

    args, _ = parser.parse_known_intermixed_args()
    kwargs = parser.hydra_argv()
    train_models(args, kwargs)
