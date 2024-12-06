"""Script to run synthetic recordings for each model of an ensemble."""

import argparse
import logging
from importlib import resources
from typing import List

from flyvis.utils.compute_cloud_utils import launch_range

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def run_synthetic_recordings(args: argparse.Namespace, kwargs: List[str]) -> None:
    """
    Launch synthetic recording job for each model of an ensemble.

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
        args.synthetic_recordings_script,
        args.dry,
        kwargs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run synthetic recordings for each model of an ensemble on the compute cloud."
        ),
        usage=(
            "\nflyvis record [-h] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME "
            "[options] [synthetic_recordings_script_args...]\n"
            "       or\n"
            "%(prog)s [-h] [...] --ensemble_id ENSEMBLE_ID --task_name TASK_NAME "
            "[options] [synthetic_recordings_script_args...]\n"
            "\n"
            "For a full list of options and default arguments, run: "
            "flyvis synthetic-recordings-single --help"
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
    DEFAULT_SCRIPT = str(
        resources.files("flyvis_cli") / "analysis" / "synthetic_recordings_single.py"
    )
    parser.add_argument(
        "--synthetic_recordings_script",
        type=str,
        default=DEFAULT_SCRIPT,
        help=f"Script to run for synthetic recordings. Default: {DEFAULT_SCRIPT}",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Perform a dry run without actually launching jobs.",
    )

    args, kwargs = parser.parse_known_intermixed_args()
    run_synthetic_recordings(args, kwargs)
