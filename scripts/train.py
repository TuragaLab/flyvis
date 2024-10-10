"""Script to train an ensemble of models."""

import argparse
import logging
from typing import List

from flyvision import script_dir
from flyvision.utils.compute_cloud_utils import launch_range

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
    if "individual_model_ids" in kwargs:
        idx = kwargs.index("individual_model_ids")
        individual_model_ids = kwargs[idx + 1]
        model_ids = individual_model_ids.split(",")
        for _id in model_ids:
            launch_range(
                int(_id),
                int(_id) + 1,
                args.ensemble_id,
                args.task_name,
                args.nP,
                args.gpu,
                args.q,
                args.train_script,
                args.dry,
                kwargs,
            )
    else:
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
    parser = argparse.ArgumentParser(
        description=(
            "Train an ensemble of models. Launches a job for each model on "
            "the cluster."
        )
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
        "--train_script",
        type=str,
        default=f"{str(script_dir)}/training/train_single.py",
        help="Script to run for training.",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Perform a dry run without actually launching jobs.",
    )

    args, kwargs = parser.parse_known_intermixed_args()
    train_models(args, kwargs)
