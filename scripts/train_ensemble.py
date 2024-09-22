"""Submit runs to the cluster."""

import argparse
import logging

from flyvision import script_dir
from flyvision.utils.lsf_utils import launch_range, launch_single

logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s", level=logging.INFO
)
logging = logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an ensemble of networks.")
    parser.add_argument("--start", type=int, default=0, help="Start id of ensemble.")
    parser.add_argument("--end", type=int, default=50, help="End id of ensemble.")

    parser.add_argument("--nP", type=int, default=4, help="Numnber of processors.")
    parser.add_argument("--gpu", type=str, default="num=1")
    parser.add_argument("--q", type=str, default="gpu_l4")
    parser.add_argument(
        "--ensemble_id",
        type=int,
        default="",
        help=(
            "Id of the ensemble, e.g. 0045. The ultimate name of a model will be of "
            "the form task/ensemble_id/<start-end>"
        ),
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="",
        help="Name given to the task, e.g. opticflow.",
    )

    parser.add_argument(
        "--train_script",
        type=str,
        default=f"{str(script_dir)}/training/train_single.py",
        help="Script to run.",
    )
    parser.add_argument(
        "--launch_training",
        dest="launch_training",
        action="store_true",
        help="Train models.",
    )
    parser.add_argument(
        "--val_script",
        type=str,
        default=f"{str(script_dir)}/validation/val_single.py",
        help="Script to run.",
    )
    parser.add_argument(
        "--launch_validation",
        dest="launch_validation",
        action="store_true",
        help="Validate models.",
    )
    parser.add_argument(
        "--synthetic_recordings_script",
        type=str,
        default=f"{str(script_dir)}/analysis/synthetic_recordings_single.py",
        help="Script to run.",
    )
    parser.add_argument(
        "--launch_synthetic_recordings",
        dest="launch_synthetic_recordings",
        action="store_true",
        help="Run synthetic recordings.",
    )
    parser.add_argument(
        "--ensemble_analysis_script",
        type=str,
        default=f"{str(script_dir)}/analysis/ensemble_analysis.py",
        help="Script to run.",
    )
    parser.add_argument(
        "--launch_ensemble_analysis",
        dest="launch_ensemble_analysis",
        action="store_true",
        help="Run ensemble analysis.",
    )
    parser.add_argument(
        "--launch_notebook_single",
        action="store_true",
        help="Run notebooks.",
    )

    parser.add_argument(
        "--dry",
        dest="dry",
        action="store_true",
    )

    # note: kwargs is an ordered list(str) of kwargs: ["-kw", "val", ...]
    args, kwargs = parser.parse_known_intermixed_args()

    if args.launch_training:
        if "individual_model_ids" in kwargs:
            # run individual model ids, e.g. when a few failed
            # because of server side errors, then specify
            # 'individual_model_ids 1,4,5' in the kwargs
            idx = kwargs.index("individual_model_ids")
            individual_model_ids = kwargs[idx + 1]
            model_ids = individual_model_ids.split(",")
            for _id in model_ids:
                launch_range(
                    _id,
                    _id + 1,
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
                kwargs,  # note: kwargs is an ordered list(str) of kwargs: ["-kw",
                #   "val", ...]
            )

    if args.launch_validation:
        launch_range(
            args.start,
            args.end,
            args.ensemble_id,
            args.task_name,
            args.nP,
            args.gpu,
            args.q,
            args.val_script,
            args.dry,
            kwargs,
        )

    if args.launch_synthetic_recordings:
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

    if args.launch_ensemble_analysis:
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

    if args.launch_notebook_single:
        launch_range(
            args.start,
            args.end,
            args.ensemble_id,
            args.task_name,
            args.nP,
            args.gpu,
            args.q,
            f"{str(script_dir)}/run_notebook_single.py",
            args.dry,
            kwargs,
        )
