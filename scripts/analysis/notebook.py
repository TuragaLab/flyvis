"""Script to run a jupyter notebook for a particular model."""

import logging
import os
import sys
import tempfile

import papermill as pm

import flyvis
from flyvis.utils.config_utils import HybridArgumentParser

if __name__ == "__main__":
    parser = HybridArgumentParser(
        hybrid_args={
            "notebook_per_model": {
                "type": bool,
                "required": False,
                "help": (
                    "Flag to set the output path to the per-model notebook path. "
                    "Requires ensemble_and_network_id."
                ),
            },
            "ensemble_and_network_id": {
                "type": str,
                "required": False,
                "help": (
                    "Id in form of task_name/ensemble_id/network_id, e.g., "
                    "flow/0000/000. Required if notebook_per_model is True."
                ),
            },
            "per_ensemble": {
                "type": bool,
                "required": False,
                "help": (
                    "Flag to set the output path to the per-ensemble notebook path. "
                    "Requires ensemble_id and task_name."
                ),
            },
            "task_name": {
                "type": str,
                "required": False,
                "help": (
                    "Name of the task, e.g. flow. Required if per_ensemble is True."
                ),
            },
            "ensemble_id": {
                "type": int,
                "required": False,
                "help": (
                    "Id of the ensemble, e.g. 0045. Required if per_ensemble is True."
                ),
            },
        },
        description=(
            "Run a Jupyter notebook using papermill. "
            "Required arguments depend on the specific notebook. "
            "Pass any additional arguments as key:type=value triplets."
        ),
    )
    parser.add_argument(
        "--notebook_path",
        type=str,
        help="Path of the notebook to execute, e.g. /path/to/__main__.ipynb.",
    )
    parser.add_argument(
        "--notebook_per_model_path",
        type=str,
        help=(
            "Path of the notebook to execute for each model, e.g. "
            "/path/to/__main_per_model__.ipynb."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help=(
            "Path for the output notebook. If not provided, a temporary file "
            "will be used."
        ),
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Perform a dry run without actually executing the notebook.",
    )

    # arguments are parsed including hybrid and typed hybrid arguments and directly
    # passed to papermill
    args = parser.parse_with_hybrid_args()

    if args.dry:
        logging.info(
            "Dry run. Not executing notebook %s. Passed args: %s",
            args.notebook_path,
            args,
        )
        sys.exit(0)

    logging.info("Executing notebook %s", args.notebook_path)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as f:
        if args.notebook_per_model:
            input_path = args.notebook_per_model_path
            output_path = (
                flyvis.results_dir
                / f"{args.task_name}/{args.ensemble_and_network_id}"
                / args.notebook_per_model_path.split(os.sep)[-1]
            )

        elif args.per_ensemble:
            input_path = args.notebook_path
            output_path = (
                flyvis.results_dir
                / f"{args.task_name}/{args.ensemble_id:04}"
                / args.notebook_path.split(os.sep)[-1]
            )
        else:
            output_path = args.output_path or f.name
        logging.info("Output will be saved to %s.", output_path)
        pm.execute_notebook(
            flyvis.repo_dir / input_path,
            output_path,
            parameters=args.__dict__,
        )
    logging.info("Done executing notebook %s", input_path)
