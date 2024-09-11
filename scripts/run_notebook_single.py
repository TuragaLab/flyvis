"""Script to run a jupyter notebook for a particular model.

If output_path is provided, a file is created that stores the output.
"""

import logging
import tempfile

import papermill as pm

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
    parser.add_argument(
        "--notebook_path",
        type=str,
        help="Path of the notebook to execute, e.g. /path/to/__main__.ipynb.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
    )

    # arguments are parsed including hybrid and typed hybrid arguments and directly
    # passed to papermill
    args = parser.parse_with_hybrid_args()

    logging.info("Executing notebook %s", args.notebook_path)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as f:
        output_path = args.output_path or f.name
        logging.info("Output will be saved to %s.", output_path)
        pm.execute_notebook(
            args.notebook_path,
            output_path,
            parameters=args.__dict__,
        )
    logging.info("Done executing notebook %s", args.notebook_path)
