"""Script to run a jupyter notebook for a particular model."""

import logging
import sys
import tempfile

import papermill as pm

from flyvision.utils.config_utils import HybridArgumentParser

if __name__ == "__main__":
    parser = HybridArgumentParser(
        hybrid_args=None,
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
        output_path = args.output_path or f.name
        logging.info("Output will be saved to %s.", output_path)
        pm.execute_notebook(
            args.notebook_path,
            output_path,
            parameters=args.__dict__,
        )
    logging.info("Done executing notebook %s", args.notebook_path)
