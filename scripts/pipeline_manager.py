"""Pipeline manager for ensemble operations.


Example usage:
    ```bash
    # Run a pipeline of operations
    python ensemble_manager.py --ensemble_id 0045 --task_name flow \
        --command train validate record analysis notebook_per_model notebook \
        --start 0 --end 50 \
        --notebook_path path/to/notebook.ipynb \
        --output_path path/to/output.ipynb
    ```
"""

import argparse
import subprocess
import sys
from typing import List


def run_script(script_name: str, args: List[str]) -> None:
    """
    Run a Python script with the given arguments.

    Args:
        script_name: Name of the script to run.
        args: List of command-line arguments to pass to the script.
    """
    cmd = [sys.executable, script_name] + args
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manage ensemble operations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ensemble_id", type=int, required=True, help="Id of the ensemble, e.g. 0045."
    )
    parser.add_argument(
        "--task_name", type=str, required=True, help="Name given to the task, e.g. flow."
    )
    parser.add_argument(
        "--command",
        nargs="+",
        choices=[
            "train",
            "validate",
            "record",
            "analysis",
            "notebook_per_model",
            "notebook",
        ],
        required=True,
        help="Commands to run in order.",
    )

    parser.epilog = """
Runs multiple operations on an ensemble of models.
This is to pipeline jobs on the cluster.
Each command corresponds to a script that launches required jobs.

train               : Runs train.py
validate            : Runs validate.py
record              : Runs record.py
analysis            : Runs analysis.py
notebook_per_model  : Runs notebook_per_model.py
notebook            : Runs notebook.py

All arguments after --command are passed directly to the respective scripts.
For detailed help on each command, run the individual script with --help.
"""

    args, remaining = parser.parse_known_args()

    for command in args.command:
        run_script(f"{command}.py", sys.argv[2:])
