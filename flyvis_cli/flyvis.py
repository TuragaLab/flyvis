"""CLI entry point and pipeline manager for ensemble operations.


Example usage:
    ```bash
    # Run a pipeline of operations
    python flyvis.py \
        --ensemble_id 0001 \
        --task_name flow \
        train \
        validate \
        record \
        analysis \
        notebook_per_model \
        notebook_per_ensemble
    ```
"""

# TODO: could look into click and typer for better CLI handling but this works for now.

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCRIPTS_DIR = Path(__file__).parent

SCRIPT_COMMANDS = {
    "train": SCRIPTS_DIR / "training/train.py",
    "train_single": SCRIPTS_DIR / "training/train_single.py",
    "validate": SCRIPTS_DIR / "validation/validate.py",
    "val_single": SCRIPTS_DIR / "validation/val_single.py",
    "record": SCRIPTS_DIR / "analysis/record.py",
    "synthetic_recordings_single": SCRIPTS_DIR
    / "analysis/synthetic_recordings_single.py",
    "analysis": SCRIPTS_DIR / "analysis/analysis.py",
    "ensemble_analysis": SCRIPTS_DIR / "analysis/ensemble_analysis.py",
    "notebook_per_model": SCRIPTS_DIR / "analysis/notebook_per_model.py",
    "notebook_per_ensemble": SCRIPTS_DIR / "analysis/notebook_per_ensemble.py",
    "notebook": SCRIPTS_DIR / "analysis/notebook.py",
    "download_pretrained_models": SCRIPTS_DIR / "download_pretrained_models.py",
}


def run_script(script_path: Path, args: List[str]) -> None:
    """
    Run a Python script with the given arguments.

    Args:
        script_path: Path to the script to run.
        args: List of command-line arguments to pass to the script.
    """
    cmd = [sys.executable, str(script_path)] + args
    subprocess.run(cmd, check=True)


def filter_args(argv: List[str], commands: List[str]) -> List[str]:
    """
    Filter out commands from command line arguments.

    Args:
        argv: List of command line arguments (typically sys.argv[1:])
        commands: List of commands to filter out

    Returns:
        List of arguments with commands removed
    """
    # Find positions of commands in original argv
    command_positions = []
    for i, arg in enumerate(argv):
        if arg in commands:
            command_positions.append(i)

    # Create filtered arguments list without the commands
    return [arg for i, arg in enumerate(argv) if i not in command_positions]


def handle_help_request(argv: List[str]) -> bool:
    """
    Check if help is requested for a specific command and handle it.

    Args:
        argv: Command line arguments (typically sys.argv)

    Returns:
        True if help was handled, False otherwise
    """
    if len(argv) > 1 and argv[1] in SCRIPT_COMMANDS:
        help_flags = {"-h", "--help"}
        if any(flag in argv[2:] for flag in help_flags):
            command = argv[1]
            run_script(SCRIPT_COMMANDS[command], argv[2:])
            return True
    return False


def main():
    # Handle help requests first
    if handle_help_request(sys.argv):
        return 0

    # Original argument parsing continues if not showing help for a specific command
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
        "commands",
        nargs="+",
        choices=list(SCRIPT_COMMANDS.keys()),
        help="Commands to run in order.",
        metavar="command",
    )

    parser.epilog = """
Runs multiple operations on an ensemble of models.
This is to pipeline jobs on the cluster.
Each command corresponds to a script that launches required jobs.

{}

All additional arguments are passed directly to the respective scripts.
For detailed help on each command, run: flyvis <command> --help
""".format('\n'.join(f"{cmd:<20} : Runs {path}" for cmd, path in SCRIPT_COMMANDS.items()))

    try:
        args, remaining = parser.parse_known_args()

        # Filter out commands from arguments
        filtered_args = filter_args(sys.argv[1:], args.commands)

        for command in args.commands:
            run_script(SCRIPT_COMMANDS[command], filtered_args)
    except SystemExit as e:
        parser.print_help()
        sys.exit(e.code)

    return 0


if __name__ == "__main__":
    sys.exit(main())
