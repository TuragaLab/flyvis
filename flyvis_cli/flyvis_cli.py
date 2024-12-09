"""CLI entry point and pipeline manager for ensemble operations.


Example usage:
    ```bash
    # Run a pipeline of operations
    flyvis \
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
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import List, Tuple

SCRIPTS_DIR = resources.files('flyvis_cli')

SCRIPT_COMMANDS = {
    "train": SCRIPTS_DIR / "training/train.py",
    "train-single": SCRIPTS_DIR / "training/train_single.py",
    "validate": SCRIPTS_DIR / "validation/validate.py",
    "val-single": SCRIPTS_DIR / "validation/val_single.py",
    "record": SCRIPTS_DIR / "analysis/record.py",
    "synthetic-recordings-single": SCRIPTS_DIR
    / "analysis/synthetic_recordings_single.py",
    "analysis": SCRIPTS_DIR / "analysis/analysis.py",
    "ensemble-analysis": SCRIPTS_DIR / "analysis/ensemble_analysis.py",
    "notebook-per-model": SCRIPTS_DIR / "analysis/notebook_per_model.py",
    "notebook-per-ensemble": SCRIPTS_DIR / "analysis/notebook_per_ensemble.py",
    "notebook": SCRIPTS_DIR / "analysis/notebook.py",
    "download-pretrained": SCRIPTS_DIR / "download_pretrained_models.py",
    "init-config": SCRIPTS_DIR / "init_config.py",
}


def run_script(script_path: Path, args: List[str]) -> None:
    """
    Run a Python script with the given arguments.

    Args:
        script_path: Path to the script to run.
        args: List of command-line arguments to pass to the script.
    """
    cmd = [sys.executable, str(script_path)] + args
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}", file=sys.stderr)
        sys.exit(1)


def filter_args(
    argv: List[str], allowed_commands: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Filter out commands from command line arguments.

    Args:
        argv: List of command line arguments (typically sys.argv[1:])
        commands: List of commands to filter out

    Returns:
        List of arguments with commands removed
    """
    selected_commands = []
    other_args = []
    for arg in argv:
        if arg in allowed_commands:
            selected_commands.append(arg)
        else:
            other_args.append(arg)

    return selected_commands, other_args


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

    parser = argparse.ArgumentParser(
        description=(
            "Run flyvis pipelines or individual scripts with compute cloud options."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ensemble_id", type=int, required=False, help="Id of the ensemble, e.g. 0045."
    )
    parser.add_argument(
        "--task_name", type=str, required=False, help="Name given to the task, e.g. flow."
    )
    parser.add_argument(
        "commands",
        nargs="+",
        # choices=list(SCRIPT_COMMANDS.keys()),
        help="Commands to run in order.",
        metavar="command",
    )

    parser.epilog = """
Runs multiple operations on an ensemble of models.
This is to pipeline jobs on the compute cloud.
Each command corresponds to a script that launches required jobs.

{}

All additional arguments are passed directly to the respective scripts.
For detailed help on each command, run: flyvis <command> --help
""".format('\n'.join(f"{cmd:<20} : Runs {path}" for cmd, path in SCRIPT_COMMANDS.items()))

    # args, remaining = parser.parse_known_args()

    # Filter out commands from arguments
    selected_commands, other_args = filter_args(
        sys.argv[1:], list(SCRIPT_COMMANDS.keys())
    )
    if not selected_commands:
        parser.print_help()
        return 1

    for command in selected_commands:
        run_script(SCRIPT_COMMANDS[command], other_args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
