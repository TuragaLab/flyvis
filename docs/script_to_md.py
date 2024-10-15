import subprocess
import sys
from pathlib import Path

HEADERS = {
    "download_pretrained_models.py": "Download Pretrained Models",
    "pipeline_manager.py": "Pipeline Manager to Launch Multiple Scripts on Cluster",
    "training/train_single.py": "Run Training for Single Model",
    "training/train.py": "Launch Ensemble Training on Cluster",
    "validation/val_single.py": "Run Validation for Single Model",
    "validation/validate.py": "Launch Ensemble Validation on Cluster",
    "analysis/synthetic_recordings_single.py": "Run Synthetic Recordings",
    "analysis/record.py": "Launch Synthetic Recordings on Cluster",
    "analysis/ensemble_analysis.py": "Run Ensemble Analysis",
    "analysis/analysis.py": "Launch Ensemble Analysis on Cluster",
    "analysis/notebook.py": "Run Notebook",
    "analysis/notebook_per_ensemble.py": "Launch Notebook Per Ensemble on Cluster",
    "analysis/notebook_per_model.py": "Launch Notebook Per Model on Cluster",
}


def capture_help(script_path: str) -> str:
    """Runs the script with --help and captures the output."""
    result = subprocess.run(
        [sys.executable, script_path, '--help'], capture_output=True, text=True
    )
    return result.stdout


def script_to_md(script_path: Path, output_dir: Path, scripts_dir: Path):
    """Converts a script's help output to a markdown file."""
    help_output = capture_help(str(script_path))

    # Create the output filename
    relative_path = script_path.relative_to(scripts_dir)
    output_file = output_dir / f"{relative_path.with_suffix('.md')}"

    # Ensure the output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Generate the script path for documentation
    doc_script_path = f"{relative_path}"

    # Write the help message to a markdown file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# {HEADERS[str(relative_path)]}\n\n")
        f.write(f"`{doc_script_path}`\n\n")
        f.write("```\n" + help_output + "\n```")


def main():
    # Path to the scripts directory
    scripts_dir = Path(__file__).parent.parent / "scripts"

    # Path to the output directory
    output_dir = Path(__file__).parent / "docs" / "reference" / "scripts"

    # Process all Python scripts in the scripts directory and its subdirectories
    for script_path in scripts_dir.rglob("*.py"):
        print(script_path)
        script_to_md(script_path, output_dir, scripts_dir)


if __name__ == "__main__":
    main()
