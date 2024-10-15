import subprocess
import sys
from pathlib import Path


def capture_help(script_path: str) -> str:
    """Runs the script with --help and captures the output."""
    result = subprocess.run(
        [sys.executable, script_path, '--help'], capture_output=True, text=True
    )
    return result.stdout


def script_to_md(script_path: Path, output_dir: Path):
    """Converts a script's help output to a markdown file."""
    help_output = capture_help(str(script_path))

    # Create the output filename
    relative_path = script_path.relative_to(script_path.parents[1])
    output_file = output_dir / f"{relative_path.with_suffix('.md')}"

    # Ensure the output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Generate the script path for documentation
    doc_script_path = f"scripts/{relative_path}"

    # Write the help message to a markdown file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# {script_path.stem.replace('_', ' ').title()}\n\n")
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
        script_to_md(script_path, output_dir)


if __name__ == "__main__":
    main()
