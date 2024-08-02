import json
import subprocess
from functools import lru_cache


@lru_cache(100)
def warn_once(logger, msg: str):
    logger.warning(msg)


def save_conda_environment(path):
    # Use the conda list command to get a list of installed packages and their versions
    result = subprocess.run(
        ["conda", "list", "--json"], stdout=subprocess.PIPE, text=True, check=False
    )

    # Parse the JSON output
    installed_packages = json.loads(result.stdout)

    # Write the parsed JSON data to the specified file path
    with open(path.with_suffix(".json"), "w") as json_file:
        json.dump(installed_packages, json_file, indent=4)
