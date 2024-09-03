import json
import logging
import subprocess
from contextlib import contextmanager
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


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.

    https://gist.github.com/simon-weber/7853144
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)
