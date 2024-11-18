import json
import logging
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(100)
def warn_once(logger: logging.Logger, msg: str) -> None:
    """
    Log a warning message only once for a given logger and message combination.

    Args:
        logger: The logger object to use for logging.
        msg: The warning message to log.

    Note:
        This function uses an LRU cache to ensure each unique combination of
        logger and message is only logged once.
    """
    logger.warning(msg)


def save_conda_environment(path: Path) -> None:
    """
    Save the current Conda environment to a JSON file.

    Args:
        path: The path where the JSON file will be saved.

    Note:
        The function appends '.json' to the provided path.
    """
    result = subprocess.run(
        ["conda", "list", "--json"], stdout=subprocess.PIPE, text=True, check=False
    )

    installed_packages = json.loads(result.stdout)

    with open(path.with_suffix(".json"), "w") as json_file:
        json.dump(installed_packages, json_file, indent=4)


@contextmanager
def all_logging_disabled(highest_level: int = logging.CRITICAL) -> Any:
    """
    A context manager that prevents any logging messages from being processed.

    Args:
        highest_level: The maximum logging level to disable. Only needs to be
            changed if a custom level greater than CRITICAL is defined.

    Example:
        ```python
        with all_logging_disabled():
            # Code here will not produce any log output
            logging.warning("This warning will not be logged")
        ```

    Reference:
        https://gist.github.com/simon-weber/7853144
    """
    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)
