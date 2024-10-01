import os
from datetime import datetime
from pathlib import Path

import dotenv
import torch
from pytz import timezone

from flyvision.version import __version__

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_device(device)
del torch


dotenv.load_dotenv(dotenv.find_dotenv())

# Set up logging
import logging


def timetz(*args):
    tz = timezone(os.getenv("TIMEZONE", "Europe/Berlin"))
    return datetime.now(tz).timetuple()


logging.Formatter.converter = timetz
logging.basicConfig(
    format="[%(asctime)s] %(module)s:%(lineno)d %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

del logging, timetz

import datamate


def resolve_root_dir():
    "Resolving the root directory in which all data is downloaded and stored."

    # Try to get root directory from environment variable
    root_dir_env = os.getenv(
        "FLYVIS_ROOT_DIR", str(Path(__file__).parent.parent / "data")
    )
    return Path(root_dir_env).expanduser().absolute()


root_dir = resolve_root_dir()
# path for results
results_dir = root_dir / "results"
renderings_dir = root_dir / "renderings"
sintel_dir = root_dir / "SintelDataSet"
connectome_file = root_dir / "connectome/fib25-fib19_v2.2.json"
source_dir = (repo_dir := Path(__file__).parent.parent) / "flyvision"
script_dir = Path(__file__).parent.parent / "scripts"

datamate.set_root_dir(root_dir)
del datamate

# from flyvision._api import *
from .network import *
