import os
from datetime import datetime
from pathlib import Path

import dotenv
import torch
from pytz import timezone


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_device(device)
del torch

# Set up logging
import logging


def timetz(*args):
    tz = timezone("Europe/Berlin")
    return datetime.now(tz).timetuple()


logging.Formatter.converter = timetz
logging.basicConfig(
    format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

del logging, timetz

import datamate


def resolve_root_dir():
    "Resolving the root directory in which all data is downloaded and stored."

    dotenv.load_dotenv(dotenv.find_dotenv())

    # Try to get root directory from environment variable
    root_dir_env = os.getenv(
        "FLYVIS_ROOT_DIR", str(Path(__file__).parent.parent / "data")
    )
    return Path(root_dir_env).expanduser().absolute()


root_dir = resolve_root_dir()
results_dir = root_dir / "results"
sintel_dir = root_dir / "SintelDataSet"
connectome_file = root_dir / "connectome/fib25-fib19_v2.2.json"
source_dir = (repo_dir := Path(__file__).parent.parent) / "flyvision"

datamate.set_root_dir(root_dir)
del datamate

from flyvision._api import *
