import os
from datetime import datetime
from pathlib import Path
from importlib import resources


import dotenv
import torch
from pytz import timezone

from flyvis.version import __version__

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_default_device(device)
del torch


dotenv.load_dotenv(dotenv.find_dotenv(usecwd=True))

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


package_dir = Path(__file__).parent
repo_dir = (
    package_dir if package_dir.parent.name == "site-packages" else package_dir.parent
)


def resolve_root_dir():
    """Resolve the root directory where all data is downloaded and stored."""
    root_dir_env = os.getenv("FLYVIS_ROOT_DIR", str(repo_dir / "data"))
    root_dir = Path(root_dir_env).expanduser().absolute()
    if not root_dir.exists():
        root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir


root_dir = resolve_root_dir()
# path for results
results_dir = root_dir / "results"
renderings_dir = root_dir / "renderings"
sintel_dir = root_dir / "SintelDataSet"
connectome_file = package_dir / "connectome/fib25-fib19_v2.2.json"
source_dir = repo_dir / "flyvis"
config_dir = repo_dir / "config"
script_dir = repo_dir / "flyvis_cli"
examples_dir = repo_dir / "examples"

import datamate

datamate.set_root_dir(root_dir)
del datamate

from .utils import *
from .connectome import *
from .datasets import *
from .network import *
from .task import *
from .analysis import *
from .solver import *
