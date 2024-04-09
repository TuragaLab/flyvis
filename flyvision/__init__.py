import os
from pathlib import Path
import dotenv

import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
del torch

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

datamate.set_root_dir(root_dir)
del datamate

from flyvision._api import *
