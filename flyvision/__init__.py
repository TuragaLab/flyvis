import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
del torch

from pathlib import Path
import datamate
import json


def resolve_root_dir():
    "Resolving the root directory in which all data is downloaded and stored."
    config_path = Path(__file__).parent / "config.json"
    config = None
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        new_config = {"root_dir": None, "status": None}

    # set from existing config
    if config and config["root_dir"] is not None:
        root_dir = Path(config["root_dir"]).absolute()
    # set from user input with default and save
    else:
        default_root_dir = Path(__file__).parent.parent.absolute() / "data"
        user_root_dir = input(
            f"Enter root directory for data. Leave empty for {str(default_root_dir)}:"
        )
        if user_root_dir in ["q", "quit", "exit"]:
            raise KeyboardInterrupt("import cancelled")
        if user_root_dir in ["", "default"]:
            root_dir = default_root_dir
            new_config["status"] = "default"
        else:
            root_dir = Path(user_root_dir).expanduser().absolute()
            new_config["status"] = "custom"
        new_config["root_dir"] = str(root_dir)
        json.dump(new_config, open(Path(__file__).parent / "config.json", "w"))

    return root_dir


root_dir = resolve_root_dir()
results_dir = root_dir / "results"
sintel_dir = root_dir / "SintelDataSet"
connectome_file = root_dir / "connectome/fib25-fib19_v2.2.json"

datamate.set_root_dir(root_dir)
del datamate

from flyvision._api import *
