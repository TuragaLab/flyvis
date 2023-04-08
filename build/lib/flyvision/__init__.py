import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
del torch

from pathlib import Path
import datamate


root_dir = Path(__file__).parent.parent.absolute() / "data"
results_dir = root_dir / "results"
sintel_dir = root_dir / "SintelDataSet"
connectome_file = root_dir / "connectome/fib25-fib19_v2.2.json"

datamate.set_root_dir(root_dir)
del datamate

from flyvision._api import *
