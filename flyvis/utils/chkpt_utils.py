import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn

import flyvis
from flyvis.utils.logging_utils import warn_once

logger = logging.getLogger(__name__)


def recover_network(
    network: nn.Module,
    state_dict: Union[Dict, Path, str],
    ensemble_and_network_id: str = None,
) -> nn.Module:
    """
    Load network parameters from state dict.

    Args:
        network: FlyVision network.
        state_dict: State or path to checkpoint containing the "network" parameters.
        ensemble_and_network_id: Optional identifier for the network.

    Returns:
        The updated network.
    """
    state = get_from_state_dict(state_dict, "network")
    if state is not None:
        network.load_state_dict(state)
        logging.info(
            "Recovered network state%s",
            f" {ensemble_and_network_id}." if ensemble_and_network_id else ".",
        )
    else:
        logging.warning("Could not recover network state.")
    return network


def recover_decoder(
    decoder: Dict[str, nn.Module], state_dict: Union[Dict, Path], strict: bool = True
) -> Dict[str, nn.Module]:
    """
    Recover multiple decoders from state dict.

    Args:
        decoder: Dictionary of decoders.
        state_dict: State or path to checkpoint.
        strict: Whether to strictly enforce that the keys in state_dict match.

    Returns:
        The updated dictionary of decoders.
    """
    states = get_from_state_dict(state_dict, "decoder")
    if states is not None:
        for key, dec in decoder.items():
            state = states.pop(key, None)
            if state is not None:
                dec.load_state_dict(state, strict=strict)
                logging.info("Recovered %s decoder state.", key)
            else:
                logging.warning("Could not recover state of %s decoder.", key)
    else:
        logging.warning("Could not recover decoder states.")
    return decoder


def recover_optimizer(
    optimizer: torch.optim.Optimizer, state_dict: Union[Dict, Path]
) -> torch.optim.Optimizer:
    """
    Recover optimizer state from state dict.

    Args:
        optimizer: PyTorch optimizer.
        state_dict: State or path to checkpoint.

    Returns:
        The updated optimizer.
    """
    state = get_from_state_dict(state_dict, "optim")
    if state is not None:
        optimizer.load_state_dict(state)
        logging.info("Recovered optimizer state.")
    else:
        logging.warning("Could not recover optimizer state.")
    return optimizer


def recover_penalty_optimizers(
    optimizers: Dict[str, torch.optim.Optimizer], state_dict: Union[Dict, Path]
) -> Dict[str, torch.optim.Optimizer]:
    """
    Recover penalty optimizers from state dict.

    Args:
        optimizers: Dictionary of penalty optimizers.
        state_dict: State or path to checkpoint.

    Returns:
        The updated dictionary of penalty optimizers.
    """
    states = get_from_state_dict(state_dict, "penalty_optims")
    if states is not None:
        for key, optim in optimizers.items():
            state = states.pop(key, None)
            if state is not None:
                optim.load_state_dict(state)
                logging.info("Recovered %s optimizer state.", key)
            else:
                logging.warning("Could not recover state of %s optimizer.", key)
    else:
        logging.warning("Could not recover penalty optimizer states.")
    return optimizers


def get_from_state_dict(state_dict: Union[Dict, Path, str], key: str) -> Dict:
    """
    Get a specific key from the state dict.

    Args:
        state_dict: State dict or path to checkpoint.
        key: Key to retrieve from the state dict.

    Returns:
        The value associated with the key in the state dict.

    Raises:
        TypeError: If state_dict is not of type Path, str, or dict.
    """
    if state_dict is None:
        return None
    if isinstance(state_dict, (Path, str)):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            state = torch.load(
                state_dict, map_location=flyvis.device, weights_only=False
            ).pop(key, None)
    elif isinstance(state_dict, dict):
        state = state_dict.get(key, None)
    else:
        raise TypeError(
            f"state_dict must be of type Path, str or dict, but is {type(state_dict)}."
        )
    return state


@dataclass
class Checkpoints:
    """
    Dataclass to store checkpoint information.

    Attributes:
        indices: List of checkpoint indices.
        paths: List of checkpoint paths.
    """

    indices: List[int]
    paths: List[Path]

    def __repr__(self):
        return (
            f"Checkpoints(\n"
            f"  indices={repr(self.indices)},\n"
            f"  paths={repr(self.paths)},\n"
            f")"
        )


def resolve_checkpoints(
    networkdir: "flyvis.network.NetworkDir",
) -> Checkpoints:
    """
    Resolve checkpoints from network directory.

    Args:
        networkdir: FlyVision network directory.

    Returns:
        A Checkpoints object containing indices and paths of checkpoints.
    """
    indices, paths = checkpoint_index_to_path_map(networkdir.chkpts.path)
    return Checkpoints(indices, paths)


def checkpoint_index_to_path_map(
    path: Path, glob: str = "chkpt_*"
) -> Tuple[List[int], List[Path]]:
    """
    Returns all numerical identifiers and paths to checkpoints stored in path.

    Args:
        path: Checkpoint directory.
        glob: Glob pattern for checkpoint files.

    Returns:
        A tuple containing a list of indices and a list of paths to checkpoints.
    """
    import re

    path.mkdir(exist_ok=True)
    paths = np.array(sorted(list((path).glob(glob))))
    try:
        _index = [int(re.findall(r"\d{1,10}", p.parts[-1])[0]) for p in paths]
        _sorting_index = np.argsort(_index)
        paths = paths[_sorting_index].tolist()
        index = np.array(_index)[_sorting_index].tolist()
        return index, paths
    except IndexError:
        return [], paths


def best_checkpoint_default_fn(
    path: Path,
    validation_subdir: str = "validation",
    loss_file_name: str = "loss",
) -> Path:
    """
    Find the best checkpoint based on the minimum loss.

    Args:
        path: Path to the network directory.
        validation_subdir: Subdirectory containing validation data.
        loss_file_name: Name of the loss file.

    Returns:
        Path to the best checkpoint.
    """
    networkdir = flyvis.NetworkDir(path)
    checkpoint_dir = networkdir.chkpts.path
    indices, paths = checkpoint_index_to_path_map(checkpoint_dir, glob="chkpt_*")
    loss_file_name = check_loss_name(networkdir[validation_subdir], loss_file_name)
    index = np.argmin(networkdir[validation_subdir][loss_file_name][()])
    index = indices[index]
    path = paths[index]
    return path


def check_loss_name(loss_folder, loss_file_name: str) -> str:
    """
    Check if the loss file name exists in the loss folder.

    Args:
        loss_folder: The folder containing loss files.
        loss_file_name: The name of the loss file to check.

    Returns:
        The validated loss file name.
    """
    if loss_file_name not in loss_folder and "loss" in loss_folder:
        warn_once(
            logging,
            f"{loss_file_name} not in {loss_folder.path}, but 'loss' is. "
            "Falling back to 'loss'. You can rerun the ensemble validation to make "
            "appropriate recordings of the losses.",
        )
        loss_file_name = "loss"
    return loss_file_name


if __name__ == "__main__":
    nv = flyvis.NetworkView("flow/9998/000")
    print(resolve_checkpoints(nv.dir))
