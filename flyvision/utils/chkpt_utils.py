import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn

import flyvision
from flyvision.utils.logging_utils import warn_once

logging = logging.getLogger(__name__)


def init_or_get_checkpoints(
    path: Path, glob: str = "chkpt_*"
) -> Tuple[List[int], List[Path]]:
    """Returns all numerical identifier and paths to checkpoints stored in path.

    Args:
        path (Path): checkpoint directory.

    Returns:
        Tuple: List of indices and list of paths to checkpoints.
    """
    import re

    path.mkdir(exist_ok=True)
    paths = np.array(sorted(list((path).glob(glob))))
    try:
        # sorting existing checkpoints:
        # if the index had up to 6 digits, but the string format
        # of the index expected 5, then the sorting is more save on the
        # numbers instead of the string.
        _index = [int(re.findall(r"\d{1,10}", p.parts[-1])[0]) for p in paths]
        _sorting_index = np.argsort(_index)
        paths = paths[_sorting_index].tolist()
        index = np.array(_index)[_sorting_index].tolist()
        return index, paths
    except IndexError:
        return [], paths


def check_loss_name(loss_folder, loss_file_name):
    if loss_file_name not in loss_folder and "loss" in loss_folder:
        warn_once(
            logging,
            f"{loss_file_name} not in {loss_folder.path}, but 'loss' is."
            "Falling back to 'loss'. You can rerun the ensemble validation to make"
            " appropriate recordings of the losses.",
        )
        loss_file_name = "loss"
    return loss_file_name


def recover_network(network: nn.Module, state_dict: Union[Dict, Path]) -> None:
    """Loads network parameters from state dict.

    Args:
        network: flyvis network.
        state_dict: state or path to checkpoint,
                    which contains the "network" parameters.
    """
    state = get_from_state_dict(state_dict, "network")
    if state is not None:
        network.load_state_dict(state)
        logging.info("Recovered network state.")
    else:
        logging.warning("Could not recover network state.")
    return network


def recover_decoder(
    decoder: Dict[str, nn.Module], state_dict: Union[Dict, Path], strict=True
) -> None:
    """Same as _recover_network for multiple decoders."""
    states = get_from_state_dict(state_dict, "decoder")
    if states is not None:
        for key, dec in decoder.items():
            state = states.pop(key, None)
            if state is not None:
                dec.load_state_dict(state, strict=strict)
                logging.info(f"Recovered {key} decoder state.")
            else:
                logging.warning(f"Could not recover state of {key} decoder.")
    else:
        logging.warning("Could not recover decoder states.")
    return decoder


def recover_optimizer(
    optimizer: torch.optim.Optimizer, state_dict: Union[Dict, Path]
) -> None:
    """Same as _recover_network for optimizer."""
    state = get_from_state_dict(state_dict, "optim")
    if state is not None:
        optimizer.load_state_dict(state)
        logging.info("Recovered optimizer state.")
    else:
        logging.warning("Could not recover optimizer state.")
    return optimizer


def recover_penalty_optimizers(
    optimizers: Dict[str, torch.optim.Optimizer], state_dict: Union[Dict, Path]
) -> None:
    """Same as _recover_network for penalty optimizers."""
    states = get_from_state_dict(state_dict, "penalty_optims")
    if states is not None:
        for key, optim in optimizers.items():
            state = states.pop(key, None)
            if state is not None:
                optim.load_state_dict(state)
                logging.info(f"Recovered {key} optimizer state.")
            else:
                logging.warning(f"Could not recover state of {key} optimizer.")
    else:
        logging.warning("Could not recover penalty optimizer states.")
    return optimizers


def get_from_state_dict(state_dict: Union[Dict, Path], key: str) -> Dict:
    if state_dict is None:
        return
    if isinstance(state_dict, Path):
        state = torch.load(state_dict, map_location=flyvision.device).pop(key, None)
    elif isinstance(state_dict, dict):
        state = state_dict.get(key, None)
    return state


@dataclass
class Checkpoints:
    choice: Union[str, int]
    index: int
    path: Path
    indices: List[int]
    paths: List[Path]
    validation_subdir: str = "validation"
    loss_file_name: str = "loss"


def resolve_checkpoints(
    networkdir: "flyvision.network.NetworkDir",
    checkpoint: Union[int, str] = "best",
    validation_subdir: str = "validation",
    loss_file_name: str = "loss",
) -> Checkpoints:
    """Resolves checkpoints from networkdir."""

    if networkdir.status == "paper results":
        # This is for the shared trained models, which follow a slightly different
        # naming convention and only store one checkpoint in comparison to new models
        # to make sure we don't mix them up.
        return _resolve_paper_results(networkdir)

    index = _check_checkpoint(networkdir, checkpoint, validation_subdir, loss_file_name)
    indices, paths = init_or_get_checkpoints(networkdir.chkpts.path)

    return Checkpoints(
        checkpoint,
        index,
        paths[index],
        indices,
        paths,
        validation_subdir,
        loss_file_name,
    )


def _check_checkpoint(
    networkdir: "flyvision.network.NetworkDir",
    checkpoint: Union[int, str] = "best",
    validation_subdir: str = "validation",
    loss_file_name: str = "loss",
) -> int:
    """Validates the checkpoint index. Transform 'best' to the index with the minimal
    validation error.
    """
    checkpoint_dir = networkdir.chkpts.path
    indices, _ = init_or_get_checkpoints(checkpoint_dir, glob="chkpt_*")

    if (
        checkpoint == "best"
        and validation_subdir in networkdir
        and loss_file_name in networkdir[validation_subdir]
    ):
        loss_file_name = check_loss_name(networkdir[validation_subdir], loss_file_name)
        index = np.argmin(networkdir[validation_subdir][loss_file_name][:])
        checkpoint = indices[index]
    elif checkpoint in indices:
        checkpoint = indices[checkpoint]
    else:
        checkpoint = slice(None)

    return checkpoint


def _resolve_paper_results(networkdir):
    warn_once(
        logging,
        "Loading paper result from stored checkpoint and validation subdir...",
    )
    return Checkpoints(
        choice="best_chkpt",
        index=0,
        path=networkdir["best_chkpt"],
        indices=[0],
        paths=[networkdir["best_chkpt"]],
        validation_subdir="",
        loss_file_name="validation_loss",
    )
