"""Normalization constants for stimulus responses.

The normalization constant of a model is the root-mean-square of its responses to
~30 minutes of naturalistic stimuli (Sintel), computed per cell type. Recomputing
it requires simulating the full naturalistic stimuli dataset for every model of an
ensemble, which is expensive and infeasible on a laptop --- while the constants
themselves are a handful of floats per model.

This module stores them so they only ever have to be computed once:

1. Constants for the released ensembles ship with the package
   ([`PRECOMPUTED_FILE`][flyvis.analysis.response_norms.PRECOMPUTED_FILE]) and are
   loaded silently.
2. Constants computed for any other ensemble are written next to the ensemble
   (`<ensemble_dir>/responses_norm.h5`) and reused from there.

[`Ensemble.responses_norm`][flyvis.network.ensemble.Ensemble.responses_norm] goes
through [`responses_norm`][flyvis.analysis.response_norms.responses_norm], which
resolves the above in order and only simulates as a last resort.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from tqdm.auto import tqdm

import flyvis

if TYPE_CHECKING:
    from flyvis.network.ensemble import Ensemble

logging = logging.getLogger(__name__)

__all__ = [
    "ResponseNorms",
    "PRECOMPUTED_FILE",
    "ENSEMBLE_FILE_NAME",
    "responses_norm",
    "compute_response_norms",
    "read_response_norms",
    "write_response_norms",
    "store_response_norms",
    "load_response_norms",
    "checkpoint_hash",
    "checkpoint_identity",
]

#: Constants shipped with the package, one HDF5 group per ensemble name.
PRECOMPUTED_FILE: Path = flyvis.package_dir / "data" / "responses_norm.h5"

#: Name of the per-ensemble cache file written into the ensemble directory.
ENSEMBLE_FILE_NAME: str = "responses_norm.h5"


@dataclass
class ResponseNorms:
    """Normalization constants of one ensemble.

    Attributes:
        ensemble_name: Name of the ensemble, e.g. `"flow/0000"`.
        model_names: Names of the models, e.g. `["flow/0000/000", ...]`.
        cell_types: Cell types in the order of the response `neuron` dimension.
        checkpoints: Checkpoint file names the constants were computed from.
        checkpoint_hashes: SHA256 of those checkpoint files. Empty for constants
            written before hashes were recorded.
        norm: Constants of the unrectified responses, shape (n_models, n_cell_types).
        rectified_norm: Constants of the rectified responses, same shape.
        dataset_config: Config of the naturalistic stimuli dataset as JSON string.

    Note:
        Everything is keyed by model name, never by position, so the order of an
        ensemble does not matter --- see
        [`select`][flyvis.analysis.response_norms.ResponseNorms.select] and
        [`covers`][flyvis.analysis.response_norms.ResponseNorms.covers].
    """

    ensemble_name: str
    model_names: List[str]
    cell_types: List[str]
    checkpoints: List[str]
    norm: np.ndarray
    rectified_norm: np.ndarray
    dataset_config: str = ""
    checkpoint_hashes: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.model_names = [str(name) for name in self.model_names]
        self.cell_types = [str(cell_type) for cell_type in self.cell_types]
        self.checkpoints = [str(checkpoint) for checkpoint in self.checkpoints]
        self.checkpoint_hashes = [str(h) for h in self.checkpoint_hashes]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.ensemble_name}, "
            f"{len(self.model_names)} models, {len(self.cell_types)} cell types)"
        )

    def covers(
        self,
        model_names: List[str],
        checkpoints: List[str],
        checkpoint_hashes: Optional[List[str]] = None,
    ) -> bool:
        """Whether constants for all given models and checkpoints are stored.

        Matching is by model name, so the order of the arguments is irrelevant; the
        i-th name and the i-th checkpoint just have to belong to the same model.

        A model is only covered if its checkpoint is the one the constants were
        computed from. That is decided on the checkpoint's SHA256 whenever both
        sides have one, so that a checkpoint which was retrained in place, or an
        unrelated ensemble that happens to reuse the same names, is rejected rather
        than silently normalized with foreign constants. Only if hashes are missing
        does it fall back to comparing checkpoint file names.

        Args:
            model_names: Model names to look up.
            checkpoints: Checkpoint file name of each of those models.
            checkpoint_hashes: SHA256 of each of those checkpoint files. None or an
                empty string per model if it could not be read.

        Returns:
            True if every model is stored and was recorded from the same checkpoint.
        """
        if checkpoint_hashes is None:
            checkpoint_hashes = [""] * len(model_names)
        stored = dict(
            zip(
                self.model_names,
                zip(self.checkpoints, self.checkpoint_hashes or [""] * len(self)),
            )
        )
        for name, checkpoint, checkpoint_hash in zip(
            model_names, checkpoints, checkpoint_hashes
        ):
            if name not in stored:
                logging.debug("no stored normalization constants for %s", name)
                return False
            stored_checkpoint, stored_hash = stored[name]
            if stored_hash and checkpoint_hash:
                if stored_hash != checkpoint_hash:
                    logging.debug(
                        "stored normalization constants for %s were computed from a "
                        "different checkpoint (%s != %s)",
                        name,
                        stored_hash[:12],
                        checkpoint_hash[:12],
                    )
                    return False
            elif stored_checkpoint != checkpoint:
                logging.debug(
                    "stored normalization constants for %s are from checkpoint %s, "
                    "but %s is requested",
                    name,
                    stored_checkpoint,
                    checkpoint,
                )
                return False
        return True

    def __len__(self) -> int:
        return len(self.model_names)

    def select(self, model_names: List[str], rectified: bool = False) -> np.ndarray:
        """Return constants for the given models, broadcastable to responses.

        Rows are looked up by model name and returned in the requested order, so
        an unsorted ensemble gets its constants in its own order --- the same order
        in which its responses are concatenated along `network_id`.

        Args:
            model_names: Model names in the requested order.
            rectified: Whether to return the constants of the rectified responses.

        Returns:
            Array of shape (n_models, 1, 1, n_cell_types), i.e. broadcastable
            against responses of shape (network_id, sample, frame, neuron).
        """
        index = {name: i for i, name in enumerate(self.model_names)}
        rows = [index[str(name)] for name in model_names]
        values = self.rectified_norm if rectified else self.norm
        return values[rows][:, None, None, :]

    def update(self, other: "ResponseNorms") -> "ResponseNorms":
        """Return a copy with the models of `other` added or overwritten.

        Args:
            other: Constants of the same ensemble to merge in.

        Returns:
            Merged constants, ordered by model name.

        Raises:
            ValueError: If the ensembles or cell types do not match.
        """
        if other.ensemble_name != self.ensemble_name:
            raise ValueError(
                f"cannot merge {other.ensemble_name} into {self.ensemble_name}"
            )
        if other.cell_types != self.cell_types:
            raise ValueError("cell types do not match")

        def rows(norms: "ResponseNorms") -> Dict[str, tuple]:
            hashes = norms.checkpoint_hashes or [""] * len(norms)
            return dict(
                zip(
                    norms.model_names,
                    zip(norms.checkpoints, hashes, norms.norm, norms.rectified_norm),
                )
            )

        merged = rows(self)
        merged.update(rows(other))
        names = sorted(merged)
        return ResponseNorms(
            ensemble_name=self.ensemble_name,
            model_names=names,
            cell_types=self.cell_types,
            checkpoints=[merged[name][0] for name in names],
            checkpoint_hashes=[merged[name][1] for name in names],
            norm=np.stack([merged[name][2] for name in names]),
            rectified_norm=np.stack([merged[name][3] for name in names]),
            dataset_config=other.dataset_config or self.dataset_config,
        )


def _decode(dataset: h5py.Dataset) -> List[str]:
    """Read a variable-length string dataset as a list of str."""
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in dataset[()]
    ]


def read_response_norms(path: Path, ensemble_name: str) -> Optional[ResponseNorms]:
    """Read stored constants of an ensemble from an HDF5 file.

    Args:
        path: Path to the HDF5 file.
        ensemble_name: Name of the ensemble, e.g. `"flow/0000"`.

    Returns:
        The stored constants, or None if the file or the ensemble group is absent.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with h5py.File(path, "r") as file:
            if ensemble_name not in file:
                return None
            group = file[ensemble_name]
            return ResponseNorms(
                ensemble_name=ensemble_name,
                model_names=_decode(group["model_names"]),
                cell_types=_decode(group["cell_types"]),
                checkpoints=_decode(group["checkpoints"]),
                checkpoint_hashes=(
                    _decode(group["checkpoint_hashes"])
                    if "checkpoint_hashes" in group
                    else []
                ),
                norm=group["norm"][()],
                rectified_norm=group["rectified_norm"][()],
                dataset_config=group.attrs.get("dataset_config", ""),
            )
    except (OSError, KeyError) as e:
        logging.debug("could not read normalization constants from %s: %s", path, e)
        return None


def write_response_norms(path: Path, norms: ResponseNorms) -> Path:
    """Write constants of an ensemble into an HDF5 file, replacing its group.

    Args:
        path: Path to the HDF5 file. Created if it does not exist.
        norms: Constants to store.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype()
    with h5py.File(path, "a") as file:
        if norms.ensemble_name in file:
            del file[norms.ensemble_name]
        group = file.require_group(norms.ensemble_name)
        group.create_dataset("model_names", data=norms.model_names, dtype=string_dtype)
        group.create_dataset("cell_types", data=norms.cell_types, dtype=string_dtype)
        group.create_dataset("checkpoints", data=norms.checkpoints, dtype=string_dtype)
        if norms.checkpoint_hashes:
            group.create_dataset(
                "checkpoint_hashes", data=norms.checkpoint_hashes, dtype=string_dtype
            )
        group.create_dataset("norm", data=norms.norm)
        group.create_dataset("rectified_norm", data=norms.rectified_norm)
        group.attrs["dataset_config"] = norms.dataset_config
        group.attrs["flyvis_version"] = flyvis.__version__
    return path


def ensemble_file(ensemble: "Ensemble") -> Path:
    """Path of the per-ensemble cache file.

    Args:
        ensemble: The ensemble.

    Returns:
        Path to `<ensemble_dir>/responses_norm.h5`.
    """
    return Path(ensemble.path) / ENSEMBLE_FILE_NAME


def checkpoint_hash(path: Union[str, Path]) -> str:
    """SHA256 of a checkpoint file.

    Args:
        path: Path to the checkpoint.

    Returns:
        Hexadecimal digest, or an empty string if the file cannot be read.
    """
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for block in iter(lambda: file.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError as e:
        logging.debug("could not hash checkpoint %s: %s", path, e)
        return ""


def checkpoint_identity(ensemble: "Ensemble") -> Tuple[List[str], List[str]]:
    """Identify the checkpoint each model of an ensemble is analyzed at.

    Args:
        ensemble: The ensemble.

    Returns:
        The checkpoint file names and their SHA256, both in the order of
        `ensemble.names`. Hashing the whole ensemble costs a few milliseconds,
        checkpoints are small.
    """
    paths = [
        Path(str(network_view.network(checkpoint="best", lazy=True).checkpoint))
        for network_view in ensemble.values()
    ]
    return [path.name for path in paths], [checkpoint_hash(path) for path in paths]


def load_response_norms(ensemble: "Ensemble") -> Optional[ResponseNorms]:
    """Load stored constants for an ensemble without computing anything.

    Looks into the ensemble directory first, then into the constants shipped with
    the package. Returns None if neither covers all models of the ensemble at the
    checkpoints it is currently pointing at.

    Args:
        ensemble: The ensemble to look up.

    Returns:
        The stored constants, or None.
    """
    checkpoints, hashes = checkpoint_identity(ensemble)
    for path in (ensemble_file(ensemble), PRECOMPUTED_FILE):
        norms = read_response_norms(path, ensemble.name)
        if norms is not None and norms.covers(list(ensemble.names), checkpoints, hashes):
            logging.debug(
                "loaded normalization constants for %s from %s", ensemble.name, path
            )
            return norms
    return None


def compute_response_norms(ensemble: "Ensemble") -> ResponseNorms:
    """Compute the constants of an ensemble from naturalistic stimuli responses.

    Simulates the naturalistic stimuli dataset for each model that has no cached
    responses yet. Models are processed one at a time to bound memory.

    Args:
        ensemble: The ensemble to compute constants for.

    Returns:
        Constants of all models of the ensemble.
    """
    norm, rectified_norm, checkpoints, hashes = [], [], [], []
    cell_types, dataset_config = None, ""

    for network_view in tqdm(
        ensemble.values(),
        desc="Computing normalization constants",
        total=len(ensemble),
    ):
        response_set = network_view.naturalistic_stimuli_responses()
        responses = response_set["responses"].values
        norm.append(_norm(responses, rectified=False)[0])
        rectified_norm.append(_norm(responses, rectified=True)[0])
        # identify the checkpoint the responses were actually computed from, rather
        # than whichever one the ensemble points at now
        checkpoint = Path(str(response_set.coords["checkpoints"].values[0]))
        checkpoints.append(checkpoint.name)
        hashes.append(checkpoint_hash(checkpoint))
        if cell_types is None:
            cell_types = [str(c) for c in response_set.coords["cell_type"].values]
            dataset_config = json.dumps(response_set.attrs.get("config", ""), default=str)
        # close the underlying file handle, the responses are already in memory
        response_set.close()

    return ResponseNorms(
        ensemble_name=ensemble.name,
        model_names=list(ensemble.names),
        cell_types=cell_types,
        checkpoints=checkpoints,
        checkpoint_hashes=hashes,
        norm=np.stack(norm),
        rectified_norm=np.stack(rectified_norm),
        dataset_config=dataset_config,
    )


def _norm(responses: np.ndarray, rectified: bool = False) -> np.ndarray:
    """Root-mean-square of responses over stimuli and frames, per cell type.

    Args:
        responses: Array of shape (n_models, n_samples, n_frames, n_cell_types).
        rectified: Whether to rectify the responses first.

    Returns:
        Array of shape (n_models, n_cell_types).
    """
    responses = np.asarray(responses)
    if rectified:
        responses = np.maximum(responses, 0)
    responses = np.nan_to_num(responses, nan=0.0)
    _, n_samples, n_frames, _ = responses.shape
    return (
        1
        / np.sqrt(n_samples * n_frames)
        * np.linalg.norm(responses, axis=(1, 2), keepdims=True)
    ).squeeze(axis=(1, 2))


def store_response_norms(
    ensemble: "Ensemble",
    norms: ResponseNorms,
    path: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Write constants into the ensemble directory, merging with what is there.

    Args:
        ensemble: The ensemble the constants belong to.
        norms: The constants to store.
        path: Target file. Defaults to `<ensemble_dir>/responses_norm.h5`.

    Returns:
        The path written to, or None if the location is not writable.
    """
    path = Path(path) if path is not None else ensemble_file(ensemble)
    stored = read_response_norms(path, norms.ensemble_name)
    if stored is not None:
        try:
            norms = stored.update(norms)
        except ValueError as e:
            logging.debug("replacing stored normalization constants: %s", e)
    try:
        return write_response_norms(path, norms)
    except OSError as e:
        logging.warning("could not store normalization constants at %s: %s", path, e)
        return None


def responses_norm(
    ensemble: "Ensemble",
    rectified: bool = False,
    force_recompute: bool = False,
    store: bool = True,
) -> np.ndarray:
    """Return the normalization constants of an ensemble.

    Resolves stored constants first (ensemble directory, then the ones shipped
    with the package) and only simulates naturalistic stimuli responses if none
    are available.

    Args:
        ensemble: The ensemble to return constants for.
        rectified: Whether to return the constants of the rectified responses.
        force_recompute: Recompute from responses even if constants are stored.
        store: Whether to write newly computed constants into the ensemble
            directory for reuse.

    Returns:
        Array of shape (n_models, 1, 1, n_cell_types) in the order of
        `ensemble.names`.
    """
    norms = None if force_recompute else load_response_norms(ensemble)
    if norms is None:
        norms = compute_response_norms(ensemble)
        if store:
            store_response_norms(ensemble, norms)
    return norms.select(list(ensemble.names), rectified=rectified)
