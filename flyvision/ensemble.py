"""Ensemble of trained networks."""

from __future__ import annotations

import logging
import pickle
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Dict, Generator, Iterable, Iterator, List, Tuple, Union

import numpy as np
import torch
from datamate import get_root_dir
from matplotlib import colormaps as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from numpy.typing import NDArray
from tqdm.auto import tqdm

from flyvision import plots
from flyvision.analysis.clustering import (
    GaussianMixtureClustering,
    compute_umap_and_clustering,
    get_cluster_to_indices,
)
from flyvision.directories import EnsembleDir
from flyvision.initialization import Parameter
from flyvision.network import Network, NetworkDir, NetworkView
from flyvision.utils.cache_utils import context_aware_cache
from flyvision.utils.class_utils import forward_subclass
from flyvision.utils.logging_utils import all_logging_disabled
from flyvision.utils.nn_utils import simulation

logging = logging.getLogger(__name__)


class Ensemble(dict):
    """Dictionary to a collection of trained networks.

    Args:
        path: Path to ensemble directory or list of paths to model directories. Can be
            a single string, then assumes the path is the root directory as configured
            by datamate.
        checkpoint: Checkpoint to load from each network.
        validation_subdir: Subdirectory to look for validation files.
        loss_file_name: Name of the loss file.
        try_sort: Whether to try to sort the ensemble by validation error.

    Attributes:
        names: List of model names.
        name: Ensemble name.
        path: Path to ensemble directory.
        model_paths: List of paths to model directories.
        dir: Directory object for ensemble directory.

    Note: the ensemble is a dynamic dictionary, so you can access the networks
    in the ensemble by name or index. For example, to access the first network
    simply do:
        ensemble[0]
    or
        ensemble['opticflow/000/0000'].
    To create a subset of the ensemble, you can use the same syntax:
        ensemble[0:2]
    """

    def __init__(
        self,
        path: Union[str, PathLike, Iterable, "EnsembleDir"],
        checkpoint="best",
        validation_subdir="validation",
        loss_file_name="loss",
        try_sort=False,
    ):
        # self.model_paths, self.path = model_paths_from_parent(path)

        if isinstance(path, EnsembleDir):
            path = path.path
            self.model_paths, self.path = model_paths_from_parent(path)
            self.dir = path
        elif isinstance(path, PathLike):
            self.model_paths, self.path = model_paths_from_parent(path)
            self.dir = EnsembleDir(self.path)
        elif isinstance(path, str):
            self.dir = EnsembleDir(path)
            self.model_paths, self.path = model_paths_from_parent(self.dir.path)
        elif isinstance(path, Iterable):
            self.model_paths, self.path = model_paths_from_names_or_paths(path)
            self.dir = EnsembleDir(self.path)

        self.names, self.name = model_path_names(self.model_paths)

        self._names = []
        self.model_index = []
        # Initialize pointers to model directories.
        for i, name in tqdm(
            enumerate(self.names), desc="Loading ensemble", total=len(self.names)
        ):
            try:
                with all_logging_disabled():
                    self[name] = NetworkView(
                        NetworkDir(self.model_paths[i]),
                        checkpoint=checkpoint,
                        validation_subdir=validation_subdir,
                        loss_file_name=loss_file_name,
                    )
                    self._names.append(name)
            except AttributeError as e:
                logging.warning(f"Failed to load {name}: {e}")
        self._broken = list(set(self.names) - set(self._names))
        self.names = self._names
        self.model_index = np.arange(len(self.names))
        logging.info(f"Loaded {len(self)} networks.")

        # try rank by validation error by default
        if try_sort:
            self.sort()

        self._init_args = (
            path,
            self[0].checkpoints.choice,
            self[0].checkpoints.validation_subdir,
            self[0].checkpoints.loss_file_name,
            try_sort,
        )
        self.connectome = self[next(iter(self))].connectome
        self.cache = {}

    def __truediv__(self, key):
        return self.__getitem__(key)

    def __getitem__(
        self, key: Union[str, int, slice, NDArray, list]
    ) -> Union[NetworkView, "Ensemble"]:
        if isinstance(key, (int, np.integer)):
            return dict.__getitem__(self, self.names[key])
        elif isinstance(key, slice):
            return self.__class__(self.names[key])
        elif isinstance(key, (np.ndarray, list)):
            return self.__class__(np.array(self.names)[key])
        elif key in self.names:
            return dict.__getitem__(self, key)
        else:
            raise ValueError(f"{key}")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.path})"

    def __dir__(self):
        return list({*dict.__dir__(self), *dict.__iter__(self)})

    def __len__(self):
        return len(self.names)

    def __iter__(self):
        yield from self.names

    def items(self) -> Iterator[Tuple[str, NetworkView]]:
        return iter((k, self[k]) for k in self)

    def keys(self):
        return list(self)

    def values(self) -> List[NetworkView]:
        return [self[k] for k in self]

    def check_configs_match(self):
        """Check if the configurations of the networks in the ensemble match."""
        config0 = self[0].dir.config
        for i in range(1, len(self)):
            diff = config0.diff(self[i].dir.config, name1="first", name2="second").first
            if diff:
                logging.warning(
                    f"{self[0].name} differs from {self[i].name}. Diff is {diff}."
                )
                return False
        return True

    def yield_networks(self) -> Generator[Network, None, None]:
        """Yield initialized networks from the ensemble."""
        # assert self.check_configs_match(), "configurations do not match"
        network = self[0].init_network()
        yield network
        for network_view in self.values()[1:]:
            yield network_view.init_network(network=network)

    def yield_decoders(self):
        """Yield initialized decoders from the ensemble."""
        assert self.check_configs_match(), "configurations do not match"
        decoder = self[0].init_decoder()
        for network_view in self.values():
            yield network_view.init_decoder(decoder=decoder)

    def simulate(self, movie_input: torch.Tensor, dt: float, fade_in: bool = True):
        """Simulate the ensemble activity from movie input.

        Args:
            movie_input: tensor requiring shape (batch_size, n_frames, 1, hexals)
            dt: integration time constant. Warns if dt > 1/50.
            fade_in: whether to use `network.fade_in_state` to compute the initial
                state. Defaults to True. If False, uses the
                `network.steady_state` after 1s of grey input.

        Yields:
            array: response of each individual network

        Note: simulates across batch_size in parallel, i.e., easily leading to OOM for
        large batch sizes.
        """
        for network in tqdm(
            self.yield_networks(),
            desc="Simulating network",
            total=len(self.names),
        ):
            yield (
                network.simulate(
                    movie_input,
                    dt,
                    initial_state=(
                        network.fade_in_state(1.0, dt, movie_input[:, 0])
                        if fade_in
                        else "auto"
                    ),
                )
                .cpu()
                .numpy()
            )

    def simulate_from_dataset(
        self,
        dataset,
        dt: float,
        indices: Iterable[int] = None,
        t_pre: float = 1.0,
        t_fade_in: float = 0.0,
        default_stim_key: str = "lum",
        batch_size: int = 1,
        central_cell_only=True,
    ):
        """Simulate the ensemble activity from a dataset."""
        if central_cell_only:
            central_cells_index = self[0].connectome.central_cells_index[:]

        progress_bar = tqdm(desc="Simulating network", total=len(self.names))

        for network in self.yield_networks():

            def handle_network(network: Network):
                for _, resp in network.stimulus_response(
                    dataset,
                    dt=dt,
                    indices=indices,
                    t_pre=t_pre,
                    t_fade_in=t_fade_in,
                    default_stim_key=default_stim_key,
                    batch_size=batch_size,
                ):
                    if central_cell_only:
                        yield resp[:, :, central_cells_index]
                    else:
                        yield resp

            r = np.stack(list(handle_network(network)))
            yield r.reshape(-1, r.shape[-2], r.shape[-1])

            progress_bar.update(1)
        progress_bar.close()

    def decode(self, movie_input, dt):
        """Decode the ensemble responses with the ensemble decoders."""
        responses = torch.tensor(list(self.simulate(movie_input, dt)))
        for i, decoder in enumerate(self.yield_decoders()):
            with simulation(decoder):
                yield decoder(responses[i]).cpu().numpy()

    def validation_file(self, validation_subdir=None, loss_file_name=None):
        """Return the validation file for each network in the ensemble."""
        network_view0 = self[0]
        if validation_subdir is None:
            validation_subdir = network_view0.checkpoints.validation_subdir
        if loss_file_name is None:
            loss_file_name = network_view0.checkpoints.loss_file_name

        for nv in self.values():
            assert nv.checkpoints.validation_subdir == validation_subdir
            assert nv.checkpoints.loss_file_name == loss_file_name

        return validation_subdir, loss_file_name

    def sort(self, validation_subdir=None, loss_file_name=None):
        """Sort the ensemble by a key."""
        try:
            self.names = sorted(
                self.keys(),
                key=lambda key: dict(
                    zip(
                        self.keys(),
                        self.validation_losses(validation_subdir, loss_file_name),
                    )
                )[key],
                reverse=False,
            )
        except Exception as e:
            logging.info(f"sorting failed: {e}")

    @context_aware_cache(context=lambda self: (self.names))
    def validation_losses(self, subdir: str = None, file: str = None):
        """Return a list of validation losses for each network in the ensemble."""
        subdir, file = self.validation_file(subdir, file)
        losses = [nv.dir[subdir][file][()].min() for nv in self.values()]
        return losses

    def update_checkpoints(
        self,
        checkpoint="best",
        validation_subdir="validation",
        loss_file_name="loss",
    ):
        """Update the checkpoints reference for the ensemble."""
        for nv in self.values():
            nv.update_checkpoint(checkpoint, validation_subdir, loss_file_name)
        self._init_args = (
            self[0].dir.path,
            checkpoint,
            validation_subdir,
            loss_file_name,
            False,
        )

    @contextmanager
    def rank_by_validation_error(self, reverse=False):
        """
        To temporarily sort the ensemble based on a type of task error. In
        ascending order.

        This method sorts the self.names attribute temporarily, which serve as
        references to the Directory instances of the trained Networks.
        """
        _names = deepcopy(self.names)

        try:
            self.names = sorted(
                self.keys(),
                key=lambda key: dict(zip(self.keys(), self.validation_losses()))[key],
                reverse=reverse,
            )
        except Exception as e:
            logging.info(f"sorting failed: {e}")
        try:
            yield
        finally:
            self.names = list(_names)

    @contextmanager
    def ratio(self, best=None, worst=None):
        """To sort and filter the ensemble temporarily by a ratio of models that
        are performing best or worst based on a type of task error.

        The temporary subset is a view of the original ensemble, so initialized
        attributes of its values persist in memory.
        """
        # no-op
        if best is None and worst is None:
            yield
            return

        _names = tuple(self.names)
        _model_index = tuple(self.model_index)

        with self.rank_by_validation_error():
            if best is not None and worst is not None and best + worst > 1:
                raise ValueError("best and worst must add up to 1")

            if best is not None or worst is not None:
                _context_best_names, _context_worst_names = [], []
                if best is not None:
                    _context_best_names = list(self.names[: int(best * len(self))])
                    self._best_ratio = best
                else:
                    self._best_ratio = 0
                if worst is not None:
                    _context_worst_names = list(self.names[-int(worst * len(self)) :])
                    self._worst_ratio = worst
                else:
                    self._worst_ratio = 0

                in_context_names = [*_context_best_names, *_context_worst_names]

                if in_context_names:  # to prevent an empty index
                    self.model_index = np.array([
                        i
                        for i, name in zip(_model_index, _names)
                        if name in in_context_names
                    ])
                self.names = in_context_names
                self.in_context = True
            try:
                yield
            finally:
                self.names = list(_names)
                self.model_index = _model_index
                self._best_ratio = 0.5
                self._worst_ratio = 0.5
                self.in_context = False

    @contextmanager
    def select_items(self, indices: List[int]):
        """To filter the ensemble temporarily by a list of items (int, slice,
        list, array) while maintaining state of the Ensemble instance."""
        # no-op
        try:
            if indices is None:
                yield
                return
            _names = tuple(self.names)
            _model_index = tuple(self.model_index)
            self._names = _names

            if isinstance(indices, (int, np.integer, slice)):
                in_context_names = self.names[indices]
            elif isinstance(indices, (list, np.ndarray)):
                if np.array(indices).dtype == np.array(self.names).dtype:
                    in_context_names = indices
                elif np.array(indices).dtype == np.int_:
                    in_context_names = np.array(self.names)[indices]
                else:
                    raise ValueError(f"{indices}")
            else:
                raise ValueError(f"{indices}")
            self.model_index = np.array([
                i for i, name in zip(_model_index, _names) if name in in_context_names
            ])
            self.names = in_context_names
            self.in_context = True
            yield
        finally:
            self.names = list(_names)
            self.model_index = list(_model_index)
            self.in_context = False

    def task_error(
        self,
        cmap="Blues_r",
        truncate=None,
        vmin=None,
        vmax=None,
    ) -> "TaskError":
        """Return a TaskError object for the ensemble.

        The TaskError object contains the validation losses, the colors, the
        colormap, the norm, and the scalar mapper.
        """
        error = np.array(self.validation_losses())

        if truncate is None:
            # truncate because the maxval would be white with the default colormap
            # which would be invisible on a white background
            truncate = {"minval": 0.0, "maxval": 0.9, "n": 256}
        cmap = cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
        cmap = plots.plt_utils.truncate_colormap(cmap, **truncate)
        sm, norm = plots.plt_utils.get_scalarmapper(
            cmap=cmap,
            vmin=vmin or np.min(error),
            vmax=vmax or np.max(error),
        )
        colors = sm.to_rgba(np.array(error))

        return TaskError(error, colors, cmap, norm, sm)

    def parameters(self):
        """Return the parameters of the ensemble."""
        network_params = {}
        for network_view in self.values():
            chkpt_params = torch.load(network_view.checkpoints.path)
            for key, val in chkpt_params["network"].items():
                if key not in network_params:
                    network_params[key] = []
                network_params[key].append(val.cpu().numpy())
        for key, val in network_params.items():
            network_params[key] = np.array(val)
        return network_params

    def parameter_keys(self):
        """Return the keys of the parameters of the ensemble."""
        self.check_configs_match()
        network_view = self[0]
        config = network_view.dir.config.network

        parameter_keys = {}
        for param_name, param_config in config.node_config.items():
            param = forward_subclass(
                Parameter,
                config={
                    "type": param_config.type,
                    "param_config": param_config,
                    "connectome": network_view.connectome,
                },
            )
            parameter_keys[f"nodes_{param_name}"] = param.keys
        for param_name, param_config in config.edge_config.items():
            param = forward_subclass(
                Parameter,
                config={
                    "type": param_config.type,
                    "param_config": param_config,
                    "connectome": network_view.connectome,
                },
            )
            parameter_keys[f"edges_{param_name}"] = param.keys
        return parameter_keys

    @context_aware_cache(context=lambda self: (self.names))
    def stored_responses(self, subdir, central=True, slice=slice(None)):
        """Return the stored responses of the ensemble.

        Args:
            subdir: The subdirectory where the responses are stored.
            chkpt_key: The checkpoint key corresponding to the checkpoint that the
                ensemble is initialized with. Must match the checkpoint chosen
                for running the synthetic recordings script to precompute responses
                because data is stored under subdir/chkpt_key.
            central: Whether only central responses are expected. Then reads
                activity_central instead of activity.

        Ignores:
            FileNotFoundError: If no recordings are found in the specified directory.
                Then returns None.
        """
        chkpt_key = self[0].checkpoints.current_chkpt_key
        full_subdir = f"{subdir}/{chkpt_key}"
        try:
            responses = np.stack([
                nv.stored_responses(subdir, central=central, slice=slice)
                for nv in self.values()
            ])
            return responses
        except FileNotFoundError:
            logging.info(
                "No recordings found in %s. Run the analysis script for synthetic "
                "recordings to precompute them.",
                full_subdir,
            )
            return None

    def clustering(self, cell_type) -> GaussianMixtureClustering:
        """Return the clustering of the ensemble for a given cell type."""

        if (
            not self.dir.umap_and_clustering
            or not self.dir.umap_and_clustering[cell_type]
        ):
            return compute_umap_and_clustering(self, cell_type)

        path = self.dir.umap_and_clustering[f"{cell_type}.pickle"]
        with open(path, "rb") as file:
            clustering = pickle.load(file)

        return clustering

    def cluster_indices(self, cell_type: str) -> Dict[int, NDArray[int]]:
        """Clusters from responses to naturalistic stimuli of the given cell type.

         Args:
             cell_type: The cell type to return the clusters for.

         Returns: keys are the cluster ids and the values are the model indices
             in the ensemble.

        TODO: add computing the clusters if not available.

         Example:
             ensemble = Ensemble("path/to/ensemble")
             cluster_indices = ensemble.cluster_indices("T4a")
             first_cluster = ensemble[cluster_indices[0]]
        """

        clustering = self.clustering(cell_type)
        cluster_indices = get_cluster_to_indices(
            clustering.embedding.mask, clustering.labels, task_error=self.task_error()
        )

        _models = sorted(np.concatenate(list(cluster_indices.values())))
        if len(_models) != clustering.embedding.mask.sum() or len(_models) > len(self):
            raise ValueError("stored clustering does not match ensemble")

        return cluster_indices

    def responses_norm(self, subdir="naturalistic_stimuli_responses", rectified=False):
        responses = self.stored_responses(subdir, central=True)

        if responses is None:
            return None

        def compute_norm(X, rectified=True):
            """Computes a normalization constant for stimulus
                responses per cell hypothesis, i.e. cell_type independent values.

            Args:
                X: (n_stimuli, n_frames, n_cell_types)
            """
            if rectified:
                X = np.maximum(X, 0)
            n_samples, n_frames, n_cell_types = X.shape

            # replace NaNs with 0
            X[np.isnan(X)] = 0

            return (
                1
                / np.sqrt(n_samples * n_frames)
                * np.linalg.norm(
                    X.reshape(-1, n_cell_types),
                    axis=0,
                    keepdims=False,
                )
            )

        return np.array([
            compute_norm(responses[i], rectified=rectified) for i in self.model_index
        ])


def model_paths_from_parent(path):
    """Return a list of model paths from a parent path."""
    model_paths = sorted(
        filter(
            lambda p: p.name.isnumeric() and p.is_dir(),
            path.iterdir(),
        )
    )
    return model_paths, path


def model_path_names(model_paths):
    """Return a list of model names and an ensemble name from a list of model paths."""
    model_names = [
        str(path).replace(str(get_root_dir()) + "/", "") for path in model_paths
    ]
    ensemble_name = ", ".join(np.unique([n[:-4] for n in model_names]).tolist())
    return model_names, ensemble_name


def model_paths_from_names_or_paths(paths: List[str | Path]) -> Tuple[List[Path], Path]:
    """Return model paths and ensemble path from model names or paths."""
    model_paths = []
    _ensemble_paths = []
    for path in paths:
        if isinstance(path, str):
            # assuming task/ensemble_id/model_id
            if len(path.split("/")) == 3:
                model_paths.append(get_root_dir() / path)
            # assuming task/ensemble_id
            elif len(path.split("/")) == 2:
                model_paths.extend(model_paths_from_parent(path)[0])
        elif isinstance(path, PathLike):
            model_paths.append(path)
        _ensemble_paths.append(model_paths[-1].parent)
    ensemble_path = np.unique(_ensemble_paths)[0]
    return model_paths, ensemble_path


@dataclass
class TaskError:
    """A dataclass that contains the validation losses, the colors,
    the colormap, the norm, and the scalar mapper."""

    values: NDArray
    colors: NDArray
    cmap: Colormap
    norm: Normalize
    scalarmappable: ScalarMappable
