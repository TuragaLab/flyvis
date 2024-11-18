"""Ensemble of trained networks."""

from __future__ import annotations

import logging
import os
import pickle
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
import xarray as xr
from cachetools import FIFOCache
from matplotlib import colormaps as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from numpy.typing import NDArray
from torch import nn
from tqdm.auto import tqdm

import flyvis
from flyvis.analysis import stimulus_responses, stimulus_responses_currents
from flyvis.analysis.clustering import (
    GaussianMixtureClustering,
    compute_umap_and_clustering,
    get_cluster_to_indices,
)
from flyvis.analysis.visualization import plots
from flyvis.connectome import get_avgfilt_connectome
from flyvis.utils.cache_utils import context_aware_cache
from flyvis.utils.chkpt_utils import (
    best_checkpoint_default_fn,
    recover_network,
    resolve_checkpoints,
)
from flyvis.utils.class_utils import forward_subclass
from flyvis.utils.logging_utils import all_logging_disabled
from flyvis.utils.nn_utils import simulation

from .directories import EnsembleDir, NetworkDir
from .initialization import Parameter
from .network import Network
from .network_view import NetworkView

logging = logging.getLogger(__name__)

__all__ = ["Ensemble"]


class Ensemble(dict):
    """Dictionary to a collection of trained networks.

    Args:
        path: Path to ensemble directory or list of paths to model directories.
            Can be a single string, then assumes the path is the root directory
            as configured by datamate.
        network_class: Class to use for initializing networks.
        root_dir: Root directory for model paths.
        connectome_getter: Function to get the connectome.
        checkpoint_mapper: Function to map checkpoints.
        best_checkpoint_fn: Function to determine best checkpoint.
        best_checkpoint_fn_kwargs: Kwargs for best_checkpoint_fn.
        recover_fn: Function to recover network.
        try_sort: Whether to try to sort the ensemble by validation error.

    Attributes:
        names (List[str]): List of model names.
        name (str): Ensemble name.
        path (Path): Path to ensemble directory.
        model_paths (List[Path]): List of paths to model directories.
        dir (EnsembleDir): Directory object for ensemble directory.

    Note:
        The ensemble is a dynamic dictionary, so you can access the networks
        in the ensemble by name or index. For example, to access the first network
        simply do:
        ```python
        ensemble[0]
        ```
        or
        ```python
        ensemble['flow/000/0000']
        ```
        Slice to create a subset of the ensemble:
        ```python
        ensemble[0:2]
        ```
    """

    def __init__(
        self,
        path: Union[str, Path, Iterable, "EnsembleDir"],
        network_class: nn.Module = Network,
        root_dir: Path = flyvis.results_dir,
        connectome_getter: Callable = get_avgfilt_connectome,
        checkpoint_mapper: Callable = resolve_checkpoints,
        best_checkpoint_fn: Callable = best_checkpoint_default_fn,
        best_checkpoint_fn_kwargs: dict = {
            "validation_subdir": "validation",
            "loss_file_name": "epe",
        },
        recover_fn: Callable = recover_network,
        try_sort: bool = False,
    ):
        if isinstance(path, EnsembleDir):
            path = path.path
            self.model_paths, self.path = model_paths_from_parent(path)
            self.dir = path
        elif isinstance(path, Path):
            self.model_paths, self.path = model_paths_from_parent(path)
            self.dir = EnsembleDir(self.path)
        elif isinstance(path, str):
            self.dir = EnsembleDir(path)
            self.model_paths, self.path = model_paths_from_parent(self.dir.path)
        elif isinstance(path, Iterable):
            self.model_paths, self.path = model_paths_from_names_or_paths(path, root_dir)
            self.dir = EnsembleDir(self.path)
        else:
            raise TypeError(
                f"Unsupported path type: {type(path)}. "
                "Expected EnsembleDir, str, PathLike, or Iterable."
            )

        self.names, self.name = model_path_names(self.model_paths)
        self.in_context = False

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
                        network_class=network_class,
                        root_dir=root_dir,
                        connectome_getter=connectome_getter,
                        checkpoint_mapper=checkpoint_mapper,
                        best_checkpoint_fn=best_checkpoint_fn,
                        best_checkpoint_fn_kwargs=best_checkpoint_fn_kwargs,
                        recover_fn=recover_fn,
                    )
                    self._names.append(name)
            except AttributeError as e:
                logging.warning(f"Failed to load {name}: {e}")
        self._broken = list(set(self.names) - set(self._names))
        self.names = self._names
        self.model_index = np.arange(len(self.names))
        logging.info(f"Loaded {len(self)} networks.")

        if try_sort:
            self.sort()

        self._init_args = (
            path,
            network_class,
            root_dir,
            connectome_getter,
            checkpoint_mapper,
            best_checkpoint_fn,
            best_checkpoint_fn_kwargs,
            recover_fn,
            try_sort,
        )
        self.connectome = self[next(iter(self))].connectome
        self.cache = FIFOCache(maxsize=3)

    def __getitem__(
        self, key: Union[str, int, slice, NDArray, list]
    ) -> Union[NetworkView, "Ensemble"]:
        """Get item from the ensemble.

        Args:
            key: Key to access the ensemble.
                Can be a string, int, slice, NDArray, or list.

        Returns:
            NetworkView or Ensemble: The requested item or subset of the ensemble.

        Raises:
            ValueError: If the key is invalid.
        """
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

    def __repr__(self) -> str:
        """Return a string representation of the Ensemble."""
        return f"{self.__class__.__name__}({self.path})"

    def __dir__(self) -> List[str]:
        """Return a list of attributes for the Ensemble."""
        return list({*dict.__dir__(self), *dict.__iter__(self)})

    def __len__(self) -> int:
        """Return the number of networks in the Ensemble."""
        return len(self.names)

    def __iter__(self) -> Iterator[str]:
        """Iterate over the names of the networks in the Ensemble."""
        yield from self.names

    def items(self) -> Iterator[Tuple[str, NetworkView]]:
        """Return an iterator over the (name, NetworkView) pairs in the Ensemble."""
        return iter((k, self[k]) for k in self)

    def keys(self) -> List[str]:
        """Return a list of network names in the Ensemble."""
        return list(self)

    def values(self) -> List[NetworkView]:
        """Return a list of NetworkViews in the Ensemble."""
        return [self[k] for k in self]

    def _clear_cache(self) -> None:
        """Clear the cache of the Ensemble."""
        self.cache = {}

    def _clear_memory(self) -> None:
        """Clear the memory of all NetworkViews in the Ensemble."""
        for nv in self.values():
            nv._clear_memory()

    def check_configs_match(self) -> bool:
        """Check if the configurations of the networks in the ensemble match.

        Returns:
            bool: True if all configurations match, False otherwise.
        """
        config0 = self[0].dir.config
        for i in range(1, len(self)):
            diff = config0.diff(self[i].dir.config, name1="first", name2="second").first
            if diff and not (len(diff) == 1 and "network_name" in diff[0]):
                logging.warning(
                    "%(first)s differs from %(second)s. Diff is %(diff)s.",
                    {"first": self[0].name, "second": self[i].name, "diff": diff},
                )
                return False
        return True

    def yield_networks(self) -> Generator[Network, None, None]:
        """Yield initialized networks from the ensemble.

        Yields:
            Network: Initialized network from the ensemble.
        """
        network = self[0].init_network()
        yield network
        for network_view in self.values()[1:]:
            yield network_view.init_network(network=network)

    def yield_decoders(self) -> Generator[nn.Module, None, None]:
        """Yield initialized decoders from the ensemble.

        Yields:
            nn.Module: Initialized decoder from the ensemble.
        """
        assert self.check_configs_match(), "configurations do not match"
        decoder = self[0].init_decoder()
        for network_view in self.values():
            yield network_view.init_decoder(decoder=decoder)

    def simulate(
        self, movie_input: torch.Tensor, dt: float, fade_in: bool = True
    ) -> Generator[np.ndarray, None, None]:
        """Simulate the ensemble activity from movie input.

        Args:
            movie_input: Tensor with shape (batch_size, n_frames, 1, hexals).
            dt: Integration time constant. Warns if dt > 1/50.
            fade_in: Whether to use `network.fade_in_state` to compute the initial
                state. Defaults to True. If False, uses the
                `network.steady_state` after 1s of grey input.

        Yields:
            np.ndarray: Response of each individual network.

        Note:
            Simulates across batch_size in parallel, which can easily lead to OOM for
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
        central_cell_only: bool = True,
    ) -> Generator[np.ndarray, None, None]:
        """Simulate the ensemble activity from a dataset.

        Args:
            dataset: Dataset to simulate from.
            dt: Integration time constant.
            indices: Indices of stimuli to simulate. Defaults to None (all stimuli).
            t_pre: Time before stimulus onset. Defaults to 1.0.
            t_fade_in: Fade-in time. Defaults to 0.0.
            default_stim_key: Default stimulus key. Defaults to "lum".
            batch_size: Batch size for simulation. Defaults to 1.
            central_cell_only: Whether to return only central cells. Defaults to True.

        Yields:
            np.ndarray: Simulated responses for each network.
        """
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

    def decode(
        self, movie_input: torch.Tensor, dt: float
    ) -> Generator[np.ndarray, None, None]:
        """Decode the ensemble responses with the ensemble decoders.

        Args:
            movie_input: Input movie tensor.
            dt: Integration time constant.

        Yields:
            np.ndarray: Decoded responses for each network.
        """
        responses = torch.tensor(list(self.simulate(movie_input, dt)))
        for i, decoder in enumerate(self.yield_decoders()):
            with simulation(decoder):
                yield decoder(responses[i]).cpu().numpy()

    def validation_file(
        self,
        validation_subdir: Optional[str] = None,
        loss_file_name: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Return the validation file for each network in the ensemble.

        Args:
            validation_subdir: Subdirectory for validation files. Defaults to None.
            loss_file_name: Name of the loss file. Defaults to None.

        Returns:
            Tuple[str, str]: Validation subdirectory and loss file name.
        """
        network_view0 = self[0]
        if validation_subdir is None:
            validation_subdir = network_view0.best_checkpoint_fn_kwargs.get(
                "validation_subdir"
            )
        if loss_file_name is None:
            loss_file_name = network_view0.best_checkpoint_fn_kwargs.get("loss_file_name")

        return validation_subdir, loss_file_name

    def sort(
        self,
        validation_subdir: Optional[str] = None,
        loss_file_name: Optional[str] = None,
    ) -> None:
        """Sort the ensemble by validation loss.

        Args:
            validation_subdir: Subdirectory for validation files. Defaults to None.
            loss_file_name: Name of the loss file. Defaults to None.
        """
        try:
            self.names = sorted(
                self.keys(),
                key=lambda key: dict(
                    zip(
                        self.keys(),
                        self.min_validation_losses(validation_subdir, loss_file_name),
                    )
                )[key],
                reverse=False,
            )
        except Exception as e:
            logging.info(f"sorting failed: {e}")

    def argsort(
        self,
        validation_subdir: Optional[str] = None,
        loss_file_name: Optional[str] = None,
    ) -> np.ndarray:
        """Return the indices that would sort the ensemble by validation loss.

        Args:
            validation_subdir: Subdirectory for validation files. Defaults to None.
            loss_file_name: Name of the loss file. Defaults to None.

        Returns:
            np.ndarray: Indices that would sort the ensemble.
        """
        return np.argsort(
            self.min_validation_losses(
                *self.validation_file(validation_subdir, loss_file_name)
            )
        )

    def zorder(
        self,
        validation_subdir: Optional[str] = None,
        loss_file_name: Optional[str] = None,
    ) -> np.ndarray:
        """Return the z-order of the ensemble based on validation loss.

        Args:
            validation_subdir: Subdirectory for validation files. Defaults to None.
            loss_file_name: Name of the loss file. Defaults to None.

        Returns:
            np.ndarray: Z-order of the ensemble.
        """
        return len(self) - self.argsort(validation_subdir, loss_file_name).argsort()

    @context_aware_cache(context=lambda self: (self.names))
    def validation_losses(
        self, subdir: Optional[str] = None, file: Optional[str] = None
    ) -> np.ndarray:
        """Return a list of validation losses for each network in the ensemble.

        Args:
            subdir: Subdirectory for validation files. Defaults to None.
            file: Name of the loss file. Defaults to None.

        Returns:
            np.ndarray: Validation losses for each network.
        """
        subdir, file = self.validation_file(subdir, file)
        losses = np.array([nv.dir[subdir][file][()] for nv in self.values()])
        return losses

    def min_validation_losses(
        self, subdir: Optional[str] = None, file: Optional[str] = None
    ) -> np.ndarray:
        """Return the minimum validation loss of the ensemble.

        Args:
            subdir: Subdirectory for validation files. Defaults to None.
            file: Name of the loss file. Defaults to None.

        Returns:
            np.ndarray: Minimum validation losses for each network.
        """
        losses = self.validation_losses(subdir, file)
        if losses.ndim == 1:
            return losses
        return np.min(losses, axis=1)

    @contextmanager
    def rank_by_validation_error(self, reverse: bool = False):
        """Temporarily sort the ensemble based on validation error.

        Args:
            reverse: Whether to sort in descending order. Defaults to False.

        Yields:
            None
        """
        _names = deepcopy(self.names)

        try:
            self.names = sorted(
                self.keys(),
                key=lambda key: dict(zip(self.keys(), self.min_validation_losses()))[key],
                reverse=reverse,
            )
        except Exception as e:
            logging.info(f"sorting failed: {e}")
        try:
            yield
        finally:
            self.names = list(_names)

    @contextmanager
    def ratio(self, best: Optional[float] = None, worst: Optional[float] = None):
        """Temporarily filter the ensemble by a ratio of best or worst performing models.

        Args:
            best: Ratio of best performing models to keep. Defaults to None.
            worst: Ratio of worst performing models to keep. Defaults to None.

        Yields:
            None

        Raises:
            ValueError: If best and worst sum to more than 1.
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
        """Temporarily filter the ensemble by a list of indices.

        Args:
            indices: List of indices to select.

        Yields:
            None

        Raises:
            ValueError: If indices are invalid.
        """
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
        cmap: Union[str, Colormap] = "Blues_r",
        truncate: Optional[Dict[str, Union[float, int]]] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> "TaskError":
        """Return a TaskError object for the ensemble.

        Args:
            cmap: Colormap to use. Defaults to "Blues_r".
            truncate: Dictionary to truncate the colormap. Defaults to None.
            vmin: Minimum value for normalization. Defaults to None.
            vmax: Maximum value for normalization. Defaults to None.

        Returns:
            TaskError: Object containing validation losses, colors, colormap, norm,
                and scalar mapper.
        """
        error = self.min_validation_losses()

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

    def parameters(self) -> Dict[str, np.ndarray]:
        """Return the parameters of the ensemble.

        Returns:
            Dict[str, np.ndarray]: Dictionary of parameter arrays.
        """
        network_params = {}
        for network_view in self.values():
            chkpt_params = torch.load(network_view.network('best').checkpoint)
            for key, val in chkpt_params["network"].items():
                if key not in network_params:
                    network_params[key] = []
                network_params[key].append(val.cpu().numpy())
        for key, val in network_params.items():
            network_params[key] = np.array(val)
        return network_params

    def parameter_keys(self) -> Dict[str, List[str]]:
        """Return the keys of the parameters of the ensemble.

        Returns:
            Dict[str, List[str]]: Dictionary of parameter keys.
        """
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

    @wraps(stimulus_responses.flash_responses)
    @context_aware_cache(context=lambda self: (self.names))
    def flash_responses(self, *args, **kwargs) -> xr.Dataset:
        """Generate flash responses."""
        return stimulus_responses.flash_responses(self, *args, **kwargs)

    @wraps(stimulus_responses.moving_edge_responses)
    @context_aware_cache(context=lambda self: (self.names))
    def moving_edge_responses(self, *args, **kwargs) -> xr.Dataset:
        """Generate moving edge responses."""
        return stimulus_responses.moving_edge_responses(self, *args, **kwargs)

    @wraps(stimulus_responses.moving_bar_responses)
    @context_aware_cache(context=lambda self: (self.names))
    def moving_bar_responses(self, *args, **kwargs) -> xr.Dataset:
        """Generate moving bar responses."""
        return stimulus_responses.moving_bar_responses(self, *args, **kwargs)

    @wraps(stimulus_responses.naturalistic_stimuli_responses)
    @context_aware_cache(context=lambda self: (self.names))
    def naturalistic_stimuli_responses(self, *args, **kwargs) -> xr.Dataset:
        """Generate naturalistic stimuli responses."""
        return stimulus_responses.naturalistic_stimuli_responses(self, *args, **kwargs)

    @wraps(stimulus_responses.central_impulses_responses)
    @context_aware_cache(context=lambda self: (self.names))
    def central_impulses_responses(self, *args, **kwargs) -> xr.Dataset:
        """Generate central ommatidium impulses responses."""
        return stimulus_responses.central_impulses_responses(self, *args, **kwargs)

    @wraps(stimulus_responses.spatial_impulses_responses)
    @context_aware_cache(context=lambda self: (self.names))
    def spatial_impulses_responses(self, *args, **kwargs) -> xr.Dataset:
        """Generate spatial ommatidium impulses responses."""
        return stimulus_responses.spatial_impulses_responses(self, *args, **kwargs)

    @wraps(stimulus_responses_currents.moving_edge_currents)
    @context_aware_cache(context=lambda self: (self.names))
    def moving_edge_currents(
        self, *args, **kwargs
    ) -> List[stimulus_responses_currents.ExperimentData]:
        """Generate moving edge currents."""
        return stimulus_responses_currents.moving_edge_currents(self, *args, **kwargs)

    @context_aware_cache
    def clustering(self, cell_type) -> GaussianMixtureClustering:
        """Return the clustering of the ensemble for a given cell type.

        Args:
            cell_type: The cell type to cluster.

        Returns:
            GaussianMixtureClustering: Clustering object for the given cell type.

        Raises:
            ValueError: If clustering is not available in context.
        """
        if self.in_context:
            raise ValueError("clustering is not available in context")

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

        Returns:
            Dict[int, NDArray[int]]: Keys are the cluster ids and the values are the
            model indices in the ensemble.

        Example:
            ```python
            ensemble = Ensemble("path/to/ensemble")
            cluster_indices = ensemble.cluster_indices("T4a")
            first_cluster = ensemble[cluster_indices[0]]
            ```

        Raises:
            ValueError: If stored clustering does not match ensemble.
        """
        clustering = self.clustering(cell_type)
        cluster_indices = get_cluster_to_indices(
            clustering.embedding.mask,
            clustering.labels,
            task_error=self.task_error(),
        )

        _models = sorted(np.concatenate(list(cluster_indices.values())))
        if len(_models) != clustering.embedding.mask.sum() or len(_models) > len(self):
            raise ValueError("stored clustering does not match ensemble")

        return cluster_indices

    def responses_norm(self, rectified: bool = False) -> np.ndarray:
        """Compute the norm of responses to naturalistic stimuli.

        Args:
            rectified: Whether to rectify responses before computing norm.

        Returns:
            np.ndarray: Norm of responses for each network.
        """
        response_set = self.naturalistic_stimuli_responses()
        responses = response_set['responses'].values

        def compute_norm(X, rectified=True):
            """Computes a normalization constant for stimulus
                responses per cell hypothesis, i.e. cell_type independent values.

            Args:
                X: (n_stimuli, n_frames, n_cell_types)
            """
            if rectified:
                X = np.maximum(X, 0)
            n_models, n_samples, n_frames, n_cell_types = X.shape

            # replace NaNs with 0
            X[np.isnan(X)] = 0

            return (
                1
                / np.sqrt(n_samples * n_frames)
                * np.linalg.norm(
                    X,
                    axis=(1, 2),
                    keepdims=True,
                )
            )

        return np.take(
            compute_norm(responses, rectified=rectified), self.model_index, axis=0
        )


def model_path_names(model_paths: List[Path]) -> Tuple[List[str], str]:
    """Return a list of model names and an ensemble name from a list of model paths.

    Args:
        model_paths: List of paths to model directories.

    Returns:
        Tuple[List[str], str]: List of model names and ensemble name.
    """
    model_names = [os.path.sep.join(path.parts[-3:]) for path in model_paths]
    ensemble_name = ", ".join(
        np.unique([
            os.path.sep.join(n.split(os.path.sep)[:2]) for n in model_names
        ]).tolist()
    )
    return model_names, ensemble_name


def model_paths_from_parent(path: Path) -> Tuple[List[Path], Path]:
    """Return a list of model paths from a parent path.

    Args:
        path: Parent path to search for model directories.

    Returns:
        Tuple[List[Path], Path]: List of model paths and parent path.
    """
    model_paths = sorted(
        filter(
            lambda p: p.name.isnumeric() and p.is_dir(),
            path.iterdir(),
        )
    )
    return model_paths, path


def model_paths_from_names_or_paths(
    paths: List[Union[str, Path]], root_dir: Path
) -> Tuple[List[Path], Path]:
    """Return model paths and ensemble path from model names or paths.

    Args:
        paths: List of model names or paths.
        root_dir: Root directory for model paths.

    Returns:
        Tuple[List[Path], Path]: List of model paths and ensemble path.

    Raises:
        ValueError: If an invalid path type is provided.
        NotImplementedError: If multiple ensemble paths are found.
    """
    model_paths = []
    _ensemble_paths = []
    for path in paths:
        if isinstance(path, str):
            path_obj = Path(path)
            path_parts = path_obj.parts
            # assuming task/ensemble_id/model_id
            if len(path_parts) == 3:
                model_paths.append(root_dir / path_obj)
            # assuming task/ensemble_id
            elif len(path_parts) == 2:
                model_paths.extend(model_paths_from_parent(path_obj)[0])
        elif isinstance(path, Path):
            model_paths.append(path)
        else:
            raise ValueError(f"Invalid path type: {path}")
        _ensemble_paths.append(model_paths[-1].parent)
    # Ensure all ensemble paths are the same
    ensemble_paths_set = set(_ensemble_paths)
    if len(ensemble_paths_set) != 1:
        raise NotImplementedError("Multiple ensemble paths found")
    ensemble_path = _ensemble_paths[0]
    return model_paths, ensemble_path


@dataclass
class TaskError:
    """A dataclass that contains the validation losses, colors, colormap, norm,
    and scalar mapper.

    Attributes:
        values (NDArray): Validation loss values.
        colors (NDArray): Colors corresponding to validation losses.
        cmap (Colormap): Colormap used for visualization.
        norm (Normalize): Normalization object for the colormap.
        scalarmappable (ScalarMappable): ScalarMappable object for the colormap.
    """

    values: NDArray
    colors: NDArray
    cmap: Colormap
    norm: Normalize
    scalarmappable: ScalarMappable
