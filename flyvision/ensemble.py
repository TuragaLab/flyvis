"""Ensemble of trained networks."""
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple, Union
import logging
from os import PathLike
from copy import deepcopy
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, Normalize
from matplotlib import colormaps as cm
import torch
import numpy as np
from numpy.typing import NDArray
from contextlib import contextmanager
from datamate import Directory

from flyvision import plots, results_dir
from flyvision.network import Network, NetworkView, NetworkDir
from flyvision.plots.plt_utils import init_plot
from flyvision.utils.nn_utils import simulation

logging = logging.getLogger()


class Ensemble(dict):
    """Dictionary to a collection of trained networks.

    Args:
        path: Path to ensemble directory or list of paths to model directories.

    Attributes:
        names: List of model names.
        name: Ensemble name.
        path: Path to ensemble directory.
        model_paths: List of paths to model directories.
        dir: Directory object for ensemble directory.

    Note, the ensemble is a dynamic dictionary, so you can access the networks
    in the ensemble by name or index. For example, to access the first network
    simply do:
        ensemble[0]
    or
        ensemble['opticflow/000/0000'].
    To create a subset of the ensemble, you can use the same syntax:
        ensemble[0:2]
    """

    def __init__(self, path: Union[PathLike, Iterable, "EnsembleDir"]):
        # self.model_paths, self.path = model_paths_from_parent(path)

        if isinstance(path, EnsembleDir):
            path = path.path
            self.model_paths, self.path = model_paths_from_parent(path)
        elif isinstance(path, PathLike):
            self.model_paths, self.path = model_paths_from_parent(path)
        elif isinstance(path, Iterable):
            self.model_paths, self.path = model_paths_from_names_or_paths(path)

        self.names, self.name = model_path_names(self.model_paths)

        # Initialize pointers to model directories.
        for i, name in enumerate(self.names):
            self[name] = NetworkView(NetworkDir(self.model_paths[i]))

        self.dir = EnsembleDir(self.path)

        # rank by validation error by default
        self.names = sorted(
            self.keys(),
            key=lambda key: dict(zip(self.keys(), self.validation_losses()))[key],
            reverse=False,
        )

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

    def yield_networks(self, checkpoint="best_chkpt") -> Iterator[Network]:
        """Yield initialized networks from the ensemble."""
        # TODO: since the nn.Module is simply updated with inidividual weights
        # for efficiency, this requires a config check somwhere to make sure the
        # networks are compatible.
        network = self[0].init_network(chkpt=checkpoint)
        yield network
        for network_view in self.values()[1:]:
            yield network_view.init_network(chkpt=checkpoint, network=network)

    def yield_decoders(self, checkpoint="best_chkpt"):
        """Yield initialized decoders from the ensemble."""
        raise NotImplementedError("not implemented yet")
        decoder = self[0].init_decoder(chkpt=checkpoint)
        for network_view in self.values():
            yield network_view.init_decoder(chkpt=checkpoint, decoder=decoder)

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
        """
        for network in self.yield_networks():
            yield network.simulate(
                movie_input,
                dt,
                initial_state=network.fade_in_state(1.0, dt, movie_input[:, 0])
                if fade_in
                else "auto",
            ).cpu().numpy()

    def decode(self, movie_input, dt):
        """Decode the ensemble responses with the ensemble decoders."""
        raise NotImplementedError("not implemented yet")
        responses = torch.tensor(list(self.simulate(movie_input, dt)))
        for i, decoder in enumerate(self.yield_decoders()):
            with simulation(decoder):
                yield decoder(responses[i]).cpu().numpy()

    def validation_losses(self):
        """Return a list of validation losses for each network in the ensemble."""
        losses = [network.dir.validation_loss[()] for network in self.values()]
        return losses

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
                self.names = [*_context_best_names, *_context_worst_names]

                if self.names:  # to prevent an empty index
                    self._model_index = np.array(
                        [i for i, name in enumerate(_names) if name in self.names]
                    )
                    self._model_mask = np.zeros(len(_names), dtype=bool)
                    self._model_mask[self._model_index] = True
            try:
                yield
            finally:
                self.names = list(_names)
                self._model_mask = np.ones(len(self)).astype(bool)
                self._model_index = np.arange(len(self))
                self._best_ratio = 0.5
                self._worst_ratio = 0.5

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
        error = self.validation_losses()

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

    def cluster_indices(self, cell_type: str) -> Dict[int, NDArray[int]]:
        """Clusters from responses to naturalistic stimuli of the given cell type.

        Args:
            cell_type: The cell type to return the clusters for.

        Returns: keys are the cluster ids and the values are the model indices
            in the ensemble.

        Note, this method accesses precomputed clusters.

        Example:
            ensemble = Ensemble("path/to/ensemble")
            cluster_indices = ensemble.cluster_indices("T4a")
            first_cluster = ensemble[cluster_indices[0]]
        """
        if not self.dir.clustering:
            raise ValueError("clustering not available")
        if cell_type not in list(self.dir.clustering.keys()):
            raise ValueError(f"cell type {cell_type} not available")
        cluster_indices = self.dir.clustering[cell_type].to_dict()

        _models = sorted(
            np.concatenate(list(self.dir.clustering[cell_type].to_dict().values()))
        )
        if len(_models) != len(self) or not np.all(_models == np.arange(len(self))):
            raise ValueError("stored clustering does not match ensemble")

        return dict(
            sorted(
                {
                    int(k): np.sort(v)
                    for k, v in cluster_indices.items()
                    if k != "masked"
                }.items()
            )
        )


class EnsembleDir(Directory):
    """A directory that contains a collection of trained networks."""

    pass


class EnsembleView(Ensemble):
    """A view of an ensemble of trained networks."""

    def __init__(self, path: Union[PathLike, Iterable, EnsembleDir, Ensemble]):
        if isinstance(path, Ensemble):
            super().__init__(path.dir)
        else:
            super().__init__(path)

    def loss_histogram(
        self,
        bins=None,
        fill=False,
        histtype="step",
        figsize=[1, 1],
        fontsize=5,
        fig=None,
        ax=None,
    ):
        """Plot a histogram of the validation losses of the ensemble."""
        losses = self.validation_losses()
        fig, ax = init_plot(figsize=figsize, fontsize=fontsize, fig=fig, ax=ax)
        ax.hist(
            losses,
            bins=bins if bins is not None else len(self),
            linewidth=0.5,
            fill=fill,
            histtype=histtype,
        )
        ax.set_xlabel("task error", fontsize=fontsize)
        ax.set_ylabel("number models", fontsize=fontsize)
        return fig, ax


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
        str(path).replace(str(results_dir) + "/", "") for path in model_paths
    ]
    ensemble_name = ", ".join(np.unique([n[:-4] for n in model_names]).tolist())
    return model_names, ensemble_name


def model_paths_from_names_or_paths(paths):
    """Return a list of model paths and an ensemble path from a list of model names or paths."""
    model_paths = []
    _ensemble_paths = []
    for path in paths:
        if isinstance(path, str):
            # assuming task/ensemble_id/model_id
            if len(path.split("/")) == 3:
                model_paths.append(results_dir / path)
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
