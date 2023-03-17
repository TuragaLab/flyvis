from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Tuple, Union
import logging
from os import PathLike
from copy import deepcopy
from matplotlib.colors import Colormap, Normalize
from matplotlib import cm
import torch
import numpy as np
from contextlib import contextmanager
from datamate import Directory

from flyvision import plots, results_dir
from flyvision.network import NetworkView, NetworkDir
from flyvision.plots.plt_utils import init_plot
from flyvision.utils.nn_utils import simulation

logging = logging.getLogger()


class Ensemble(dict):
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

        self.dir = Directory(self.path)

        # rank by validation error by default
        self.names = sorted(
            self.keys(),
            key=lambda key: dict(zip(self.keys(), self.validation_losses()))[key],
            reverse=False,
        )

    def __truediv__(self, key):
        return self.__getitem__(key)

    def __getitem__(
        self, key: Union[str, int, slice, np.ndarray, list]
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

    def yield_networks(self, checkpoint="best_chkpt"):
        network = self[0].init_network(chkpt=checkpoint)
        for network_view in self.values():
            yield network_view.init_network(chkpt=checkpoint, network=network)

    def yield_decoders(self, checkpoint="best_chkpt"):
        decoder = self[0].init_decoder(chkpt=checkpoint)
        for network_view in self.values():
            yield network_view.init_decoder(chkpt=checkpoint, decoder=decoder)

    def validation_losses(self):
        losses = [network.dir.validation_loss[()] for network in self.values()]
        return losses

    @contextmanager
    def rank_by_validation_error(self, reverse=False):
        """
        To temporarily sort the ensemble based on a type of task error. In
        ascending order.

        This method sorts the self.names attribute temporarily, which serve as
        references to the Datawrap instances of the trained Networks.
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
    def model_ratio(self, best=None, worst=None):
        """To sort and filter the ensemble temporarily by a ratio of models that
        are performing good or bad based on a type of task error."""
        # no-op
        if best is None and worst is None:
            yield
            return

        _names = tuple(self.names)
        # because even if new stimuli are initialized, these are only for a
        # subset of the models. but then these are not necessarily true anymore.
        # needs to track which field changes and null it.
        # _initialized = deepcopy(self._initialized)

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
                # a subset of models can initialize stimuli
                # responses of one type from different configs. this will
                # change self._initialized. if different models have differently
                # configured responses of the same type loaded, this would lead
                # to inconsistent analyses. this is to reset the initialization.
                # for key in _initialized:
                #     if _initialized[key] != self._initialized[key]:
                #         _initialized[key] = ""
                # self._initialized = _initialized
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
    ):

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


class EnsembleDir(Directory):
    pass


class EnsembleView(Ensemble):
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

    def simulate(self, movie_input: torch.Tensor, dt: float):
        for network in self.yield_networks():
            yield network.simulate(movie_input, dt).cpu().numpy()

    def decode(self, movie_input, dt):
        responses = torch.tensor(list(self.simulate(movie_input, dt)))
        for i, decoder in enumerate(self.yield_decoders()):
            with simulation(decoder):
                yield decoder(responses[i]).cpu().numpy()

    def cluster_indices(self, cell_type):
        cluster_indices = self.dir.clustering[cell_type].to_dict()
        return dict(sorted({int(k): v for k, v in cluster_indices.items()}.items()))


def model_paths_from_parent(path):
    model_paths = sorted(
        filter(
            lambda p: p.name.isnumeric() and p.is_dir(),
            path.iterdir(),
        )
    )
    return model_paths, path


def model_path_names(model_paths):
    model_names = [
        str(path).replace(str(results_dir) + "/", "") for path in model_paths
    ]
    ensemble_name = ", ".join(np.unique([n[:-4] for n in model_names]).tolist())
    return model_names, ensemble_name


def model_paths_from_names_or_paths(paths):
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
    values: np.ndarray
    colors: np.ndarray
    cmap: Colormap
    norm: Normalize
    scalarmappable: cm.ScalarMappable
