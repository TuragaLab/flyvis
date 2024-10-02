"""Ensemble of trained networks."""

from __future__ import annotations

import logging
from functools import wraps
from os import PathLike
from typing import Callable, Iterable, List, Union

import numpy as np
from matplotlib import colormaps as cm
from torch import nn

import flyvision
from flyvision.analysis.flash_responses import flash_response_index, plot_fris
from flyvision.analysis.moving_bar_responses import direction_selectivity_index, plot_dsis
from flyvision.analysis.visualization import plots
from flyvision.connectome import flyvision_connectome
from flyvision.utils.chkpt_utils import (
    best_checkpoint_default_fn,
    recover_network,
    resolve_checkpoints,
)

from .directories import EnsembleDir
from .ensemble import Ensemble
from .network import Network

logging = logging.getLogger(__name__)

__all__ = ["EnsembleView"]


class EnsembleView(Ensemble):
    """A view of an ensemble of trained networks."""

    def __init__(
        self,
        path: Union[str, PathLike, Iterable, EnsembleDir, Ensemble],
        network_class: nn.Module = Network,
        root_dir: PathLike = flyvision.results_dir,
        connectome_getter: Callable = flyvision_connectome,
        checkpoint_mapper: Callable = resolve_checkpoints,
        best_checkpoint_fn: Callable = best_checkpoint_default_fn,
        best_checkpoint_fn_kwargs: dict = {
            "validation_subdir": "validation",
            "loss_file_name": "loss",
        },
        recover_fn: Callable = recover_network,
        try_sort=False,
    ):
        init_args = {
            "path": path,
            "network_class": network_class,
            "root_dir": root_dir,
            "connectome_getter": connectome_getter,
            "checkpoint_mapper": checkpoint_mapper,
            "best_checkpoint_fn": best_checkpoint_fn,
            "best_checkpoint_fn_kwargs": best_checkpoint_fn_kwargs,
            "recover_fn": recover_fn,
            "try_sort": try_sort,
        }
        if isinstance(path, Ensemble):
            init_args = path._init_args
        super().__init__(**init_args)

    @wraps(plots.loss_curves)
    def training_loss(self, **kwargs):
        task_error = self.task_error()
        losses = np.array([nv.dir.loss[:] for nv in self.values()])
        return plots.loss_curves(
            losses,
            cbar=True,
            colors=task_error.colors,
            cmap=task_error.cmap,
            norm=task_error.norm,
            xlabel="iterations",
            ylabel="training loss",
            **kwargs,
        )

    @wraps(plots.loss_curves)
    def validation_loss(self, validation_subdir=None, loss_file_name=None, **kwargs):
        task_error = self.task_error()
        losses = self.validation_losses(validation_subdir, loss_file_name)
        return plots.loss_curves(
            losses,
            cbar=True,
            colors=task_error.colors,
            cmap=task_error.cmap,
            norm=task_error.norm,
            xlabel="checkpoints",
            ylabel="validation loss",
            **kwargs,
        )

    @wraps(plots.histogram)
    def task_error_histogram(self, **kwargs):
        """Plot a histogram of the validation losses of the ensemble."""
        losses = self.min_validation_losses()
        return plots.histogram(
            losses, xlabel="task error", ylabel="number models", **kwargs
        )

    @wraps(plots.violins)
    def node_parameters(self, key, max_per_ax=34, **kwargs):
        """Return the node parameters of the ensemble."""
        parameters = self.parameters()[f"nodes_{key}"]
        parameter_keys = self.parameter_keys()[f"nodes_{key}"]
        return plots.violins(
            parameter_keys, parameters, ylabel=key, max_per_ax=max_per_ax, **kwargs
        )

    @wraps(plots.violins)
    def edge_parameters(self, key, max_per_ax=120, **kwargs):
        """Return the edge parameters of the ensemble."""
        parameters = self.parameters()[f"edges_{key}"]
        parameter_keys = self.parameter_keys()[f"edges_{key}"]
        variable_names = np.array([
            f"{source}->{target}" for source, target in parameter_keys
        ])
        return plots.violins(
            variable_names,
            variable_values=parameters,
            ylabel=key,
            max_per_ax=max_per_ax,
            **kwargs,
        )

    @wraps(plots.heatmap)
    def dead_or_alive(self, **kwargs):
        """Return the number of dead cells in the ensemble."""
        responses = self.naturalistic_stimuli_responses()
        dead_count = (responses['responses'].values < 0).all(axis=(1, 2))
        return plots.heatmap(
            dead_count,
            ylabels=np.arange(len(self)),
            xlabels=responses.cell_type.values,
            size_scale=15,
            cbar=False,
            **kwargs,
        )

    @wraps(plot_fris)
    def flash_response_index(self, cell_types: List[str] = None, **kwargs):
        """Plot the flash response indices of the ensemble."""
        responses = self.flash_responses()
        fris = flash_response_index(responses, radius=6)
        if cell_types is not None:
            fris = fris.custom.where(cell_type=cell_types)
        else:
            cell_types = fris.cell_type.values
        task_error = self.task_error()
        best_index = np.argmin(task_error.values)
        return plot_fris(
            fris.values,
            cell_types,
            scatter_best=True,
            scatter_best_index=best_index,
            scatter_best_color=cm.get_cmap("Blues")(1.0),
            **kwargs,
        )

    @wraps(plot_dsis)
    def direction_selectivity_index(self, **kwargs):
        """Plot the direction selectivity indices of the ensemble."""
        responses = self.movingedge_responses()
        dsis = direction_selectivity_index(responses)
        task_error = self.task_error()
        best_index = np.argmin(task_error.values)
        return plot_dsis(
            dsis,
            responses.cell_type,
            bold_output_type_labels=False,
            figsize=[10, 1.2],
            color_known_types=True,
            fontsize=6,
            scatter_best_index=best_index,
            scatter_best_color=cm.get_cmap("Blues")(1.0),
            **kwargs,
        )
