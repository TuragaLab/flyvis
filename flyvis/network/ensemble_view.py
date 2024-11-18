"""Ensemble of trained networks."""

from __future__ import annotations

import logging
from functools import wraps
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps as cm
from torch import nn

import flyvis
from flyvis.analysis.flash_responses import flash_response_index, plot_fris
from flyvis.analysis.moving_bar_responses import (
    direction_selectivity_index,
    dsi_violins_on_and_off,
)
from flyvis.analysis.visualization import plots
from flyvis.connectome import get_avgfilt_connectome
from flyvis.utils.chkpt_utils import (
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
    """A view of an ensemble of trained networks.

    This class extends the Ensemble class with visualization and analysis methods.

    Args:
        path: Path to the ensemble directory or an existing Ensemble object.
        network_class: The network class to use for instantiation.
        root_dir: Root directory for results.
        connectome_getter: Function to get the connectome.
        checkpoint_mapper: Function to resolve checkpoints.
        best_checkpoint_fn: Function to select the best checkpoint.
        best_checkpoint_fn_kwargs: Keyword arguments for best_checkpoint_fn.
        recover_fn: Function to recover the network.
        try_sort: Whether to try sorting the ensemble.

    Attributes:
        Inherits all attributes from the Ensemble class.
    """

    def __init__(
        self,
        path: Union[str, Path, Iterable, EnsembleDir, Ensemble],
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
        init_args = (
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
        if isinstance(path, Ensemble):
            init_args = path._init_args
        super().__init__(*init_args)

    @wraps(plots.loss_curves)
    def training_loss(self, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Plot training loss curves for the ensemble.

        Args:
            **kwargs: Additional keyword arguments to pass to plots.loss_curves.

        Returns:
            A tuple containing the matplotlib Figure and Axes objects.
        """
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
    def validation_loss(
        self,
        validation_subdir: Optional[str] = None,
        loss_file_name: Optional[str] = None,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot validation loss curves for the ensemble.

        Args:
            validation_subdir: Subdirectory containing validation data.
            loss_file_name: Name of the loss file.
            **kwargs: Additional keyword arguments to pass to plots.loss_curves.

        Returns:
            A tuple containing the matplotlib Figure and Axes objects.
        """
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
    def task_error_histogram(self, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a histogram of the validation losses of the ensemble.

        Args:
            **kwargs: Additional keyword arguments to pass to plots.histogram.

        Returns:
            A tuple containing the matplotlib Figure and Axes objects.
        """
        losses = self.min_validation_losses()
        return plots.histogram(
            losses, xlabel="task error", ylabel="number models", **kwargs
        )

    @wraps(plots.violins)
    def node_parameters(
        self, key: str, max_per_ax: int = 34, **kwargs
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Plot violin plots of node parameters for the ensemble.

        Args:
            key: The parameter key to plot.
            max_per_ax: Maximum number of violins per axis.
            **kwargs: Additional keyword arguments to pass to plots.violins.

        Returns:
            A tuple containing the matplotlib Figure and a list of Axes objects.
        """
        parameters = self.parameters()[f"nodes_{key}"]
        parameter_keys = self.parameter_keys()[f"nodes_{key}"]
        return plots.violins(
            parameter_keys, parameters, ylabel=key, max_per_ax=max_per_ax, **kwargs
        )

    @wraps(plots.violins)
    def edge_parameters(
        self, key: str, max_per_ax: int = 120, **kwargs
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Plot violin plots of edge parameters for the ensemble.

        Args:
            key: The parameter key to plot.
            max_per_ax: Maximum number of violins per axis.
            **kwargs: Additional keyword arguments to pass to plots.violins.

        Returns:
            A tuple containing the matplotlib Figure and a list of Axes objects.
        """
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
    def dead_or_alive(self, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a heatmap of dead cells in the ensemble.

        Args:
            **kwargs: Additional keyword arguments to pass to plots.heatmap.

        Returns:
            A tuple containing the matplotlib Figure and Axes objects.
        """
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
    def flash_response_index(
        self, cell_types: Optional[List[str]] = None, **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the flash response indices of the ensemble.

        Args:
            cell_types: List of cell types to include. If None, all cell types are used.
            **kwargs: Additional keyword arguments to pass to plot_fris.

        Returns:
            A tuple containing the matplotlib Figure and Axes objects.
        """
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

    @wraps(dsi_violins_on_and_off)
    def direction_selectivity_index(
        self, **kwargs
    ) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
        """Plot the direction selectivity indices of the ensemble.

        Args:
            **kwargs: Additional keyword arguments to pass to plot_dsis.

        Returns:
            A tuple containing the matplotlib Figure and a tuple of Axes objects.
        """
        responses = self.moving_edge_responses()
        dsis = direction_selectivity_index(responses)
        task_error = self.task_error()
        best_index = np.argmin(task_error.values)
        return dsi_violins_on_and_off(
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
