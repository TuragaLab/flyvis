"""Ensemble of trained networks."""

from __future__ import annotations

import logging
from functools import wraps
from os import PathLike
from typing import Iterable, List, Union

import numpy as np
import torch
from matplotlib import colormaps as cm

from flyvision import device, plots
from flyvision.analysis import views
from flyvision.analysis.flash_responses import FlashResponseView, plot_fris
from flyvision.analysis.moving_bar_responses import MovingEdgeResponseView, plot_dsis
from flyvision.datasets.flashes import Flashes
from flyvision.datasets.moving_bar import MovingEdge
from flyvision.directories import EnsembleDir
from flyvision.ensemble import Ensemble
from flyvision.initialization import Parameter
from flyvision.utils.activity_utils import CellTypeArray
from flyvision.utils.class_utils import forward_subclass

logging = logging.getLogger(__name__)


class EnsembleView(Ensemble):
    """A view of an ensemble of trained networks."""

    def __init__(
        self,
        path: Union[str, PathLike, Iterable, EnsembleDir, Ensemble],
        checkpoint="best",
        validation_subdir="validation",
        loss_file_name="loss",
        try_sort=False,
    ):
        if isinstance(path, Ensemble):
            path, checkpoint, validation_subdir, loss_file_name, try_sort = (
                path._init_args
            )
        super().__init__(path, checkpoint, validation_subdir, loss_file_name, try_sort)

    @wraps(views.loss_curves)
    def training_loss(self, **kwargs):
        task_error = self.task_error()
        losses = np.array([nv.dir.loss[:] for nv in self.values()])
        return views.loss_curves(
            losses,
            cbar=True,
            colors=task_error.colors,
            cmap=task_error.cmap,
            norm=task_error.norm,
            xlabel="iterations",
            ylabel="training loss",
            **kwargs,
        )

    @wraps(views.loss_curves)
    def validation_loss(self, **kwargs):
        task_error = self.task_error()
        losses = np.array([
            nv.dir[nv.checkpoints.validation_subdir][nv.checkpoints.loss_file_name][:]
            for nv in self.values()
        ])
        return views.loss_curves(
            losses,
            cbar=True,
            colors=task_error.colors,
            cmap=task_error.cmap,
            norm=task_error.norm,
            xlabel="checkpoints",
            ylabel="validation loss",
            **kwargs,
        )

    @wraps(views.histogram)
    def task_error_histogram(self, **kwargs):
        """Plot a histogram of the validation losses of the ensemble."""
        losses = self.validation_losses()
        return views.histogram(
            losses, xlabel="task error", ylabel="number models", **kwargs
        )

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

    @wraps(views.violins)
    def node_parameters(self, key, max_per_ax=34, **kwargs):
        """Return the node parameters of the ensemble."""
        parameters = self.parameters()[f"nodes_{key}"]
        parameter_keys = self.parameter_keys()[f"nodes_{key}"]
        return views.violins(
            parameter_keys, parameters, ylabel=key, max_per_ax=max_per_ax, **kwargs
        )

    @wraps(views.violins)
    def edge_parameters(self, key, max_per_ax=120, **kwargs):
        """Return the edge parameters of the ensemble."""
        parameters = self.parameters()[f"edges_{key}"]
        parameter_keys = self.parameter_keys()[f"edges_{key}"]
        variable_names = np.array([
            f"{source}->{target}" for source, target in parameter_keys
        ])
        return views.violins(
            variable_names,
            variable_values=parameters,
            ylabel=key,
            max_per_ax=max_per_ax,
            **kwargs,
        )

    @wraps(plots.plots.heatmap)
    def dead_or_alive(self, **kwargs):
        """Return the number of dead cells in the ensemble."""
        network_view = self[0]
        chkpts = network_view.checkpoints

        if chkpts.choice == "best":
            subdirchild = f"chkpt_best_{chkpts.validation_subdir}_{chkpts.loss_file_name}"
        else:
            subdirchild = f"chkpt_{chkpts.index}"

        path = f"naturalistic_stimuli_responses/{subdirchild}"

        dead_count = np.zeros([len(self), len(network_view.cell_types_sorted)])
        for i, network_view in enumerate(self.values()):
            dead_count[i] = (
                network_view.dir[path].network_states.nodes.activity_central[:] < 0
            ).all(axis=(0, 1))

        return plots.plots.heatmap(
            dead_count,
            ylabels=np.arange(len(self)),
            xlabels=network_view.cell_types_unsorted,
            size_scale=15,
            cbar=False,
            **kwargs,
        )

    @wraps(plot_fris)
    def flash_response_indices(
        self, subdir="flash_responses", cell_types: List[str] = None, **kwargs
    ):
        """Plot the flash response indices of the ensemble."""
        dataset = Flashes(
            dynamic_range=[0, 1],
            t_stim=1.0,
            t_pre=1.0,
            dt=1 / 200,
            radius=[-1, 6],
            alternations=(0, 1, 0),
        )
        central_cells_index = self[0].connectome.central_cells_index[:]

        responses = self.stored_responses(subdir, central=True)
        if responses is None:
            logging.info("Computing responses.")
            responses = np.stack(
                [
                    resp[:, :, central_cells_index].copy()
                    for resp in self.simulate(
                        torch.stack(dataset[:]).unsqueeze(2).to(device), dt=dataset.dt
                    )
                ],
                axis=0,
            )
        responses_cta = CellTypeArray(
            responses,
            cell_types=self[0].connectome.unique_cell_types[:].astype(str),
        )
        if cell_types is not None:
            responses_cta = responses_cta.from_cell_types(cell_types)
        frv = FlashResponseView(
            arg_df=dataset.arg_df,
            config=dataset.config,
            responses=responses_cta,
            stim_sample_dim=1,
            temporal_dim=2,
        )
        fri_all = frv.fri(radius=6)
        fris = fri_all.responses.array.squeeze()
        cell_types = fri_all.responses.cell_types
        task_error = self.task_error()
        best_index = np.argmin(task_error.values)
        return plot_fris(
            fris,
            cell_types,
            scatter_best_index=best_index,
            scatter_best_color=cm.get_cmap("Blues")(1.0),
            **kwargs,
        )

    @wraps(plot_dsis)
    def direction_selectivity_indices(self, subdir="movingedge_responses", **kwargs):
        """Plot the direction selectivity indices of the ensemble."""
        responses = self.stored_responses(subdir, central=True)
        if responses is None:
            dataset = MovingEdge(
                offsets=[-10, 11],
                intensities=[0, 1],
                speeds=[9.7, 13, 19, 25],
                height=80,
                post_pad_mode="continue",
                t_pre=1.0,
                t_post=1.0,
                dt=1 / 200,
                angles=list(np.arange(0, 360, 30)),
            )
            logging.info("Computing responses for a subset of stimuli.")
            responses = np.stack(
                list(
                    self.simulate_from_dataset(
                        dataset,
                        dt=dataset.dt,
                        batch_size=4,
                        central_cell_only=True,
                    )
                )
            )
        else:
            # default dataset
            # TODO: this is a hack, we should be able to get the dataset in a more
            # principled way
            config = next(iter(self[0].movingedge_responses())).config
            dataset = MovingEdge(**config)

        responses_cta = CellTypeArray(
            responses,
            cell_types=self[0].connectome.unique_cell_types[:].astype(str),
        )
        merv = MovingEdgeResponseView(
            arg_df=dataset.arg_df,
            responses=responses_cta,
            config=dataset.config,
            stim_sample_dim=1,
            temporal_dim=2,
        )

        # compute FRIs for all cell types
        dsi_all = merv.dsi()
        # get FRI values and corresponding cell types
        dsis = dsi_all.responses.array.squeeze()
        cell_types = dsi_all.responses.cell_types
        task_error = self.task_error()
        best_index = np.argmin(task_error.values)
        return plot_dsis(
            dsis,
            cell_types,
            bold_output_type_labels=False,
            figsize=[10, 1.2],
            color_known_types=True,
            fontsize=6,
            scatter_best_index=best_index,
            scatter_best_color=cm.get_cmap("Blues")(1.0),
            **kwargs,
        )
