"""
Analysis of currents in response to moving edges.

Info:
    Relies on dataclass defined in `flyvis.analysis.stimulus_responses_currents`.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datamate import Namespace
from matplotlib.axes import Axes
from matplotlib.colors import hex2color
from matplotlib.lines import Line2D
from toolz import valfilter, valmap

from flyvis.connectome import ReceptiveFields
from flyvis.datasets.moving_bar import MovingEdge
from flyvis.utils.color_utils import (
    ND,
    PD,
    adapt_color_alpha,
    cmap_iter,
    truncate_colormap,
)
from flyvis.utils.df_utils import where_dataframe as get_stimulus_index
from flyvis.utils.nodes_edges_utils import CellTypeArray

from .stimulus_responses_currents import ExperimentData
from .visualization import plots, plt_utils
from .visualization.figsize_utils import (
    cm_to_inch,
    figsize_from_n_items,
)

__all__ = ["MovingEdgeCurrentView"]


class MovingEdgeCurrentView:
    """Represents a view of moving edge currents for analysis and visualization.

    This class provides methods for analyzing and visualizing currents and responses
    related to moving edge stimuli in neural simulations.

    Args:
        ensemble: The ensemble of models.
        target_type: The type of target cell.
        exp_data: Experimental data.
        arg_df: DataFrame containing stimulus arguments.
        currents: Currents for each source type.
        rfs: Receptive fields for the target cells.
        time: Time array for the simulation.
        responses: Responses of the target cells.

    Attributes:
        target_type: The type of target cell.
        ensemble: The ensemble of models.
        config: Configuration settings.
        arg_df: DataFrame containing stimulus arguments.
        rfs: Receptive fields for the target cells.
        exp_data: Experimental data.
        source_types: Types of source cells.
        time: Time array for the simulation.
        currents: Currents for each source type.
        responses: Responses of the target cells.

    Note:
        This class is intended to be updated to use xarray datasets in the future.
    """

    def __init__(
        self,
        ensemble,
        target_type: str,
        exp_data: List[ExperimentData],
        arg_df: pd.DataFrame | None = None,
        currents: Namespace | None = None,
        rfs: ReceptiveFields | None = None,
        time: np.ndarray | None = None,
        responses: np.ndarray | None = None,
    ):
        self.target_type = target_type
        self.ensemble = ensemble
        self.config = exp_data[0].config
        if arg_df is None:
            self.arg_df = MovingEdge(**exp_data[0].config).arg_df
        else:
            self.arg_df = arg_df
        self.rfs = rfs or reset_index(
            ReceptiveFields(target_type, ensemble[0].connectome.edges.to_df())
        )
        self.exp_data = exp_data
        self.source_types = self.rfs.source_types
        self.time = time
        self.init_currents(currents)
        self.init_time(time)
        self.init_responses(responses)

    def init_currents(self, currents: Namespace | None) -> None:
        """Initialize the currents for each source type.

        Args:
            currents: Currents for each source type.
        """
        if currents is not None:
            self.currents = currents
            return
        self.currents = Namespace()
        for source_type in self.rfs.source_types:
            # (on/off, n_models, n_angles, n_timesteps, n_input_cells)
            self.currents[source_type] = np.array(
                [
                    np.array(exp.target_data[self.target_type].source_data[source_type])
                    for exp in self.exp_data
                ],
            )

    def init_responses(self, responses: np.ndarray | None) -> None:
        """Initialize the responses of the target cells.

        Args:
            responses: Responses of the target cells.
        """
        if responses is not None:
            self.responses = responses
            return
        # (on/off, n_models, n_angles, n_timesteps)
        self.responses = np.array(
            [
                np.array(exp.target_data[self.target_type].activity_central)
                for exp in self.exp_data
            ],
        )

    def init_time(self, time: np.ndarray | None) -> None:
        """Initialize the time array for the simulation.

        Args:
            time: Time array for the simulation.
        """
        if time is not None:
            self.time = time
            return
        self.time = self.time or (
            np.arange(0, next(iter(self.currents.values())).shape[-2]) * self.config.dt
            - self.config.t_pre
        )

    @property
    def on(self) -> "MovingEdgeCurrentView":
        """Return a view of the ON responses."""
        on_index = get_stimulus_index(self.arg_df, intensity=0)
        arg_df = self.arg_df.iloc[on_index]
        return self.view(
            Namespace({
                cell_type: np.take(c, indices=on_index, axis=1)
                for cell_type, c in self.currents.items()
            }),
            responses=np.take(self.responses, indices=on_index, axis=1),
            arg_df=arg_df,
        )

    @property
    def off(self) -> "MovingEdgeCurrentView":
        """Return a view of the OFF responses."""
        off_index = get_stimulus_index(self.arg_df, intensity=0)
        arg_df = self.arg_df.iloc[off_index]
        return self.view(
            Namespace({
                cell_type: np.take(c, indices=off_index, axis=1)
                for cell_type, c in self.currents.items()
            }),
            responses=np.take(self.responses, indices=off_index, axis=1),
            arg_df=arg_df,
        )

    def divide_by_given_norm(self, norm: CellTypeArray) -> "MovingEdgeCurrentView":
        """Divide currents and responses by a given norm.

        Args:
            norm: The norm to divide by.

        Returns:
            A new view with normalized currents and responses.

        Raises:
            ValueError: If norm is not a CellTypeArray.
        """
        if not isinstance(norm, CellTypeArray):
            raise ValueError

        response_dims = np.arange(len(self.responses.shape))
        response_norm = np.expand_dims(
            norm[self.target_type].squeeze(), list(set(response_dims) - set([0]))
        )

        # divide the responses by the norm
        new_responses = self.responses[:] / response_norm

        # note: we also divide by the norm of the target cell type

        currents_dims = np.arange(len(next(iter(self.currents.values())).shape))

        currents_norm = np.expand_dims(
            norm[self.target_type].squeeze(), list(set(currents_dims) - set([0]))
        )

        # divide the currents by the norm
        new_currents = Namespace({
            cell_type: c / currents_norm for cell_type, c in self.currents.items()
        })
        return self.view(currents=new_currents, responses=new_responses)

    def at_contrast(self, contrast: float) -> "MovingEdgeCurrentView":
        """Create a new view filtered by contrast.

        Args:
            contrast: The contrast value to filter by.

        Returns:
            A new view with data filtered by the specified contrast.
        """
        contrast_index = get_stimulus_index(self.arg_df, intensity=contrast)
        arg_df = self.arg_df.iloc[contrast_index]
        return self.view(
            Namespace({
                cell_type: np.take(c, indices=contrast_index, axis=1)
                for cell_type, c in self.currents.items()
            }),
            responses=np.take(self.responses, indices=contrast_index, axis=1),
            arg_df=arg_df,
        )

    def at_angle(self, angle: float) -> "MovingEdgeCurrentView":
        """Create a new view filtered by angle.

        Args:
            angle: The angle value to filter by.

        Returns:
            A new view with data filtered by the specified angle.
        """
        angle_index = get_stimulus_index(self.arg_df, angle=angle)
        arg_df = self.arg_df.iloc[angle_index]
        return self.view(
            Namespace({
                cell_type: np.take(c, indices=angle_index, axis=1)
                for cell_type, c in self.currents.items()
            }),
            responses=np.take(self.responses, indices=angle_index, axis=1),
            arg_df=arg_df,
        )

    def at_position(
        self, u: float | None = None, v: float | None = None, central: bool = True
    ) -> "MovingEdgeCurrentView":
        """Create a new view filtered by position.

        Args:
            u: The u-coordinate.
            v: The v-coordinate.
            central: Whether to use central position.

        Returns:
            A new view with data filtered by the specified position.
        """
        rfs = at_position(self.rfs, u, v, central)
        currents = Namespace({
            cell_type: c[:, :, :, :, rfs[cell_type].index]
            for cell_type, c in self.currents.items()
        })
        return self.view(currents, rfs=rfs)

    def between_seconds(self, t_start: float, t_end: float) -> "MovingEdgeCurrentView":
        """Create a new view filtered by time range.

        Args:
            t_start: Start time in seconds.
            t_end: End time in seconds.

        Returns:
            A new view with data filtered by the specified time range.
        """
        slice = np.where((self.time >= t_start) & (self.time < t_end))[0]
        newview = self[:, :, slice, :]
        newview.time = self.time[slice]
        newview.responses = self.responses[:, :, slice]
        return newview

    def model_selection(self, mask: np.ndarray) -> "MovingEdgeCurrentView":
        """Create a new view with selected models.

        Args:
            mask: Boolean mask for model selection.

        Returns:
            A new view with selected models.
        """
        return self[mask, :, :, :]

    def __getattr__(self, key):
        return self.__getitem__(key)

    def __getitem__(self, key) -> Union["MovingEdgeCurrentView", Any]:
        # e.g. view.C3
        if isinstance(key, str) and key in self.source_types:
            return self.view(Namespace({key: self.currents[key]}))
        # e.g. view["C3", 0, 0, 0]
        elif (
            isinstance(key, Iterable)
            and isinstance(key[0], str)
            and key[0] in self.source_types
            and len(key[1:]) == self.shape
        ):
            return self.view(self.currents[key[0]][key[1:]])
        # e.g. view[index, :, :, :]
        elif isinstance(key, Iterable) and len(key) == len(self.shape):
            return self.view(
                Namespace({cell_type: c[key] for cell_type, c in self.currents.items()}),
                responses=self.responses[key[:-1]],
            )
        # view[:]
        elif key == slice(None):
            if len(self.currents) == 1:
                return next(iter(self.currents.values()))
            return self.currents
        return object.__getattribute__(self, key)

    def __repr__(self):
        cv = {ct: v.shape for ct, v in self.currents.items()}
        formatted_cv = ",\n        ".join(
            f"'{ct}': Array(shape={v})" for ct, v in cv.items()
        )
        return (
            f"{self.__class__.__name__}(\n"
            f"    ensemble={self.ensemble.name},\n"
            f"    target_type={self.target_type},\n"
            f"    currents={{\n        {formatted_cv}\n    }},\n"
            f"    rfs={self.rfs}\n"
            f")"
        )

    @property
    def shape(self):
        return next(iter(self.currents.values())).shape

    def sorting(self, average_over_models: bool = True, mode: str = "all") -> np.ndarray:
        """Sort cell types based on their contributions.

        Args:
            average_over_models: Whether to average over models.
            mode: Sorting mode ("all", "excitatory", or "inhibitory").

        Returns:
            Sorted array of cell types.

        Raises:
            ValueError: If an invalid mode is provided.
        """
        summed = self if len(self.shape) == 4 else self.sum_over_cells()
        signs = self.signs()
        if average_over_models:
            absmax = {
                k: v * signs[k]
                for k, v in valmap(
                    lambda v: np.nanmax(
                        np.abs(np.nanmean(v, axis=1, keepdims=True)),
                        axis=(0, 2, 3),
                    ),
                    summed[:],
                ).items()
            }
        else:
            # summing over on/off, angles and time to sort -- results in n_models sortings
            absmax = {
                k: v * signs[k]
                for k, v in valmap(
                    lambda v: np.nanmax(np.abs(v), axis=(0, 2, 3)), summed[:]
                ).items()
            }
        cell_types = np.array(list(absmax.keys()))
        values = np.array(list(absmax.values()))
        sorting = np.argsort(values, axis=0).T
        #         if average_over_models:
        #             # add extra dimension here for the next operation
        #             sorting = sorting[None]
        self.sorted_cell_types = cell_types[sorting[:, ::-1]]

        # return all excitatory and inhibitory from most excitatory to most inhibitory
        if mode == "all":
            return self.sorted_cell_types
        # from most excitatory to least excitatory
        elif mode == "excitatory":
            assert average_over_models
            return np.array([
                cell_type
                for cell_type in self.sorted_cell_types[0]
                if signs[cell_type] == 1
            ])
        # from most inhibitory to least inhibitory
        elif mode == "inhibitory":
            assert average_over_models
            return np.array([
                cell_type
                for cell_type in self.sorted_cell_types[0][::-1]
                if signs[cell_type] == -1
            ])
        else:
            raise ValueError(f"mode {mode}")

    def filter_cell_types_by_contribution(
        self,
        bins: int = 3,
        cut_off_edge: int = 1,
        mode: str = "above_cut_off",
        statistic: Callable = np.max,
    ) -> np.ndarray:
        """Filter cell types based on their contribution.

        Args:
            bins: Number of bins for contribution levels.
            cut_off_edge: Edge index for cut-off.
            mode: Filtering mode ("above_cut_off" or "below_cut_off").
            statistic: Function to compute the statistic.

        Returns:
            Filtered array of cell types.

        Raises:
            ValueError: If an invalid mode is provided.

        Info:
            In principle, chunks the y-axis of the current plots into excitatory and
            inhibitory parts and each of the parts into bins. All cell types with currents
            above or below, depending on the mode, the specified bin edge are discarded.
        """
        sorting = self.sorting()[0]
        signs = self.signs()
        currents = self.sum_over_cells().currents

        filtered_cell_types = []
        for sign in [1, -1]:
            # compute the std over all inputs
            values = {
                cell_type: statistic(np.abs(currents[cell_type][:]))
                for cell_type in sorting
                if signs[cell_type] == sign
            }
            # bin into three bins
            # ala (low contribution, medium contribution, high contribution)
            counts, bins = np.histogram(list(values.values()), bins=bins)
            cut_off_value = bins[cut_off_edge]
            if mode == "above_cut_off":
                filtered_cell_types.extend(
                    list(valfilter(lambda v, cut_off=cut_off_value: v >= cut_off, values))
                )
            elif mode == "below_cut_off":
                filtered_cell_types.extend(
                    list(valfilter(lambda v, cut_off=cut_off_value: v < cut_off, values))
                )
            else:
                raise ValueError(f"mode {mode}")
        return np.array(filtered_cell_types)

    def filter_source_types(
        self,
        hide_source_types: str | list | None,
        bins: int,
        edge: int,
        mode: str,
        statistic: Callable = np.max,
    ) -> np.ndarray:
        """Filter source types based on various criteria.

        Args:
            hide_source_types: Source types to hide or "auto".
            bins: Number of bins for contribution levels.
            edge: Edge index for cut-off.
            mode: Filtering mode.
            statistic: Function to compute the statistic.

        Returns:
            Filtered array of source types.
        """
        source_types = self.sorting()[0]
        if isinstance(hide_source_types, str) and hide_source_types == "auto":
            hide_source_types = self.filter_cell_types_by_contribution(
                bins=bins, cut_off_edge=edge, mode=mode, statistic=statistic
            )

        if hide_source_types is not None:
            source_types = np.array([
                source_type
                for source_type in source_types
                if source_type not in hide_source_types
            ])
        return source_types

    def signs(self) -> dict[str, float]:
        """Compute the signs of receptive fields for each source type.

        Returns:
            Dictionary of signs for each source type.
        """
        return {ct: np.mean(self.rfs[ct].sign) for ct in self.rfs.source_types}

    def sum_over_cells(self) -> "MovingEdgeCurrentView":
        """Sum currents over cells.

        Returns:
            A new view with currents summed over cells.
        """
        return self.view(
            Namespace({
                cell_type: c.sum(axis=-1) for cell_type, c in self.currents.items()
            }),
        )

    def plot_spatial_contribution(
        self,
        source_type: str,
        t_start: float,
        t_end: float,
        mode: str = "peak",
        title: str = "{source_type} :→",
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        max_extent: float | None = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot the spatial contribution of a source type.

        Args:
            source_type: The source type to plot.
            t_start: Start time for the plot.
            t_end: End time for the plot.
            mode: Mode for calculating values ("peak", "mean", or "std").
            title: Title format string for the plot.
            fig: Existing figure to use.
            ax: Existing axes to use.
            max_extent: Maximum extent of the spatial filter.
            **kwargs: Additional keyword arguments for plt_utils.kernel.

        Returns:
            Axes object containing the plot.
        """
        current_view = kwargs.get("current_view") or (
            self.between_seconds(t_start, t_end)  # .at_contrast(contrast).at_angle(angle)
        )

        vmin = kwargs.get("vmin") or (
            np.floor(
                min(
                    0,
                    min(
                        current.mean(axis=(0, 1, 2)).min()
                        for current in list(current_view[:].values())
                    ),
                )
                * 100
            )
            / 100
        )

        vmax = kwargs.get("vmax") or (
            np.ceil(
                max(
                    0,
                    max(
                        current.mean(axis=(0, 1, 2)).max()
                        for current in list(current_view[:].values())
                    ),
                )
                * 100
            )
            / 100
        )

        u, v = current_view.rfs[source_type][["source_u", "source_v"]].values.T
        # average over models
        # (1, n_models, 1, n_timesteps, n_models) -> (n_timesteps, n_models)
        # import pdb

        # pdb.set_trace()
        values = current_view[source_type][:].mean(axis=(0, 1))
        if mode == "peak":
            values = values[
                np.argmax(np.abs(values), axis=0), np.arange(values.shape[-1])
            ]
        elif mode == "mean":
            values = np.mean(values, axis=0)
        elif mode == "std":
            signs = self.signs()
            values = signs[source_type] * np.std(values, axis=0)
        fig, ax, _ = plots.kernel(
            u,
            v,
            values,
            fill=True,
            max_extent=max_extent or current_view.rfs.max_extent,
            label=title.format(source_type=source_type),
            labelxy="auto",
            strict_sign=False,
            fig=fig,
            ax=ax,
            **kwargs,
        )
        (xmin, ymin, xmax, ymax) = ax.dataLim.extents
        ax.set_xlim(plt_utils.get_lims((xmin, xmax), 0.01))
        ax.set_ylim(plt_utils.get_lims((ymin, ymax), 0.01))

    def plot_spatial_contribution_grid(
        self,
        t_start: float,
        t_end: float,
        max_extent: float = 3,
        mode: str = "peak",
        title: str = "{source_type} :→",
        fig: plt.Figure | None = None,
        axes: np.ndarray[plt.Axes] | None = None,
        fontsize: float = 5,
        edgewidth: float = 0.125,
        title_y: float = 0.8,
        max_figure_height_cm: float = 9.271,
        panel_height_cm: float | str = "auto",
        max_figure_width_cm: float = 2.54,
        panel_width_cm: float = 2.54,
        annotate: bool = False,
        cbar: bool = False,
        hide_source_types: str | list | None = "auto",
        hide_source_types_bins: int = 5,
        hide_source_types_cut_off_edge: int = 1,
        hide_source_types_mode: str = "below_cut_off",
        max_axes: int | None = None,
        **kwargs,
    ) -> tuple[
        plt.Figure,
        np.ndarray[plt.Axes],
        tuple[plt.Colorbar, plt.Colormap, plt.Normalize, float, float],
    ]:
        """Plot a grid of spatial contributions for different source types.

        Args:
            t_start: Start time for the plot.
            t_end: End time for the plot.
            max_extent: Maximum extent of the spatial filter.
            mode: Mode for calculating values ("peak", "mean", or "std").
            title: Title format string for each subplot.
            fig: Existing figure to use.
            axes: Existing axes to use.
            fontsize: Font size for labels and titles.
            edgewidth: Width of edges in the plot.
            title_y: Y-position of the title.
            max_figure_height_cm: Maximum figure height in centimeters.
            panel_height_cm: Height of each panel in centimeters.
            max_figure_width_cm: Maximum figure width in centimeters.
            panel_width_cm: Width of each panel in centimeters.
            annotate: Whether to annotate the plots.
            cbar: Whether to add a colorbar.
            hide_source_types: Source types to hide or "auto".
            hide_source_types_bins: Number of bins for auto-hiding.
            hide_source_types_cut_off_edge: Cut-off edge for auto-hiding.
            hide_source_types_mode: Mode for auto-hiding source types.
            max_axes: Maximum number of axes to create.
            **kwargs: Additional keyword arguments for plot_spatial_contribution.

        Returns:
            Figure, axes, and colorbar information (cbar, cmap, norm, vmin, vmax).
        """
        current_view = self.between_seconds(t_start, t_end)

        vmin = (
            np.floor(
                min(
                    0,
                    min(
                        current.mean(axis=(0, 1, 2)).min()
                        for current in list(current_view[:].values())
                    ),
                )
                * 10
            )
            / 10
        )

        vmax = (
            np.ceil(
                max(
                    0,
                    max(
                        current.mean(axis=(0, 1, 2)).max()
                        for current in list(current_view[:].values())
                    ),
                )
                * 10
            )
            / 10
        )

        source_types = self.filter_source_types(
            hide_source_types,
            bins=hide_source_types_bins,
            edge=hide_source_types_cut_off_edge,
            mode=hide_source_types_mode,
        )

        if fig is None and axes is None:
            figsize = figsize_from_n_items(
                max_axes or len(source_types),
                max_figure_height_cm=max_figure_height_cm,
                panel_height_cm=(
                    max_figure_height_cm / (max_axes or len(source_types))
                    if panel_height_cm == "auto"
                    else panel_height_cm
                ),
                max_figure_width_cm=max_figure_width_cm,
                panel_width_cm=panel_width_cm,
            )
            fig, axes = figsize.axis_grid(
                unmask_n=max_axes or len(source_types), hspace=0.1, wspace=0
            )
            if max_axes is not None and len(source_types) < max_axes:
                for ax in np.array(axes).flatten():
                    if isinstance(ax, Axes):
                        ax.axis("off")

        for i, source_type in enumerate(source_types):
            self.plot_spatial_contribution(
                source_type,
                #                 contrast,
                #                 angle,
                t_start,
                t_end,
                mode=mode,
                title=title,
                fontsize=fontsize,
                edgewidth=edgewidth,
                title_y=title_y,
                fig=fig,
                ax=axes[i],
                current_view=current_view,
                vmin=vmin,
                vmax=vmax,
                annotate=annotate,
                cbar=False,
                max_extent=max_extent or current_view.rfs.max_extent,
                **kwargs,
            )

        cmap = plt.cm.seismic
        norm = plt_utils.get_norm(vmin=vmin, vmax=vmax, midpoint=0)
        if cbar:
            cbar = plt_utils.add_colorbar_to_fig(
                fig,
                width=0.01,
                height=0.25,
                fontsize=fontsize,
                cmap=cmap,
                norm=norm,
                label=f"{mode} input currents",
                n_ticks=4,
                n_decimals=1,
            )
        return fig, axes, (cbar, cmap, norm, vmin, vmax)

    def plot_spatial_filter(
        self,
        source_type: str,
        title: str = "{source_type} :→",
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        max_extent: float | None = None,
        **kwargs,
    ) -> plt.Axes:
        """Plot the spatial filter for a given source type.

        Args:
            source_type: The source type to plot.
            title: Title format string for the plot.
            fig: Existing figure to use.
            ax: Existing axes to use.
            max_extent: Maximum extent of the spatial filter.
            **kwargs: Additional keyword arguments for plt_utils.kernel.

        Returns:
            Axes object containing the plot.
        """
        filter = self.rfs

        def filter_values(rf):
            return (rf.n_syn * rf.sign).values

        vmin = kwargs.pop("vmin", None) or (
            np.floor(
                min(
                    0,
                    min(
                        min(filter_values(filter[source_type]))
                        for source_type in self.source_types
                    ),
                )
                * 100
            )
            / 100
        )

        vmax = kwargs.pop("vmax", None) or (
            np.ceil(
                max(
                    0,
                    max(
                        max(filter_values(filter[source_type]))
                        for source_type in self.source_types
                    ),
                )
                * 100
            )
            / 100
        )

        u, v = filter[source_type][["source_u", "source_v"]].values.T
        # average over models
        # (1, n_models, 1, n_timesteps, n_models) -> (n_timesteps, n_models)
        values = filter_values(filter[source_type])

        label = title.format(source_type=source_type)
        fig, ax, _ = plt_utils.kernel(
            u,
            v,
            values,
            fill=True,
            max_extent=max_extent or filter.max_extent,
            label=label,
            labelxy="auto",
            strict_sign=False,
            fig=fig,
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )
        (xmin, ymin, xmax, ymax) = ax.dataLim.extents
        ax.set_xlim(plt_utils.get_lims((xmin, xmax), 0.01))
        ax.set_ylim(plt_utils.get_lims((ymin, ymax), 0.01))
        return ax

    def plot_spatial_filter_grid(
        self,
        title: str = "{source_type} :→",
        fig: plt.Figure | None = None,
        axes: np.ndarray[plt.Axes] | None = None,
        max_extent: float | None = None,
        fontsize: float = 5,
        edgewidth: float = 0.125,
        title_y: float = 0.8,
        max_figure_height_cm: float = 9.271,
        panel_height_cm: float | str = "auto",
        max_figure_width_cm: float = 2.54,
        panel_width_cm: float = 2.54,
        annotate: bool = False,
        cbar: bool = False,
        hide_source_types: str | list | None = "auto",
        hide_source_types_bins: int = 5,
        hide_source_types_cut_off_edge: int = 1,
        hide_source_types_mode: str = "below_cut_off",
        max_axes: int | None = None,
        wspace: float = 0.0,
        hspace: float = 0.1,
        **kwargs,
    ) -> tuple[
        plt.Figure,
        np.ndarray[plt.Axes],
        tuple[plt.Colorbar, plt.Colormap, plt.Normalize, float, float],
    ]:
        """Plot a grid of spatial filters for different source types.

        Args:
            title: Title format string for each subplot.
            fig: Existing figure to use.
            axes: Existing axes to use.
            max_extent: Maximum extent of the spatial filter.
            fontsize: Font size for labels and titles.
            edgewidth: Width of edges in the plot.
            title_y: Y-position of the title.
            max_figure_height_cm: Maximum figure height in centimeters.
            panel_height_cm: Height of each panel in centimeters.
            max_figure_width_cm: Maximum figure width in centimeters.
            panel_width_cm: Width of each panel in centimeters.
            annotate: Whether to annotate the plots.
            cbar: Whether to add a colorbar.
            hide_source_types: Source types to hide or "auto".
            hide_source_types_bins: Number of bins for auto-hiding.
            hide_source_types_cut_off_edge: Cut-off edge for auto-hiding.
            hide_source_types_mode: Mode for auto-hiding source types.
            max_axes: Maximum number of axes to create.
            wspace: Width space between subplots.
            hspace: Height space between subplots.
            **kwargs: Additional keyword arguments for plot_spatial_filter.

        Returns:
            Figure, axes, and colorbar information (cbar, cmap, norm, vmin, vmax).
        """
        filter = self.rfs

        def filter_values(rf):
            return (rf.n_syn * rf.sign).values

        vmin = kwargs.pop("vmin", None) or (
            np.floor(
                min(
                    0,
                    min(
                        min(filter_values(filter[source_type]))
                        for source_type in self.source_types
                    ),
                )
                * 100
            )
            / 100
        )

        vmax = kwargs.pop("vmax", None) or (
            np.ceil(
                max(
                    0,
                    max(
                        max(filter_values(filter[source_type]))
                        for source_type in self.source_types
                    ),
                )
                * 100
            )
            / 100
        )

        source_types = self.filter_source_types(
            hide_source_types,
            bins=hide_source_types_bins,
            edge=hide_source_types_cut_off_edge,
            mode=hide_source_types_mode,
        )

        if fig is None and axes is None:
            figsize = figsize_from_n_items(
                max_axes or len(source_types),
                max_figure_height_cm=max_figure_height_cm,
                panel_height_cm=(
                    max_figure_height_cm / (max_axes or len(source_types))
                    if panel_height_cm == "auto"
                    else panel_height_cm
                ),
                max_figure_width_cm=max_figure_width_cm,
                panel_width_cm=panel_width_cm,
            )
            fig, axes = figsize.axis_grid(
                unmask_n=max_axes or len(source_types), hspace=hspace, wspace=wspace
            )
            if max_axes is not None and len(source_types) < max_axes:
                for ax in np.array(axes).flatten():
                    if isinstance(ax, Axes):
                        ax.axis("off")

        for i, source_type in enumerate(source_types):
            self.plot_spatial_filter(
                source_type,
                title=title,
                fontsize=fontsize,
                edgewidth=edgewidth,
                title_y=title_y,
                fig=fig,
                ax=axes[i],
                vmin=vmin,
                vmax=vmax,
                annotate=annotate,
                cbar=False,
                max_extent=max_extent or filter.max_extent,
                **kwargs,
            )

        cmap = plt.cm.seismic
        norm = plt_utils.get_norm(vmin=vmin, vmax=vmax, midpoint=0)
        if cbar:
            cbar = plt_utils.add_colorbar_to_fig(
                fig,
                width=0.01,
                height=0.25,
                fontsize=fontsize,
                cmap=cmap,
                norm=norm,
                label="spatial filters",
                n_ticks=4,
                n_decimals=1,
            )
        return fig, axes, (cbar, cmap, norm, vmin, vmax)

    def view(
        self,
        currents: Namespace,
        rfs: ReceptiveFields | None = None,
        time: np.ndarray | None = None,
        responses: np.ndarray | None = None,
        arg_df: pd.DataFrame | None = None,
    ) -> "MovingEdgeCurrentView":
        """
        Create a new view with the given currents, rfs, time, responses, and arg_df.

        Args:
            currents: Currents for each source type.
            rfs: Receptive fields for the target cells.
            time: Time array for the simulation.
            responses: Responses of the target cells.
            arg_df: DataFrame containing stimulus arguments.

        Returns:
            A new view with the given data.
        """
        arg_df = arg_df.reset_index(drop=True) if arg_df is not None else self.arg_df
        return MovingEdgeCurrentView(
            self.ensemble,
            self.target_type,
            self.exp_data,
            arg_df,
            currents,
            rfs if rfs is not None else self.rfs,
            time if time is not None else self.time,
            responses if responses is not None else self.responses,
        )

    def subtract_baseline(self) -> "MovingEdgeCurrentView":
        """
        Create a new view with baseline subtracted from the currents and responses.

        Returns:
            A new view with baseline subtracted data.
        """
        return self.view(
            Namespace({
                cell_type: c - np.take(c, [0], -2)
                for cell_type, c in self.currents.items()
            }),
            responses=self.responses - np.take(self.responses, [0], -1),
        )

    def subtract_mean(self) -> "MovingEdgeCurrentView":
        """
        Create a new view with mean subtracted from the currents and responses.

        Returns:
            A new view with mean subtracted data.
        """
        return self.view(
            Namespace({
                cell_type: c - np.mean(c, -2, keepdims=True)
                for cell_type, c in self.currents.items()
            }),
            responses=self.responses - np.mean(self.responses, -1, keepdims=True),
        )

    def standardize(self) -> "MovingEdgeCurrentView":
        """
        Create a new view with standardized currents and responses.

        Returns:
            A new view with standardized data.
        """
        return self.view(
            Namespace({
                cell_type: (c - np.mean(c, -2, keepdims=True))
                / (np.std(c, -2, keepdims=True) + 1e-15)
                for cell_type, c in self.currents.items()
            }),
            responses=(self.responses - np.mean(self.responses, -1, keepdims=True))
            / (np.std(self.responses, -1, keepdims=True) + 1e-15),
        )

    def standardize_over_time_and_pd_nd(
        self, t_start: float, t_end: float, pd: float
    ) -> "MovingEdgeCurrentView":
        """
        Create a new view with standardized currents and responses over time and PD/ND.

        Args:
            t_start: Start time for standardization.
            t_end: End time for standardization.
            pd: Preferred direction for standardization.

        Returns:
            A new view with standardized data.
        """
        temp = self.between_seconds(t_start, t_end).at_angle([pd, (pd - 180) % 360])
        return self.view(
            Namespace({
                cell_type: (
                    c - np.mean(temp.currents[cell_type], (-2, -3), keepdims=True)
                )
                / (np.std(temp.currents[cell_type], (-2, -3), keepdims=True) + 1e-15)
                for cell_type, c in self.currents.items()
            }),
            responses=(self.responses - np.mean(temp.responses, (-1, -2), keepdims=True))
            / (np.std(temp.responses, (-1, -2), keepdims=True) + 1e-15),
        )

    def init_colors(self, source_types: list[str]) -> None:
        """
        Initialize colors for source types.

        Args:
            source_types: List of source types.
        """
        signs = self.signs()
        signs = {cell_type: signs[cell_type] for cell_type in source_types}
        signs_reversed = {cell_type: signs[cell_type] for cell_type in source_types[::-1]}
        n_exc = len([v for v in signs.values() if v == 1])
        n_inh = len([v for v in signs.values() if v == -1])
        exc_colors_pd = cmap_iter(
            truncate_colormap(plt.cm.RdBu, minval=0.05, maxval=0.45, n=n_exc)
        )
        inh_cmap_pd = cmap_iter(
            truncate_colormap(plt.cm.RdBu_r, minval=0.05, maxval=0.45, n=n_inh)
        )
        exc_colors_nd = cmap_iter(
            truncate_colormap(plt.cm.BrBG_r, minval=0.05, maxval=0.45, n=n_exc)
        )
        inh_cmap_nd = cmap_iter(
            truncate_colormap(plt.cm.BrBG, minval=0.05, maxval=0.45, n=n_inh)
        )
        colors_pd = {}
        colors_nd = {}
        for _, (cell_type, sign) in enumerate(signs.items()):
            if sign == 1:
                # take the first half of the RdBu colormap, i.e. red
                colors_pd[cell_type] = next(exc_colors_pd)
                colors_nd[cell_type] = next(exc_colors_nd)

        for _, (cell_type, sign) in enumerate(signs_reversed.items()):
            if sign == -1:
                # take the second half of the RdBu colormap, i.e. blue
                colors_pd[cell_type] = next(inh_cmap_pd)
                colors_nd[cell_type] = next(inh_cmap_nd)
        self.colors_pd = colors_pd
        self.colors_nd = colors_nd

    def color(self, source_type: str, pd: bool = True) -> tuple[float, float, float]:
        """
        Get the color for a given source type.

        Args:
            source_type: The source type.
            pd: Whether to use PD or ND colors.

        Returns:
            The color as an RGB tuple.
        """
        if pd:
            return self.colors_pd[source_type]
        return self.colors_nd[source_type]

    def zorder(
        self,
        source_types: list[str],
        source_type: str,
        start_exc: int = 1000,
        start_inh: int = 1000,
    ) -> int:
        """
        Get the z-order for a given source type.

        Args:
            source_types: List of source types.
            source_type: The source type.
            start_exc: Starting z-order for excitatory cells.
            start_inh: Starting z-order for inhibitory cells.

        Returns:
            The z-order for the given source type.
        """
        signs = self.signs()
        signs_reversed = {cell_type: signs[cell_type] for cell_type in source_types[::-1]}

        z_order = start_exc
        for _, (cell_type, sign) in enumerate(signs.items()):
            if sign == 1:
                if cell_type == source_type:
                    return z_order
                z_order -= 10

        z_order = start_inh
        for _, (cell_type, sign) in enumerate(signs_reversed.items()):
            if sign == -1:
                if cell_type == source_type:
                    return z_order
                z_order -= 10

    def ylims(
        self, source_types: list[str] | None = None, offset: float = 0.02
    ) -> dict[str, tuple[float, float]]:
        """
        Get the y-limits for temporal contributions summed over cells.

        Args:
            source_types: List of source types to consider.
            offset: Offset for the y-limits.

        Returns:
            Y-limits for the given source types or all source types.
        """
        if source_types is not None:
            return {
                cell_type: plt_utils.get_lims(c, offset)
                for cell_type, c in self.sum_over_cells().currents.items()
                if cell_type in source_types
            }
        return plt_utils.get_lims(list(self.sum_over_cells().currents.values()), offset)

    def plot_response(
        self,
        contrast: float,
        angle: float,
        t_start: float = 0,
        t_end: float = 1,
        max_figure_height_cm: float = 1.4477,
        panel_height_cm: float = 1.4477,
        max_figure_width_cm: float = 4.0513,
        panel_width_cm: float = 4.0513,
        fontsize: float = 5,
        model_average: bool = True,
        color: tuple[float, float, float] = (0, 0, 0),
        legend: bool = False,
        hide_yaxis: bool = True,
        trim_axes: bool = True,
        quantile: float | None = None,
        scale_position: str | None = None,  # "lower left",
        scale_label: str = "{:.0f} ms",
        scale_unit: float = 1000,
        hline: bool = False,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
    ):
        """
        Plot the response to a moving edge stimulus.

        Args:
            contrast: The contrast of the stimulus.
            angle: The angle of the stimulus.
            t_start: Start time for the plot.
            t_end: End time for the plot.
            max_figure_height_cm: Maximum figure height in centimeters.
            panel_height_cm: Height of each panel in centimeters.
            max_figure_width_cm: Maximum figure width in centimeters.
            panel_width_cm: Width of each panel in centimeters.
            fontsize: Font size for labels and titles.
            model_average: Whether to plot the model average.
            color: Color for the plot.
            legend: Whether to show the legend.
            hide_yaxis: Whether to hide the y-axis.
            trim_axes: Whether to trim the axes.
            quantile: Quantile for shading.
            scale_position: Position of the scale.
            scale_label: Label format for the scale.
            scale_unit: Unit for the scale.
            hline: Whether to show a horizontal line at 0.
            fig: Existing figure to use.
            ax: Existing axes to use.

        Returns:
            Figure and axes objects.
        """
        r_pd = (
            self.at_angle(angle)
            .at_contrast(contrast)
            .between_seconds(t_start, t_end)
            .responses.squeeze(axis=-2)
        )
        r_nd = (
            self.at_angle((angle - 180) % 360)
            .at_contrast(contrast)
            .between_seconds(t_start, t_end)
            .responses.squeeze(axis=-2)
        )

        if fig is None and ax is None:
            figsize = figsize_from_n_items(
                1,
                max_figure_height_cm=max_figure_height_cm,
                panel_height_cm=panel_height_cm,
                max_figure_width_cm=max_figure_width_cm,
                panel_width_cm=panel_width_cm,
            )
            fig, axes = figsize.axis_grid(hspace=0.0, wspace=0, fontsize=fontsize)
            ax = axes[0]

        color = [hex2color(PD), hex2color(ND)] if color is None else [color, color]

        if model_average:
            fig, ax, _, _ = plots.traces(
                [r_pd.mean(axis=0), r_nd.mean(axis=0)],
                x=self.between_seconds(t_start, t_end).time,
                color=color,
                linewidth=1,
                fontsize=fontsize,
                null_line=False,
                fig=fig,
                ax=ax,
                linestyle=["solid", "dashed"],
                legend="" if not legend else [f"{self.target_type}", "null direction"],
                scale_pos=scale_position,
                scale_label=scale_label,
                scale_unit=scale_unit,
            )
        else:
            fig, ax, _, _ = plots.traces(
                r_pd,
                x=self.between_seconds(t_start, t_end).time,
                mean_color=adapt_color_alpha(color[0], 1),
                color=adapt_color_alpha(color[0], 0.5),
                linewidth=0.25,
                zorder_traces=0,
                zorder_mean=10,
                fontsize=fontsize,
                null_line=False,
                highlight_mean=True,
                fig=fig,
                ax=ax,
            )
            plots.traces(
                r_nd,
                x=self.between_seconds(t_start, t_end).time,
                mean_color=adapt_color_alpha(color[1], 1),
                color=adapt_color_alpha(color[1], 0.5),
                linewidth=0.25,
                zorder_traces=0,
                zorder_mean=10,
                fontsize=fontsize,
                null_line=False,
                highlight_mean=True,
                fig=fig,
                linestyle="dashed",
                ax=ax,
            )
        if quantile:
            quantile_pd = np.quantile(r_pd, quantile, axis=0)
            quantile_nd = np.quantile(r_nd, quantile, axis=0)
            ax.fill_between(
                self.between_seconds(t_start, t_end).time,
                quantile_pd[0],
                quantile_pd[1],
                facecolor=adapt_color_alpha(color[0], 0.1),
                edgecolor=adapt_color_alpha(color[0], 1),
                linewidth=0.25,
            )
            ax.fill_between(
                self.between_seconds(t_start, t_end).time,
                quantile_nd[0],
                quantile_nd[1],
                facecolor=adapt_color_alpha(color[1], 0.1),
                edgecolor=adapt_color_alpha(color[1], 1),
                linewidth=0.25,
                linestyle="dashed",
            )

        if hline:
            # horizontal line at 0
            ax.axhline(0, color=(0, 0, 0, 1), linewidth=0.25, zorder=-10)

        if hide_yaxis:
            plt_utils.rm_spines(ax, ("left",))
        if trim_axes:
            plt_utils.trim_axis(ax)
        if legend:
            ax.legend(
                fontsize=fontsize,
                ncols=1,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                borderaxespad=0.0,
            )
        return fig, ax

    def plot_response_pc_nc(
        self,
        contrast: float,
        angle: float,
        t_start: float = 0,
        t_end: float = 1,
        max_figure_height_cm: float = 1.4477,
        panel_height_cm: float = 1.4477,
        max_figure_width_cm: float = 4.0513,
        panel_width_cm: float = 4.0513,
        fontsize: float = 5,
        model_average: bool = True,
        color: tuple[float, float, float] = (0, 0, 0),
        legend: bool = False,
        hide_yaxis: bool = True,
        trim_axes: bool = True,
        quantile: float | None = None,
        scale_position: str | None = None,
        scale_label: str = "{:.0f} ms",
        scale_unit: float = 1000,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        hline: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the response to a moving edge stimulus with positive and negative contrasts.

        Args:
            contrast: The contrast of the stimulus.
            angle: The angle of the stimulus.
            t_start: Start time for the plot.
            t_end: End time for the plot.
            max_figure_height_cm: Maximum figure height in centimeters.
            panel_height_cm: Height of each panel in centimeters.
            max_figure_width_cm: Maximum figure width in centimeters.
            panel_width_cm: Width of each panel in centimeters.
            fontsize: Font size for labels and titles.
            model_average: Whether to plot the model average.
            color: Color for the plot.
            legend: Whether to show the legend.
            hide_yaxis: Whether to hide the y-axis.
            trim_axes: Whether to trim the axes.
            quantile: Quantile for shading.
            scale_position: Position of the scale.
            scale_label: Label format for the scale.
            scale_unit: Unit for the scale.
            fig: Existing figure to use.
            ax: Existing axes to use.
            hline: Whether to show a horizontal line at 0.

        Returns:
            Figure and axes objects.
        """
        r_pc = (
            self.at_angle(angle)
            .at_contrast(contrast)
            .between_seconds(t_start, t_end)
            .responses.squeeze(axis=-2)
        )
        r_nc = (
            self.at_angle(angle)
            .at_contrast(0 if contrast == 1 else 1)
            .between_seconds(t_start, t_end)
            .responses.squeeze(axis=-2)
        )

        if fig is None and ax is None:
            figsize = figsize_from_n_items(
                1,
                max_figure_height_cm=max_figure_height_cm,
                panel_height_cm=panel_height_cm,
                max_figure_width_cm=max_figure_width_cm,
                panel_width_cm=panel_width_cm,
            )
            fig, axes = figsize.axis_grid(hspace=0.0, wspace=0, fontsize=fontsize)
            ax = axes[0]

        color = [hex2color(PD), hex2color(ND)] if color is None else [color, color]

        if model_average:
            fig, ax, _, _ = plots.traces(
                [r_pc.mean(axis=0), r_nc.mean(axis=0)],
                x=self.between_seconds(t_start, t_end).time,
                color=color,
                linewidth=1,
                fontsize=fontsize,
                null_line=False,
                fig=fig,
                ax=ax,
                linestyle=["solid", "dotted"],
                legend="" if not legend else [f"{self.target_type}", "null contrast"],
                scale_pos=scale_position,
                scale_label=scale_label,
                scale_unit=scale_unit,
            )
        else:
            fig, ax, _, _ = plots.traces(
                r_pc,
                x=self.between_seconds(t_start, t_end).time,
                mean_color=adapt_color_alpha(color[0], 1),
                color=adapt_color_alpha(color[0], 0.5),
                linewidth=0.25,
                zorder_traces=0,
                zorder_mean=10,
                fontsize=fontsize,
                null_line=False,
                highlight_mean=True,
                fig=fig,
                ax=ax,
            )
            plots.traces(
                r_nc,
                x=self.between_seconds(t_start, t_end).time,
                mean_color=adapt_color_alpha(color[1], 1),
                color=adapt_color_alpha(color[1], 0.5),
                linewidth=0.25,
                zorder_traces=0,
                zorder_mean=10,
                fontsize=fontsize,
                null_line=False,
                highlight_mean=True,
                fig=fig,
                linestyle="dashed",
                ax=ax,
            )
        if quantile:
            quantile_pd = np.quantile(r_pc, quantile, axis=0)
            quantile_nd = np.quantile(r_nc, quantile, axis=0)
            ax.fill_between(
                self.between_seconds(t_start, t_end).time,
                quantile_pd[0],
                quantile_pd[1],
                facecolor=adapt_color_alpha(color[0], 0.1),
                edgecolor=adapt_color_alpha(color[0], 1),
                linewidth=0.25,
            )
            ax.fill_between(
                self.between_seconds(t_start, t_end).time,
                quantile_nd[0],
                quantile_nd[1],
                facecolor=adapt_color_alpha(color[1], 0.1),
                edgecolor=adapt_color_alpha(color[1], 1),
                linewidth=0.25,
                linestyle="dashed",
            )

        # horizontal line at 0
        if hline:
            ax.axhline(0, color=(0, 0, 0, 1), linewidth=0.25, zorder=-10)

        if hide_yaxis:
            plt_utils.rm_spines(ax, ("left",))
        if trim_axes:
            plt_utils.trim_axis(ax)
        if legend:
            ax.legend(
                fontsize=fontsize,
                ncols=1,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                borderaxespad=0.0,
            )
        return fig, ax

    def plot_temporal_contributions(
        self,
        contrast: float,
        angle: float,
        t_start: float = 0,
        t_end: float = 1,
        fontsize: float = 5,
        linewidth: float = 0.25,
        legend: bool = False,
        legend_standalone: bool = True,
        legend_figsize_cm: tuple[float, float] = (4.0572, 1),
        legend_n_rows: int | None = None,
        max_figure_height_cm: float = 3.3941,
        panel_height_cm: float = 3.3941,
        max_figure_width_cm: float = 4.0572,
        panel_width_cm: float = 4.0572,
        model_average: bool = True,
        highlight_mean: bool = True,  # only applies if model_average is False
        sum_exc_inh: bool = False,
        only_sum: bool = False,
        hide_source_types: str | list | None = "auto",
        hide_source_types_bins: int = 5,
        hide_source_types_cut_off_edge: int = 1,
        hide_source_types_mode: str = "below_cut_off",
        hide_yaxis: bool = True,
        trim_axes: bool = True,
        quantile: float | None = None,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        legend_ax: plt.Axes | None = None,
        hline: bool = True,
        legend_n_cols: int | None = None,
        baseline_color: tuple[float, float, float, float] | None = None,
        colors: dict[str, tuple[float, float, float, float]] | None = None,
    ):
        """
        Plot temporal contributions of different source types.

        Args:
            contrast: The contrast of the stimulus.
            angle: The angle of the stimulus.
            t_start: Start time for the plot.
            t_end: End time for the plot.
            fontsize: Font size for labels and titles.
            linewidth: Line width for traces.
            legend: Whether to show the legend.
            legend_standalone: Whether to create a standalone legend.
            legend_figsize_cm: Figure size for the standalone legend.
            legend_n_rows: Number of rows for the standalone legend.
            max_figure_height_cm: Maximum figure height in centimeters.
            panel_height_cm: Height of each panel in centimeters.
            max_figure_width_cm: Maximum figure width in centimeters.
            panel_width_cm: Width of each panel in centimeters.
            model_average: Whether to plot the model average.
            highlight_mean: Whether to highlight the mean trace.
            sum_exc_inh: Whether to sum excitatory and inhibitory contributions.
            only_sum: Whether to only plot the summed contributions.
            hide_source_types: Source types to hide or "auto".
            hide_source_types_bins: Number of bins for auto-hiding.
            hide_source_types_cut_off_edge: Cut-off edge for auto-hiding.
            hide_source_types_mode: Mode for auto-hiding source types.
            hide_yaxis: Whether to hide the y-axis.
            trim_axes: Whether to trim the axes.
            quantile: Quantile for shading.
            fig: Existing figure to use.
            ax: Existing axes to use.
            legend_ax: Existing axes for the standalone legend.
            hline: Whether to show a horizontal line at 0.
            legend_n_cols: Number of columns for the standalone legend.
            baseline_color: Color for the baseline.
            colors: Colors for each source type.

        Returns:
            Figure, axes, and legend axes objects.

        Example:
            ```
            view = MovingEdgeCurrentView(...)
            fig, ax = view.plot_temporal_contributions(
                contrast=1.0,
                angle=0,
                t_start=0,
                t_end=1,
                fontsize=5,
                linewidth=0.25,
                legend=True
            )
            ```
        """
        if fig is None and ax is None:
            figsize = figsize_from_n_items(
                1,
                max_figure_height_cm=max_figure_height_cm,
                panel_height_cm=panel_height_cm,
                max_figure_width_cm=max_figure_width_cm,
                panel_width_cm=panel_width_cm,
            )
            fig, axes = figsize.axis_grid(hspace=0.0, wspace=0, fontsize=fontsize)
            ax = axes[0]
        cv_pd = (
            self.at_contrast(contrast)
            .at_angle(angle)
            .between_seconds(t_start, t_end)
            .sum_over_cells()
        )
        cv_nd = (
            self.at_contrast(contrast)
            .at_angle((angle - 180) % 360)
            .between_seconds(t_start, t_end)
            .sum_over_cells()
        )

        source_types = (
            self.at_contrast(contrast)
            .at_angle([angle, (angle - 180) % 360])
            .between_seconds(t_start, t_end)
            .filter_source_types(
                hide_source_types,
                hide_source_types_bins,
                hide_source_types_cut_off_edge,
                hide_source_types_mode,
            )
        )

        color_source_types = (
            self.at_contrast(contrast)
            .at_angle([angle, (angle - 180) % 360])
            .between_seconds(t_start, t_end)
            .filter_source_types(
                None,
                hide_source_types_bins,
                hide_source_types_cut_off_edge,
                hide_source_types_mode,
            )
        )
        cv_pd.init_colors(color_source_types)
        cv_nd.init_colors(color_source_types)

        def plot_mean_trace(
            time, trace, label, color, zorder, linestyle="solid", ax=None, fig=None
        ):
            ax.plot(
                time,
                trace,
                label=label,
                color=color,
                zorder=zorder,
                linestyle=linestyle,
            )

        def plot_individual_traces(
            traces, time, color, zorder, label, linestyle="solid", legend=None
        ):
            if not only_sum and not model_average:
                plots.traces(
                    traces,
                    time,
                    mean_color=color,
                    color=color,
                    linewidth=linewidth,
                    zorder_traces=0,
                    zorder_mean=zorder,
                    fontsize=fontsize,
                    null_line=True,
                    highlight_mean=highlight_mean,
                    fig=fig,
                    ax=ax,
                    legend=legend or label,
                    linestyle=linestyle,
                )

        def plot_quantile(traces, time, color, zorder, linestyle="solid"):
            if quantile:
                Q = np.quantile(traces, quantile, axis=0)
                ax.fill_between(
                    time,
                    Q[0],
                    Q[1],
                    facecolor=adapt_color_alpha(color, 0.1),
                    edgecolor=color,
                    linewidth=0.25,
                    linestyle=linestyle,
                    zorder=zorder - 1,
                )

        def plot_summed_trace(time, trace, label, color, zorder, linestyle="solid"):
            if np.any(trace):
                ax.plot(
                    time,
                    trace,
                    label=label,
                    color=color,
                    zorder=zorder,
                    linestyle=linestyle,
                )

        def get_summed_traces(signs, source_types, cv_pd, cv_nd):
            # sum over cell types then average over models
            exc_pd = np.zeros(cv_pd.shape)
            inh_pd = np.zeros(cv_pd.shape)
            exc_nd = np.zeros(cv_nd.shape)
            inh_nd = np.zeros(cv_nd.shape)
            # sum over cell types
            for source_type in source_types:
                if signs[source_type] == 1:
                    exc_pd += cv_pd[source_type][:]  # (1, n_models, 1, n_timesteps)
                    exc_nd += cv_nd[source_type][:]
                else:
                    inh_pd += cv_pd[source_type][:]
                    inh_nd += cv_nd[source_type][:]
            # (n_models, n_timesteps)
            return (
                exc_pd.squeeze(),
                inh_pd.squeeze(),
                exc_nd.squeeze(),
                inh_nd.squeeze(),
            )

        for source_type in source_types:
            if model_average and not only_sum:
                # mean traces solid for PD and dashed for ND
                if baseline_color is not None:
                    color = baseline_color
                elif colors:
                    color = colors[source_type]
                else:
                    color = cv_pd.color(source_type)

                plot_mean_trace(
                    cv_pd.time,
                    cv_pd[source_type][:].squeeze(axis=-2).T.mean(axis=1),
                    source_type,
                    color,
                    cv_pd.zorder(source_types, source_type),
                    ax=ax,
                    fig=fig,
                )
                plot_mean_trace(
                    cv_nd.time,
                    cv_nd[source_type][:].squeeze(axis=-2).T.mean(axis=1),
                    source_type,
                    color,
                    linestyle="dashed",
                    zorder=cv_pd.zorder(source_types, source_type),
                    ax=ax,
                    fig=fig,
                )

            elif not model_average and not only_sum:
                # individual traces
                plot_individual_traces(
                    cv_pd[source_type][:].squeeze(axis=-2),
                    cv_pd.time,
                    cv_pd.color(source_type),
                    cv_pd.zorder(source_types, source_type),
                    source_type,
                )
                plot_individual_traces(
                    cv_nd[source_type][:].squeeze(axis=-2),
                    cv_nd.time,
                    cv_pd.color(source_type),
                    cv_pd.zorder(source_types, source_type),
                    source_type,
                    linestyle="dashed",
                    legend="null direction",
                )

            # quantiles
            plot_quantile(
                cv_pd[source_type][:].squeeze(axis=-2),
                cv_pd.time,
                cv_pd.color(source_type),
                cv_pd.zorder(source_types, source_type),
                linestyle="solid",
            )
            plot_quantile(
                cv_nd[source_type][:].squeeze(axis=-2),
                cv_nd.time,
                cv_pd.color(source_type),
                cv_pd.zorder(source_types, source_type),
                linestyle="dashed",
            )
        if sum_exc_inh or only_sum:
            # plot summed traces
            signs = cv_pd.signs()
            exc_pd, inh_pd, exc_nd, inh_nd = get_summed_traces(
                signs, source_types, cv_pd, cv_nd
            )
            plot_summed_trace(
                cv_pd.time,
                exc_pd.mean(axis=0),
                "excitatory",
                (0.931, 0.0, 0.0, 1.0),
                zorder=2000,
            )
            plot_quantile(
                exc_pd,
                cv_pd.time,
                (0.931, 0.0, 0.0, 1.0),
                zorder=0,
                linestyle="solid",
            )
            plot_summed_trace(
                cv_nd.time,
                exc_nd.mean(axis=0),
                "excitatory",
                (0.931, 0.0, 0.0, 1.0),
                zorder=2000,
                linestyle="dashed",
            )
            plot_quantile(
                exc_nd,
                cv_pd.time,
                (0.931, 0.0, 0.0, 1.0),
                zorder=0,
                linestyle="dashed",
            )
            plot_summed_trace(
                cv_pd.time,
                inh_pd.mean(axis=0),
                "inhibitory",
                (0.0, 0.0, 0.849, 1.0),
                zorder=2000,
            )
            plot_quantile(
                inh_pd,
                cv_pd.time,
                (0.0, 0.0, 0.849, 1.0),
                zorder=0,
                linestyle="solid",
            )
            plot_summed_trace(
                cv_nd.time,
                inh_nd.mean(axis=0),
                "inhibitory",
                (0.0, 0.0, 0.849, 1.0),
                zorder=2000,
                linestyle="dashed",
            )
            plot_quantile(
                inh_nd,
                cv_pd.time,
                (0.0, 0.0, 0.849, 1.0),
                zorder=0,
                linestyle="dashed",
            )

        if hline:
            ax.hlines(
                0,
                cv_pd.time.min(),
                cv_pd.time.max(),
                color=(0, 0, 0, 1),
                linewidth=0.25,
                zorder=-10,
            )

        if legend:
            ax.legend(
                fontsize=fontsize,
                ncols=1,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                borderaxespad=0.0,
            )
        else:
            ax.legend().set_visible(False)

        ax.set_xlabel("time (s)", fontsize=fontsize)
        #         ax.set_ylabel("current (a.u.)", fontsize=fontsize)

        if hide_yaxis:
            plt_utils.rm_spines(ax, ("left",))

        if trim_axes:
            plt_utils.trim_axis(ax)

        if legend_standalone:
            handles, labels = ax.get_legend_handles_labels()
            nd_handle = Line2D(
                [0], [0], color="k", lw=1, label="null direction", ls="dashed"
            )
            legend_n_rows = legend_n_rows or len(labels) + 1
            # legend_n_cols = (len(labels) + 1) // legend_n_rows
            legend_fig, legend_ax = plt_utils.standalone_legend(
                [*labels[::2], "null direction"],
                None,
                [*handles[::2], nd_handle],
                fontsize=fontsize,
                n_cols=legend_n_cols,
                handlelength=2,
                columnspacing=0.8,
                labelspacing=0.25,
                figsize=cm_to_inch(legend_figsize_cm),
                fig=fig if legend_ax is not None else None,
                ax=legend_ax,
            )
            return fig, ax, legend_fig, legend_ax
        return fig, ax

    def plot_temporal_contributions_pc_nc(
        self,
        contrast: float,
        angle: float,
        t_start: float = 0,
        t_end: float = 1,
        fontsize: float = 5,
        linewidth: float = 0.25,
        legend: bool = False,
        legend_standalone: bool = True,
        legend_figsize_cm: tuple[float, float] = (4.0572, 1),
        legend_n_rows: int | None = None,
        max_figure_height_cm: float = 3.3941,
        panel_height_cm: float = 3.3941,
        max_figure_width_cm: float = 4.0572,
        panel_width_cm: float = 4.0572,
        model_average: bool = True,
        highlight_mean: bool = True,
        sum_exc_inh: bool = False,
        only_sum: bool = False,
        hide_source_types: str | list | None = "auto",
        hide_source_types_bins: int = 5,
        hide_source_types_cut_off_edge: int = 1,
        hide_source_types_mode: str = "below_cut_off",
        hide_yaxis: bool = True,
        trim_axes: bool = True,
        quantile: float | None = None,
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        legend_ax: plt.Axes | None = None,
        null_linestyle: str = "dotted",
        legend_n_cols: int | None = None,
    ) -> tuple[plt.Figure, plt.Axes, plt.Figure | None, plt.Axes | None]:
        """
        Temporal contributions of different source types for positive/negative contrasts.

        Args:
            contrast: The contrast of the stimulus.
            angle: The angle of the stimulus.
            t_start: Start time for the plot.
            t_end: End time for the plot.
            fontsize: Font size for labels and titles.
            linewidth: Line width for traces.
            legend: Whether to show the legend.
            legend_standalone: Whether to create a standalone legend.
            legend_figsize_cm: Figure size for the standalone legend.
            legend_n_rows: Number of rows for the standalone legend.
            max_figure_height_cm: Maximum figure height in centimeters.
            panel_height_cm: Height of each panel in centimeters.
            max_figure_width_cm: Maximum figure width in centimeters.
            panel_width_cm: Width of each panel in centimeters.
            model_average: Whether to plot the model average.
            highlight_mean: Whether to highlight the mean trace.
            sum_exc_inh: Whether to sum excitatory and inhibitory contributions.
            only_sum: Whether to only plot the summed contributions.
            hide_source_types: Source types to hide or "auto".
            hide_source_types_bins: Number of bins for auto-hiding.
            hide_source_types_cut_off_edge: Cut-off edge for auto-hiding.
            hide_source_types_mode: Mode for auto-hiding source types.
            hide_yaxis: Whether to hide the y-axis.
            trim_axes: Whether to trim the axes.
            quantile: Quantile for shading.
            fig: Existing figure to use.
            ax: Existing axes to use.
            legend_ax: Existing axes for the standalone legend.
            null_linestyle: Linestyle for null direction traces.
            legend_n_cols: Number of columns for the standalone legend.

        Returns:
            Figure, axes, and legend axes objects.

        Example:
            ```
            view = MovingEdgeCurrentView(...)
            fig, ax = view.plot_temporal_contributions_pc_nc(
                contrast=1.0,
                angle=0,
                t_start=0,
                t_end=1,
                fontsize=5,
                linewidth=0.25,
                legend=True
            )
            ```
        """
        if fig is None and ax is None:
            figsize = figsize_from_n_items(
                1,
                max_figure_height_cm=max_figure_height_cm,
                panel_height_cm=panel_height_cm,
                max_figure_width_cm=max_figure_width_cm,
                panel_width_cm=panel_width_cm,
            )
            fig, axes = figsize.axis_grid(hspace=0.0, wspace=0, fontsize=fontsize)
            ax = axes[0]
        cv_pd = (
            self.at_contrast(contrast)
            .at_angle(angle)
            .between_seconds(t_start, t_end)
            .sum_over_cells()
        )
        cv_nd = (
            self.at_contrast(contrast)
            .at_angle((angle - 180) % 360)
            .between_seconds(t_start, t_end)
            .sum_over_cells()
        )

        source_types = (
            self.at_contrast(contrast)
            .at_angle([angle, (angle - 180) % 360])
            .between_seconds(t_start, t_end)
            .filter_source_types(
                hide_source_types,
                hide_source_types_bins,
                hide_source_types_cut_off_edge,
                hide_source_types_mode,
            )
        )

        color_source_types = (
            self.at_contrast(contrast)
            .at_angle([angle, (angle - 180) % 360])
            .between_seconds(t_start, t_end)
            .filter_source_types(
                None,
                hide_source_types_bins,
                hide_source_types_cut_off_edge,
                hide_source_types_mode,
            )
        )
        cv_pd.init_colors(color_source_types)
        cv_nd.init_colors(color_source_types)

        def plot_mean_trace(
            time, trace, label, color, zorder, linestyle="solid", ax=None, fig=None
        ):
            ax.plot(
                time,
                trace,
                label=label,
                color=color,
                zorder=zorder,
                linestyle=linestyle,
            )

        def plot_individual_traces(
            traces, time, color, zorder, label, linestyle="solid", legend=None
        ):
            if not only_sum and not model_average:
                plots.traces(
                    traces,
                    time,
                    mean_color=color,
                    color=color,
                    linewidth=linewidth,
                    zorder_traces=0,
                    zorder_mean=zorder,
                    fontsize=fontsize,
                    null_line=True,
                    highlight_mean=highlight_mean,
                    fig=fig,
                    ax=ax,
                    legend=legend or label,
                    linestyle=linestyle,
                )

        def plot_quantile(traces, time, color, zorder, linestyle="solid"):
            if quantile:
                Q = np.quantile(traces, quantile, axis=0)
                ax.fill_between(
                    time,
                    Q[0],
                    Q[1],
                    facecolor=adapt_color_alpha(color, 0.1),
                    edgecolor=color,
                    linewidth=0.25,
                    linestyle=linestyle,
                    zorder=zorder - 1,
                )

        def plot_summed_trace(time, trace, label, color, zorder, linestyle="solid"):
            if np.any(trace):
                ax.plot(
                    time,
                    trace,
                    label=label,
                    color=color,
                    zorder=zorder,
                    linestyle=linestyle,
                )

        def get_summed_traces(signs, source_types, cv_pd, cv_nd):
            # sum over cell types then average over models
            exc_pd = np.zeros(cv_pd.shape)
            inh_pd = np.zeros(cv_pd.shape)
            exc_nd = np.zeros(cv_nd.shape)
            inh_nd = np.zeros(cv_nd.shape)
            # sum over cell types
            for source_type in source_types:
                if signs[source_type] == 1:
                    exc_pd += cv_pd[source_type][:]  # (1, n_models, 1, n_timesteps)
                    exc_nd += cv_nd[source_type][:]
                else:
                    inh_pd += cv_pd[source_type][:]
                    inh_nd += cv_nd[source_type][:]
            # (n_models, n_timesteps)
            return (
                exc_pd.squeeze(),
                inh_pd.squeeze(),
                exc_nd.squeeze(),
                inh_nd.squeeze(),
            )

        for source_type in source_types:
            if model_average and not only_sum:
                # mean traces solid for PD and dashed for ND
                color = cv_pd.color(source_type)

                plot_mean_trace(
                    cv_pd.time,
                    cv_pd[source_type][:].squeeze(axis=-2).T.mean(axis=1),
                    source_type,
                    color,
                    cv_pd.zorder(source_types, source_type),
                    ax=ax,
                    fig=fig,
                )
                plot_mean_trace(
                    cv_nd.time,
                    cv_nd[source_type][:].squeeze(axis=-2).T.mean(axis=1),
                    source_type,
                    color,
                    linestyle=null_linestyle,
                    zorder=cv_pd.zorder(source_types, source_type),
                    ax=ax,
                    fig=fig,
                )

            elif not model_average and not only_sum:
                # individual traces
                plot_individual_traces(
                    cv_pd[source_type][:].squeeze(axis=-2),
                    cv_pd.time,
                    cv_pd.color(source_type),
                    cv_pd.zorder(source_types, source_type),
                    source_type,
                )
                plot_individual_traces(
                    cv_nd[source_type][:].squeeze(axis=-2),
                    cv_nd.time,
                    cv_pd.color(source_type),
                    cv_pd.zorder(source_types, source_type),
                    source_type,
                    linestyle=null_linestyle,
                    legend="null direction",
                )

            # quantiles
            plot_quantile(
                cv_pd[source_type][:].squeeze(axis=-2),
                cv_pd.time,
                cv_pd.color(source_type),
                cv_pd.zorder(source_types, source_type),
                linestyle="solid",
            )
            plot_quantile(
                cv_nd[source_type][:].squeeze(axis=-2),
                cv_nd.time,
                cv_pd.color(source_type),
                cv_pd.zorder(source_types, source_type),
                linestyle=null_linestyle,
            )
        if sum_exc_inh or only_sum:
            # plot summed traces
            signs = cv_pd.signs()
            exc_pd, inh_pd, exc_nd, inh_nd = get_summed_traces(
                signs, source_types, cv_pd, cv_nd
            )
            plot_summed_trace(
                cv_pd.time,
                exc_pd.mean(axis=0),
                "excitatory",
                (0.931, 0.0, 0.0, 1.0),
                zorder=2000,
            )
            plot_quantile(
                exc_pd,
                cv_pd.time,
                (0.931, 0.0, 0.0, 1.0),
                zorder=0,
                linestyle="solid",
            )
            plot_summed_trace(
                cv_nd.time,
                exc_nd.mean(axis=0),
                "excitatory",
                (0.931, 0.0, 0.0, 1.0),
                zorder=2000,
                linestyle=null_linestyle,
            )
            plot_quantile(
                exc_nd,
                cv_pd.time,
                (0.931, 0.0, 0.0, 1.0),
                zorder=0,
                linestyle=null_linestyle,
            )
            plot_summed_trace(
                cv_pd.time,
                inh_pd.mean(axis=0),
                "inhibitory",
                (0.0, 0.0, 0.849, 1.0),
                zorder=2000,
            )
            plot_quantile(
                inh_pd,
                cv_pd.time,
                (0.0, 0.0, 0.849, 1.0),
                zorder=0,
                linestyle="solid",
            )
            plot_summed_trace(
                cv_nd.time,
                inh_nd.mean(axis=0),
                "inhibitory",
                (0.0, 0.0, 0.849, 1.0),
                zorder=2000,
                linestyle=null_linestyle,
            )
            plot_quantile(
                inh_nd,
                cv_pd.time,
                (0.0, 0.0, 0.849, 1.0),
                zorder=0,
                linestyle=null_linestyle,
            )

        if legend:
            ax.legend(
                fontsize=fontsize,
                ncols=1,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                borderaxespad=0.0,
            )
        else:
            ax.legend().set_visible(False)

        ax.set_xlabel("time (s)", fontsize=fontsize)
        #         ax.set_ylabel("current (a.u.)", fontsize=fontsize)

        if hide_yaxis:
            plt_utils.rm_spines(ax, ("left",))

        if trim_axes:
            plt_utils.trim_axis(ax)

        if legend_standalone:
            handles, labels = ax.get_legend_handles_labels()
            nd_handle = Line2D(
                [0], [0], color="k", lw=1, label="null direction", ls=null_linestyle
            )
            legend_n_rows = legend_n_rows or len(labels) + 1
            # legend_n_cols = (len(labels) + 1) // legend_n_rows
            legend_fig, legend_ax = plt_utils.standalone_legend(
                [*labels[::2], "null direction"],
                None,
                [*handles[::2], nd_handle],
                fontsize=fontsize,
                n_cols=legend_n_cols,
                handlelength=2,
                columnspacing=0.8,
                labelspacing=0.25,
                figsize=cm_to_inch(legend_figsize_cm),
                fig=fig if legend_ax is not None else None,
                ax=legend_ax,
            )
            return fig, ax, legend_fig, legend_ax
        return fig, ax, None, None

    def get_temporal_contributions(
        self,
        contrast: float,
        angle: float,
        t_start: float = 0,
        t_end: float = 1,
        hide_source_types: str | list | None = "auto",
        hide_source_types_bins: int = 5,
        hide_source_types_cut_off_edge: int = 1,
        hide_source_types_mode: str = "below_cut_off",
        summed_traces: bool = False,
    ) -> tuple[
        "MovingEdgeCurrentView" | np.ndarray,
        "MovingEdgeCurrentView" | np.ndarray,
        list[str],
        list[str],
    ]:
        cv_pd = (
            self.at_contrast(contrast)
            .at_angle(angle)
            .between_seconds(t_start, t_end)
            .sum_over_cells()
        )
        cv_nd = (
            self.at_contrast(contrast)
            .at_angle((angle - 180) % 360)
            .between_seconds(t_start, t_end)
            .sum_over_cells()
        )

        source_types = (
            self.at_contrast(contrast)
            .at_angle([angle, (angle - 180) % 360])
            .between_seconds(t_start, t_end)
            .filter_source_types(
                hide_source_types,
                hide_source_types_bins,
                hide_source_types_cut_off_edge,
                hide_source_types_mode,
            )
        )

        color_source_types = (
            self.at_contrast(contrast)
            .at_angle([angle, (angle - 180) % 360])
            .between_seconds(t_start, t_end)
            .filter_source_types(
                None,
                hide_source_types_bins,
                hide_source_types_cut_off_edge,
                hide_source_types_mode,
            )
        )
        cv_pd.init_colors(color_source_types)
        cv_nd.init_colors(color_source_types)

        def get_summed_traces(signs, source_types, cv_pd, cv_nd):
            # sum over cell types then average over models
            exc_pd = np.zeros(cv_pd.shape)
            inh_pd = np.zeros(cv_pd.shape)
            exc_nd = np.zeros(cv_nd.shape)
            inh_nd = np.zeros(cv_nd.shape)
            # sum over cell types
            for source_type in source_types:
                if signs[source_type] == 1:
                    exc_pd += cv_pd[source_type][:]  # (1, n_models, 1, n_timesteps)
                    exc_nd += cv_nd[source_type][:]
                else:
                    inh_pd += cv_pd[source_type][:]
                    inh_nd += cv_nd[source_type][:]
            # (n_models, n_timesteps)
            return (
                exc_pd.squeeze(),
                inh_pd.squeeze(),
                exc_nd.squeeze(),
                inh_nd.squeeze(),
            )

        if summed_traces:
            exc_pd, inh_pd, exc_nd, inh_nd = get_summed_traces(
                cv_pd.signs(), source_types, cv_pd, cv_nd
            )
            # return exc_pd, inh_pd, exc_nd, inh_nd, source_types, color_source_types
            return exc_pd, inh_pd, exc_nd, inh_nd

        return cv_pd, cv_nd, source_types, color_source_types

    def get_response(
        self,
        contrast: float,
        angle: float,
        t_start: float = 0,
        t_end: float = 1,
        model_average: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r_pd = (
            self.at_angle(angle)
            .at_contrast(contrast)
            .between_seconds(t_start, t_end)
            .responses.squeeze(axis=-2)
        )
        r_nd = (
            self.at_angle((angle - 180) % 360)
            .at_contrast(contrast)
            .between_seconds(t_start, t_end)
            .responses.squeeze(axis=-2)
        )

        if model_average:
            return (
                r_pd.mean(axis=0),
                r_nd.mean(axis=0),
                self.between_seconds(t_start, t_end).time,
            )

        return r_pd, r_nd, self.between_seconds(t_start, t_end).time


def reset_index(rfs: ReceptiveFields, inplace: bool = False) -> ReceptiveFields:
    if inplace:
        for source_type in rfs.source_types:
            edges = rfs[source_type]
            if isinstance(edges, pd.DataFrame):
                rfs[source_type] = edges.reset_index(drop=True)
        return rfs
    else:
        new = rfs.deepcopy()
        return reset_index(new, inplace=True)


def at_position(
    rfs: ReceptiveFields,
    u: float | None = None,
    v: float | None = None,
    central: bool = True,
    inplace: bool = False,
) -> ReceptiveFields:
    if inplace:
        for source_type in rfs.source_types:
            edges = rfs[source_type]
            if isinstance(edges, pd.DataFrame):
                if central:
                    rfs[source_type] = edges.iloc[
                        [np.argmin(np.abs(edges.source_u) + np.abs(edges.source_v))]
                    ]
                else:
                    rfs[source_type] = edges[
                        (edges.source_u == u) & (edges.source_v == v)
                    ]
        return rfs
    else:
        new = rfs.deepcopy()
        return at_position(new, u, v, inplace=True)
