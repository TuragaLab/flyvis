from typing import Any, Iterable, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datamate import Namespace
from matplotlib.axes import Axes
from matplotlib.colors import hex2color
from matplotlib.lines import Line2D
from toolz import valfilter, valmap

from flyvision.connectome import ReceptiveFields
from flyvision.datasets.moving_bar import MovingEdge
from flyvision.utils.color_utils import (
    ND,
    PD,
    adapt_color_alpha,
    cmap_iter,
    truncate_colormap,
)
from flyvision.utils.df_utils import where_dataframe as get_stimulus_index
from flyvision.utils.nodes_edges_utils import CellTypeArray

from .visualization import plots, plt_utils
from .visualization.figsize_utils import (
    cm_to_inch,
    figsize_from_n_items,
)

__all__ = ["MovingEdgeCurrentView"]


# TODO: update to functions on xarray dataset
class MovingEdgeCurrentView:
    def __init__(
        self,
        ensemble,
        target_type,
        exp_data,
        arg_df=None,
        currents=None,
        rfs=None,
        time=None,
        responses=None,
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

    def init_currents(self, currents) -> None:
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

    def init_responses(self, responses) -> None:
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

    def init_time(self, time) -> None:
        if time is not None:
            self.time = time
            return
        self.time = self.time or (
            np.arange(0, next(iter(self.currents.values())).shape[-2]) * self.config.dt
            - self.config.t_pre
        )

    @property
    def on(self) -> "MovingEdgeCurrentView":
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

    def at_contrast(self, contrast) -> "MovingEdgeCurrentView":
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

    def at_angle(self, angle) -> "MovingEdgeCurrentView":
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

    def at_position(self, u=None, v=None, central=True) -> "MovingEdgeCurrentView":
        rfs = at_position(self.rfs, u, v, central)
        currents = Namespace({
            cell_type: c[:, :, :, :, rfs[cell_type].index]
            for cell_type, c in self.currents.items()
        })
        return self.view(currents, rfs=rfs)

    def between_seconds(self, t_start, t_end) -> "MovingEdgeCurrentView":
        slice = np.where((self.time >= t_start) & (self.time < t_end))[0]
        newview = self[:, :, slice, :]
        newview.time = self.time[slice]
        newview.responses = self.responses[:, :, slice]
        return newview

    def model_selection(self, mask) -> "MovingEdgeCurrentView":
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

    def sorting(self, average_over_models=True, mode="all"):
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
        self, bins=3, cut_off_edge=1, mode="above_cut_off", statistic=np.max
    ):
        """
        Intuitively chunks the y-axis of the current plots into two parts:
            - excitatory
            - inhibitory
        and each of these into bins. In case of 3 corresponding to low-contribution,
        moderate-contribution, high-contribution. Then all cell types above or below the
        bin edge specified by cut_off_edge are discarded.
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

    def filter_source_types(self, hide_source_types, bins, edge, mode, statistic=np.max):
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

    def signs(self):
        return {ct: np.mean(self.rfs[ct].sign) for ct in self.rfs.source_types}

    def sum_over_cells(self) -> "MovingEdgeCurrentView":
        return self.view(
            Namespace({
                cell_type: c.sum(axis=-1) for cell_type, c in self.currents.items()
            }),
        )

    def plot_spatial_contribution(
        self,
        source_type,
        #         contrast,
        #         angle,
        t_start,
        t_end,
        mode="peak",
        title="{source_type} :→",
        fig=None,
        ax=None,
        max_extent=None,
        **kwargs,
    ):
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
        #         contrast,
        #         angle,
        t_start,
        t_end,
        max_extent=3,
        mode="peak",
        title="{source_type} :→",
        fig=None,
        axes=None,
        fontsize=5,
        edgewidth=0.125,
        title_y=0.8,
        max_figure_height_cm=9.271,
        panel_height_cm="auto",
        max_figure_width_cm=2.54,
        panel_width_cm=2.54,
        annotate=False,
        cbar=False,
        hide_source_types="auto",
        hide_source_types_bins=5,
        hide_source_types_cut_off_edge=1,
        hide_source_types_mode="below_cut_off",
        max_axes=None,
        **kwargs,
    ):
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
        source_type,
        #         contrast,
        #         angle,
        title="{source_type} :→",
        fig=None,
        ax=None,
        max_extent=None,
        **kwargs,
    ):
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
        title="{source_type} :→",
        fig=None,
        axes=None,
        max_extent=None,
        fontsize=5,
        edgewidth=0.125,
        title_y=0.8,
        max_figure_height_cm=9.271,
        panel_height_cm="auto",
        max_figure_width_cm=2.54,
        panel_width_cm=2.54,
        annotate=False,
        cbar=False,
        hide_source_types="auto",
        hide_source_types_bins=5,
        hide_source_types_cut_off_edge=1,
        hide_source_types_mode="below_cut_off",
        max_axes=None,
        wspace=0.0,
        hspace=0.1,
        **kwargs,
    ):
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
        # import pdb

        # pdb.set_trace()
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
        self, currents, rfs=None, time=None, responses=None, arg_df=None
    ) -> "MovingEdgeCurrentView":
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
        return self.view(
            Namespace({
                cell_type: c - np.take(c, [0], -2)
                for cell_type, c in self.currents.items()
            }),
            responses=self.responses - np.take(self.responses, [0], -1),
        )

    def subtract_mean(self) -> "MovingEdgeCurrentView":
        return self.view(
            Namespace({
                cell_type: c - np.mean(c, -2, keepdims=True)
                for cell_type, c in self.currents.items()
            }),
            responses=self.responses - np.mean(self.responses, -1, keepdims=True),
        )

    def standardize(self) -> "MovingEdgeCurrentView":
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
        self, t_start, t_end, pd
    ) -> "MovingEdgeCurrentView":
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

    def init_colors(self, source_types):
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

    def color(self, source_type, pd=True):
        if pd:
            return self.colors_pd[source_type]
        return self.colors_nd[source_type]

    def zorder(self, source_types, source_type, start_exc=1000, start_inh=1000):
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

    def ylims(self, source_types=None, offset=0.02):
        "Ylims for temporal contributions summed over cells."
        if source_types is not None:
            return {
                cell_type: plt_utils.get_lims(c, offset)
                for cell_type, c in self.sum_over_cells().currents.items()
                if cell_type in source_types
            }
        return plt_utils.get_lims(list(self.sum_over_cells().currents.values()), offset)

    def plot_response(
        self,
        contrast,
        angle,
        t_start=0,
        t_end=1,
        max_figure_height_cm=1.4477,
        panel_height_cm=1.4477,
        max_figure_width_cm=4.0513,
        panel_width_cm=4.0513,
        fontsize=5,
        model_average=True,
        color=(0, 0, 0),
        legend=False,
        hide_yaxis=True,
        trim_axes=True,
        quantile=None,
        scale_position=None,  # "lower left",
        scale_label="{:.0f} ms",
        scale_unit=1000,
        hline=False,
        fig=None,
        ax=None,
    ):
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
        contrast,
        angle,
        t_start=0,
        t_end=1,
        max_figure_height_cm=1.4477,
        panel_height_cm=1.4477,
        max_figure_width_cm=4.0513,
        panel_width_cm=4.0513,
        fontsize=5,
        model_average=True,
        color=(0, 0, 0),
        legend=False,
        hide_yaxis=True,
        trim_axes=True,
        quantile=None,
        scale_position=None,  # "lower left",
        scale_label="{:.0f} ms",
        scale_unit=1000,
        fig=None,
        ax=None,
        hline=False,
    ):
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
        contrast,
        angle,
        t_start=0,
        t_end=1,
        fontsize=5,
        linewidth=0.25,
        legend=False,
        legend_standalone=True,
        legend_figsize_cm=(4.0572, 1),
        legend_n_rows=None,
        # for supplementary
        # max_figure_height_cm=1.4477,
        # panel_height_cm=1.4477,
        # max_figure_width_cm=4.0513,
        # panel_width_cm=4.0513,
        # for Fig 3
        max_figure_height_cm=3.3941,
        panel_height_cm=3.3941,
        max_figure_width_cm=4.0572,
        panel_width_cm=4.0572,
        model_average=True,
        highlight_mean=True,  # only applies if model_average is False
        sum_exc_inh=False,
        only_sum=False,
        hide_source_types="auto",
        hide_source_types_bins=5,
        hide_source_types_cut_off_edge=1,
        hide_source_types_mode="below_cut_off",
        hide_yaxis=True,
        trim_axes=True,
        quantile=None,
        fig=None,
        ax=None,
        legend_ax=None,
        hline=True,
        legend_n_cols=None,
        baseline_color=None,
        colors=None,
    ):
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
        contrast,
        angle,
        t_start=0,
        t_end=1,
        fontsize=5,
        linewidth=0.25,
        legend=False,
        legend_standalone=True,
        legend_figsize_cm=(4.0572, 1),
        legend_n_rows=None,
        max_figure_height_cm=3.3941,
        panel_height_cm=3.3941,
        max_figure_width_cm=4.0572,
        panel_width_cm=4.0572,
        model_average=True,
        highlight_mean=True,  # only applies if model_average is False
        sum_exc_inh=False,
        only_sum=False,
        hide_source_types="auto",
        hide_source_types_bins=5,
        hide_source_types_cut_off_edge=1,
        hide_source_types_mode="below_cut_off",
        hide_yaxis=True,
        trim_axes=True,
        quantile=None,
        fig=None,
        ax=None,
        legend_ax=None,
        null_linestyle="dotted",
        legend_n_cols=None,
    ):
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
        cv_pc = (
            self.at_contrast(contrast)
            .at_angle(angle)
            .between_seconds(t_start, t_end)
            .sum_over_cells()
        )
        cv_nc = (
            self.at_contrast(0 if contrast == 1 else 1)
            .at_angle(angle)
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
        cv_pc.init_colors(color_source_types)
        cv_nc.init_colors(color_source_types)

        def plot_mean_trace(time, trace, label, color, zorder, linestyle="solid"):
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

        def get_summed_traces(signs, source_types, cv_pc, cv_nc):
            # sum over cell types then average over models
            exc_pd = np.zeros(cv_pc.shape)
            inh_pd = np.zeros(cv_pc.shape)
            exc_nd = np.zeros(cv_nc.shape)
            inh_nd = np.zeros(cv_nc.shape)
            # sum over cell types
            for source_type in source_types:
                if signs[source_type] == 1:
                    exc_pd += cv_pc[source_type][:]  # (1, n_models, 1, n_timesteps)
                    exc_nd += cv_nc[source_type][:]
                else:
                    inh_pd += cv_pc[source_type][:]
                    inh_nd += cv_nc[source_type][:]
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
                plot_mean_trace(
                    cv_pc.time,
                    cv_pc[source_type][:].squeeze(axis=-2).T.mean(axis=1),
                    source_type,
                    cv_pc.color(source_type),
                    cv_pc.zorder(source_types, source_type),
                )
                plot_mean_trace(
                    cv_nc.time,
                    cv_nc[source_type][:].squeeze(axis=-2).T.mean(axis=1),
                    source_type,
                    cv_pc.color(source_type),
                    linestyle=null_linestyle,
                    zorder=cv_pc.zorder(source_types, source_type),
                )

            elif not model_average and not only_sum:
                # individual traces
                plot_individual_traces(
                    cv_pc[source_type][:].squeeze(axis=-2),
                    cv_pc.time,
                    cv_pc.color(source_type),
                    cv_pc.zorder(source_types, source_type),
                    source_type,
                )
                plot_individual_traces(
                    cv_nc[source_type][:].squeeze(axis=-2),
                    cv_nc.time,
                    cv_pc.color(source_type),
                    cv_pc.zorder(source_types, source_type),
                    source_type,
                    linestyle=null_linestyle,
                    legend="null contrast",
                )

            # quantiles
            plot_quantile(
                cv_pc[source_type][:].squeeze(axis=-2),
                cv_pc.time,
                cv_pc.color(source_type),
                cv_pc.zorder(source_types, source_type),
                linestyle="solid",
            )
            plot_quantile(
                cv_nc[source_type][:].squeeze(axis=-2),
                cv_nc.time,
                cv_pc.color(source_type),
                cv_pc.zorder(source_types, source_type),
                linestyle=null_linestyle,
            )
        if sum_exc_inh or only_sum:
            # plot summed traces
            signs = cv_pc.signs()
            exc_pd, inh_pd, exc_nd, inh_nd = get_summed_traces(
                signs, source_types, cv_pc, cv_nc
            )
            plot_summed_trace(
                cv_pc.time,
                exc_pd.mean(axis=0),
                "excitatory",
                (0.931, 0.0, 0.0, 1.0),
                zorder=2000,
            )
            plot_quantile(
                exc_pd,
                cv_pc.time,
                (0.931, 0.0, 0.0, 1.0),
                zorder=0,
                linestyle="solid",
            )
            plot_summed_trace(
                cv_nc.time,
                exc_nd.mean(axis=0),
                "excitatory",
                (0.931, 0.0, 0.0, 1.0),
                zorder=2000,
                linestyle=null_linestyle,
            )
            plot_quantile(
                exc_nd,
                cv_pc.time,
                (0.931, 0.0, 0.0, 1.0),
                zorder=0,
                linestyle=null_linestyle,
            )
            plot_summed_trace(
                cv_pc.time,
                inh_pd.mean(axis=0),
                "inhibitory",
                (0.0, 0.0, 0.849, 1.0),
                zorder=2000,
            )
            plot_quantile(
                inh_pd,
                cv_pc.time,
                (0.0, 0.0, 0.849, 1.0),
                zorder=0,
                linestyle="solid",
            )
            plot_summed_trace(
                cv_nc.time,
                inh_nd.mean(axis=0),
                "inhibitory",
                (0.0, 0.0, 0.849, 1.0),
                zorder=2000,
                linestyle=null_linestyle,
            )
            plot_quantile(
                inh_nd,
                cv_pc.time,
                (0.0, 0.0, 0.849, 1.0),
                zorder=0,
                linestyle=null_linestyle,
            )
        ax.axhline(0, color=(0, 0, 0, 1), linewidth=0.25, zorder=-10)

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
                [0],
                [0],
                color="k",
                lw=1,
                label="null contrast",
                ls=null_linestyle,
            )
            legend_n_rows = legend_n_rows or len(labels) + 1
            # legend_n_cols = (len(labels) + 1) // legend_n_rows
            legend_fig, legend_ax = plt_utils.standalone_legend(
                [*labels[::2], "null contrast"],
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

    def get_temporal_contributions(
        self,
        contrast,
        angle,
        t_start=0,
        t_end=1,
        hide_source_types="auto",
        hide_source_types_bins=5,
        hide_source_types_cut_off_edge=1,
        hide_source_types_mode="below_cut_off",
        summed_traces=False,
    ):
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
        contrast,
        angle,
        t_start=0,
        t_end=1,
        model_average=True,
    ):
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


def reset_index(rfs, inplace=False):
    if inplace:
        for source_type in rfs.source_types:
            edges = rfs[source_type]
            if isinstance(edges, pd.DataFrame):
                rfs[source_type] = edges.reset_index(drop=True)
        return rfs
    else:
        new = rfs.deepcopy()
        return reset_index(new, inplace=True)


def at_position(rfs, u=None, v=None, central=True, inplace=False):
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
