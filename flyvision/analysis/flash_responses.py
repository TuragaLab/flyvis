import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union

from datamate import Namespace

from flyvision.utils import groundtruth_utils, nodes_edges_utils
from flyvision.utils.activity_utils import StimulusResponseIndexer
from flyvision.utils.nodes_edges_utils import CellTypeArray
from flyvision.plots import plt_utils
from flyvision.plots.plots import violin_groups, traces, grouped_traces
from flyvision.analysis.simple_correlation import correlation


# -- response indexer ---------------

class FlashResponseView(StimulusResponseIndexer):
    def __init__(
        self,
        arg_df: pd.DataFrame,
        config: Namespace,
        responses: CellTypeArray,
        stim_sample_dim=0,
        temporal_dim=1,
        time=None,
    ):
        self.config = config
        super().__init__(
            arg_df=arg_df,  # could also construct from config
            responses=responses,
            dt=config.dt,
            t_pre=config.t_pre,
            stim_sample_dim=stim_sample_dim,
            temporal_dim=temporal_dim,
            time=time,
        )

    def view(
        self,
        arg_df: pd.DataFrame = None,
        config: Namespace = None,
        responses: Union[np.ndarray, CellTypeArray] = None,
        stim_sample_dim=None,
        temporal_dim=None,
        time=None,
    ) -> "FlashResponseView":
        if isinstance(responses, np.ndarray):
            responses = CellTypeArray(responses, cell_types=self.responses.cell_types)

        return self.__class__(
            arg_df if np.any(arg_df) else self.arg_df,
            config if config is not None else self.config,
            responses if responses is not None else self.responses,
            stim_sample_dim if np.any(stim_sample_dim) else self.stim_sample_dim,
            temporal_dim or self.temporal_dim,
            time if np.any(time) else self.time,
        )

    def init_time(self, time=None) -> None:
        if time is not None:
            self.time = time
            return
        if self.config:
            self.time = (
                self.time
                if np.any(self.time)
                and len(self.time) == self.responses.shape[self.temporal_dim]
                else (
                    np.arange(0, self.responses.shape[self.temporal_dim]) * self.dt
                ) # don't subtract t_pre here, since flashes *could* start with stimulus
            )

    def fri(
        self,
        radius,
        on_intensity=1.0,
        off_intensity=0.0,
        nonnegative=True,
    ):
        assert (
            self.config.alternations[0] == 0 and self.config.alternations[1] == 1
        ), f"Invalid Flashes stimulus for computing FRI. Please use `alternations=[0,1,0]`"
        stim_response = self.between_seconds(
            self.config.t_pre - self.config.dt,
            self.config.t_pre + self.config.t_stim
        )

        stim_response = stim_response.where_stim_args(radius=radius)
        if nonnegative:
            stim_response = stim_response.nonnegative()
        r_on = stim_response.where_stim_args(intensity=on_intensity)
        r_off = stim_response.where_stim_args(intensity=off_intensity)

        on_peak = r_on.peak()
        off_peak = r_off.peak()
        fri = on_peak - off_peak
        fri /= on_peak + off_peak + np.array([1e-16])

        return fri
    
    def plot_traces(self, cell_type, time=None, plot_kwargs=dict(), **stim_kwargs):
        if self.responses.shape[self.temporal_dim] != len(self.time):
            raise ValueError(
                "Cannot plot. Previous operations have mis-aligned the FlashResponseView "
                "response data and timestamps."
            )
        cell_trace = self.cell_type(cell_type).where_stim_args(**stim_kwargs).view(time=time)
        time_shape = cell_trace.shape[self.temporal_dim]
        stim_shape = cell_trace.shape[self.stim_sample_dim]
        response_arr = cell_trace.responses[:]
        label_cols = [col for col in cell_trace.arg_df.columns if cell_trace.arg_df[col].nunique() > 1]
        response_arr = response_arr.transpose(
            cell_trace.stim_sample_dim,
            *tuple(set(range(response_arr.ndim)) - {cell_trace.stim_sample_dim, cell_trace.temporal_dim}),
            cell_trace.temporal_dim,
        ).reshape(stim_shape, -1, time_shape)
        if response_arr.shape[1] > 1:
            return grouped_traces(
                response_arr,
                cell_trace.time,
                linewidth=0.5,
                legend=tuple([", ".join([
                    f"{col}={cell_trace.arg_df[col].iloc[i].item()}" 
                    for col in label_cols
                ]) for i in range(len(cell_trace.arg_df))]),
                ylabel="activity (a.u.)",
                xlabel="time (s)",
                title=f"{cell_type} flash response",
                **plot_kwargs,
            )
        else:
            return traces(
                response_arr[:, 0, :],
                cell_trace.time,
                legend=tuple([", ".join([
                    f"{col}={cell_trace.arg_df[col].iloc[i].item()}" 
                    for col in label_cols
                ]) for i in range(len(cell_trace.arg_df))]),
                ylabel="activity (a.u.)",
                xlabel="time (s)",
                title=f"{cell_type} flash response",
                **plot_kwargs,
            )


# -- correlation ------------

def fri_correlation_to_known(fris, cell_types):
    known_preferred_contrasts = {k: v for k, v in groundtruth_utils.polarity.items() if v != 0}
    known_cell_types = list(known_preferred_contrasts.keys())
    groundtruth = list(known_preferred_contrasts.values())

    index = np.array(
        [
            np.where(nt == cell_types)[0].item()
            for i, nt in enumerate(known_cell_types)
        ]
    )

    fris_for_known = fris[:, index]

    corr_fri, _ = correlation(
        groundtruth, fris_for_known
    )

    return corr_fri

# -- plotting code -------------

ON_FR = "#c1b933"  # yellow
OFF_FR = "#b140cc"  # violett


def plot_fris(
    fris,  # fris.responses[:]
    cell_types,  # fris.responses.cell_types
    scatter_best=True,
    scatter_all=True,
    bold_output_type_labels=True,
    output_cell_types=None,
    known_first=True,
    sorted_type_list=None,
    figsize=[10, 1],
    cmap=plt.cm.Greys_r,
    ylim=(-1, 1),
    color_known_types=True,
    fontsize=6,
    **kwargs
):
    
    fig, ax, colors, fris = fri_violins(
        fris=fris,
        cell_types=cell_types,
        cmap=cmap,
        fontsize=fontsize,
        sorted_type_list=sorted_type_list,
        figsize=figsize,
        scatter_best=scatter_best,
        scatter_all=scatter_all,
        known_first=known_first,
        **kwargs,
    )
    ax.grid(False)

    if bold_output_type_labels and output_cell_types is not None:
        plt_utils.boldify_labels(output_cell_types, ax)

    ax.set_ylim(*ylim)
    plt_utils.trim_axis(ax)
    plt_utils.set_spine_tick_params(
        ax,
        tickwidth=0.5,
        ticklength=3,
        ticklabelpad=2,
        spinewidth=0.5,
    )

    if color_known_types:
        ax = flash_response_color_labels(ax)

    ax.hlines(
        0,
        min(ax.get_xticks()),
        max(ax.get_xticks()),
        linewidth=0.25,
        # linestyles="dashed",
        color="k",
        zorder=0,
    )
    ax.set_yticks(np.arange(-1.0, 1.5, 0.5))

    return fig, ax


def fri_violins(
    fris,
    cell_types,
    scatter_best=True,
    scatter_all=True,
    cmap=plt.cm.Oranges_r,
    colors=None,
    color="b",
    figsize=[10, 1],
    fontsize=6,
    showmeans=False,
    showmedians=True,
    sorted_type_list=None,
    scatter_edge_width=0.5,
    scatter_best_edge_width=0.75,
    scatter_edge_color="k",
    scatter_face_color="none",
    scatter_alpha=0.35,
    scatter_best_alpha=1.0,
    scatter_all_marker="o",
    scatter_best_index=None,
    scatter_best_marker="o",
    scatter_best_color=None,
    known_first=True,
    **kwargs,
):
    # always add empty group axis for violin plot unless fris is provided
    # with 3 axes
    if len(fris.shape) != 3:
        fris = fris[:, None]

    # transpose to #cell_types, #groups, #samples
    if fris.shape[0] != len(cell_types):
        fris = np.transpose(fris, (2, 1, 0))

    if sorted_type_list is not None:
        fris = nodes_edges_utils.sort_by_mapping_lists(
            cell_types, sorted_type_list, fris, axis=0
        )
        cell_types = np.array(sorted_type_list)

    if known_first:
        _cell_types = nodes_edges_utils.nodes_list_sorting_on_off_unknown(cell_types)
        fris = nodes_edges_utils.sort_by_mapping_lists(
            cell_types, _cell_types, fris, axis=0
        )
        cell_types = np.array(_cell_types)

    if colors is not None:
        pass
    elif cmap is not None:
        colors = None
    elif color is not None:
        cmap = None
        colors = (color,)

    fig, ax, colors = violin_groups(
        fris,
        cell_types[:],
        rotation=90,
        scatter=False,
        cmap=cmap,
        colors=colors,
        fontsize=fontsize,
        figsize=figsize,
        width=0.7,
        # scatter_edge_color=scatter_edge_color,
        # scatter_radius=5,
        # scatter_edge_width=scatter_edge_width,
        showmeans=showmeans,
        showmedians=showmedians,
        **kwargs,
    )

    if fris.shape[1] == 1:
        plt_utils.scatter_on_violins_with_best(
            fris.T.squeeze(),
            ax,
            scatter_best,
            scatter_all,
            best_index=scatter_best_index,
            linewidth=scatter_edge_width,
            best_linewidth=scatter_best_edge_width,
            edgecolor=scatter_edge_color,
            facecolor=scatter_face_color,
            all_scatter_alpha=scatter_alpha,
            best_scatter_alpha=scatter_best_alpha,
            all_marker=scatter_all_marker,
            best_marker=scatter_best_marker,
            best_color=scatter_best_color,
        )
    return fig, ax, colors, fris


def flash_response_color_labels(ax):
    on = [
        key
        for key, value in groundtruth_utils.polarity.items()
        if value == 1
    ]
    off = [
        key
        for key, value in groundtruth_utils.polarity.items()
        if value == -1
    ]
    plt_utils.color_labels(on, ON_FR, ax)
    plt_utils.color_labels(off, OFF_FR, ax)
    return ax

