"""
Analysis of responses to flash stimuli.

Info:
    Relies on xarray dataset format defined in `flyvis.analysis.stimulus_responses`.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from flyvis.utils import groundtruth_utils, nodes_edges_utils
from flyvis.utils.color_utils import OFF_FR, ON_FR

from .visualization import plt_utils
from .visualization.plots import violin_groups

__all__ = ["flash_response_index", "fri_correlation_to_known", "plot_fris"]

# -- FRI computation ------------


def flash_response_index(
    self: xr.DataArray,
    radius: float,
    on_intensity: float = 1.0,
    off_intensity: float = 0.0,
    nonnegative: bool = True,
) -> xr.DataArray:
    """
    Compute the Flash Response Index (FRI) using xarray methods.

    Args:
        self: The input DataArray containing response data.
        radius: The radius value to select data for.
        on_intensity: The intensity value for the 'on' state.
        off_intensity: The intensity value for the 'off' state.
        nonnegative: If True, applies a nonnegative constraint to the data.

    Returns:
        xr.DataArray: The computed Flash Response Index.

    Note:
        Ensures that the stimulus configuration is correct for FRI computation.
    """

    # Ensure that the stimulus configuration is correct for FRI computation
    assert tuple(self.attrs['config']['alternations']) == (0, 1, 0)

    responses = self['responses']

    # Select the time window for the stimulus response using query
    time_query = (
        f"{-self.attrs['config']['dt']} <= time <= {self.attrs['config']['t_stim']}"
    )
    stim_response = responses.query(frame=time_query)

    # Select the data for the given radius
    stim_response = stim_response.query(sample=f'radius=={radius}')

    # Apply nonnegative constraint if required
    if nonnegative:
        minimum = stim_response.min(dim=['frame', 'sample'])
        stim_response += np.abs(minimum)

    # Select the response data for on and off intensities
    r_on = stim_response.query(sample=f'intensity=={on_intensity}')
    r_off = stim_response.query(sample=f'intensity=={off_intensity}')

    # Compute the peak responses by finding the maximum along the 'frame' dimension
    on_peak = r_on.max(dim='frame')
    off_peak = r_off.max(dim='frame')

    # Drop the 'sample' coordinate to avoid broadcasting issues
    on_peak = on_peak.drop('sample')
    off_peak = off_peak.drop('sample')

    # Compute the Flash Response Index (FRI)
    fri = on_peak - off_peak
    fri /= on_peak + off_peak + np.array([1e-16])

    # Optionally, you can drop NaN values after computation
    return fri.dropna(dim='sample', how='any')


# -- correlation ------------


def fri_correlation_to_known(fris: xr.DataArray) -> xr.DataArray:
    """
    Compute the correlation of the FRI to known cell type tunings.

    Args:
        fris: DataArray containing Flash Response Index values.

    Returns:
        xr.DataArray: Correlation of FRIs to known cell type tunings.
    """
    known_preferred_contrasts = {
        k: v for k, v in groundtruth_utils.polarity.items() if v != 0
    }
    known_cell_types = list(known_preferred_contrasts.keys())
    groundtruth = list(known_preferred_contrasts.values())

    index = np.array([
        np.where(nt == fris.cell_type)[0].item() for i, nt in enumerate(known_cell_types)
    ])
    fris = fris.isel(neuron=index)
    groundtruth = xr.DataArray(
        data=groundtruth,
        dims=["neuron"],
    )
    return xr.corr(fris, groundtruth, dim="neuron")


# -- plotting code -------------


def plot_fris(
    fris: np.ndarray,
    cell_types: np.ndarray,
    scatter_best: bool = False,
    scatter_all: bool = True,
    bold_output_type_labels: bool = True,
    output_cell_types: Optional[List[str]] = None,
    known_first: bool = True,
    sorted_type_list: Optional[List[str]] = None,
    figsize: List[int] = [10, 1],
    cmap: plt.cm = plt.cm.Greys_r,
    ylim: Tuple[float, float] = (-1, 1),
    color_known_types: bool = True,
    fontsize: int = 6,
    colors: Optional[List[str]] = None,
    color: str = "b",
    showmeans: bool = False,
    showmedians: bool = True,
    scatter_edge_width: float = 0.5,
    scatter_best_edge_width: float = 0.75,
    scatter_edge_color: str = "none",
    scatter_face_color: str = "k",
    scatter_alpha: float = 0.35,
    scatter_best_alpha: float = 1.0,
    scatter_all_marker: str = "o",
    scatter_best_index: Optional[int] = None,
    scatter_best_marker: str = "o",
    scatter_best_color: Optional[str] = None,
    mean_median_linewidth: float = 1.5,
    mean_median_bar_length: float = 1.0,
    violin_alpha: float = 0.3,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot flash response indices (FRIs) for the given cell types with violins.

    Args:
        fris: Array of FRI values (n_random_variables, n_groups, n_samples).
        cell_types: Array of cell type labels, corresponding to the first axis
            (n_random_variables) of `fris`.
        scatter_best: If True, scatter the best points.
        scatter_all: If True, scatter all points.
        bold_output_type_labels: If True, bold the output type labels.
        output_cell_types: List of cell types to bold in the output.
        known_first: If True, sort known cell types first.
        sorted_type_list: List of cell types to sort by.
        figsize: Figure size as [width, height].
        cmap: Colormap for the plot.
        ylim: Y-axis limits as (min, max).
        color_known_types: If True, color known cell type labels.
        fontsize: Font size for labels.
        colors: List of colors for the violins.
        color: Single color for all violins if cmap is None.
        showmeans: If True, show means on the violins.
        showmedians: If True, show medians on the violins.
        scatter_edge_width: Width of scatter point edges.
        scatter_best_edge_width: Width of best scatter point edges.
        scatter_edge_color: Color of scatter point edges.
        scatter_face_color: Color of scatter point faces.
        scatter_alpha: Alpha value for scatter points.
        scatter_best_alpha: Alpha value for best scatter points.
        scatter_all_marker: Marker style for all scatter points.
        scatter_best_index: Index of the best scatter point.
        scatter_best_marker: Marker style for the best scatter point.
        scatter_best_color: Color of the best scatter point.
        mean_median_linewidth: Line width for mean/median lines.
        mean_median_bar_length: Length of mean/median bars.
        violin_alpha: Alpha value for violin plots.
        **kwargs: Additional keyword arguments for violin_groups.

    Returns:
        Tuple containing the Figure and Axes objects.
    """
    # Process FRIs data
    if len(fris.shape) != 3:
        fris = fris[:, None]
    if fris.shape[0] != len(cell_types):
        fris = np.transpose(fris, (2, 1, 0))

    # Sort cell types
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

    # Set colors
    if colors is not None:
        pass
    elif cmap is not None:
        colors = None
    elif color is not None:
        cmap = None
        colors = (color,)

    # Create violin plot
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
        showmeans=showmeans,
        showmedians=showmedians,
        mean_median_linewidth=mean_median_linewidth,
        mean_median_bar_length=mean_median_bar_length,
        violin_alpha=violin_alpha,
        **kwargs,
    )

    # Add scatter points if necessary
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

    # Customize plot appearance
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
        color="k",
        zorder=0,
    )
    ax.set_yticks(np.arange(-1.0, 1.5, 0.5))

    return fig, ax


def flash_response_color_labels(ax: plt.Axes) -> plt.Axes:
    """
    Color the labels of ON and OFF cells in the plot.

    Args:
        ax: The matplotlib Axes object to modify.

    Returns:
        The modified matplotlib Axes object.
    """
    on = [key for key, value in groundtruth_utils.polarity.items() if value == 1]
    off = [key for key, value in groundtruth_utils.polarity.items() if value == -1]
    plt_utils.color_labels(on, ON_FR, ax)
    plt_utils.color_labels(off, OFF_FR, ax)
    return ax
