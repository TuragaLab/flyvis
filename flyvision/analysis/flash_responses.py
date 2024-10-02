import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from flyvision.utils import groundtruth_utils, nodes_edges_utils
from flyvision.utils.color_utils import OFF_FR, ON_FR

from .visualization import plt_utils
from .visualization.plots import violin_groups

__all__ = ["flash_response_index", "fri_correlation_to_known", "plot_fris"]

# -- FRI computation ------------


def flash_response_index(
    self: xr.DataArray,
    radius,
    on_intensity=1.0,
    off_intensity=0.0,
    nonnegative=True,
) -> xr.DataArray:
    """Compute the Flash Response Index (FRI) using xarray methods."""

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
    """Compute the correlation of the FRI to known cell type tunings."""
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
    fris,  # fris.responses[:]
    cell_types,  # fris.responses.cell_types
    scatter_best=False,
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
    **kwargs,
):
    """Plot flash response indices (FRIs) for the given cell types with violins."""
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
    scatter_edge_color="none",
    scatter_face_color="k",
    scatter_alpha=0.35,
    scatter_best_alpha=1.0,
    scatter_all_marker="o",
    scatter_best_index=None,
    scatter_best_marker="o",
    scatter_best_color=None,
    known_first=True,
    mean_median_linewidth=1.5,
    mean_median_bar_length=1.0,
    violin_alpha=0.3,
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
        showmeans=showmeans,
        showmedians=showmedians,
        mean_median_linewidth=mean_median_linewidth,
        mean_median_bar_length=mean_median_bar_length,
        violin_alpha=violin_alpha,
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
    on = [key for key, value in groundtruth_utils.polarity.items() if value == 1]
    off = [key for key, value in groundtruth_utils.polarity.items() if value == -1]
    plt_utils.color_labels(on, ON_FR, ax)
    plt_utils.color_labels(off, OFF_FR, ax)
    return ax
