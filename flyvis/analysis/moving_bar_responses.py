"""
Analysis of responses to moving edges or bars.

Info:
    Relies on xarray dataset format defined in `flyvis.analysis.stimulus_responses`.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from flyvis.utils import groundtruth_utils, nodes_edges_utils
from flyvis.utils.color_utils import OFF, ON

from .visualization import plt_utils
from .visualization.plots import polar, violin_groups

__all__ = [
    "peak_responses",
    "peak_responses_angular",
    "direction_selectivity_index",
    "dsi_correlation_to_known",
    "dsi_violins_on_and_off",
    "dsi_violins",
    "preferred_direction",
    "plot_angular_tuning",
    "plot_T4_tuning",
    "plot_T5_tuning",
    "get_known_tuning_curves",
    "correlation_to_known_tuning_curves",
    "angular_distance_to_known",
]


def peak_responses(
    dataset: xr.Dataset,
    norm: xr.DataArray = None,
    from_degree: float = None,
    to_degree: float = None,
) -> xr.DataArray:
    """
    Compute peak responses from rectified voltages, optionally normalized.

    Args:
        dataset: Input dataset containing 'responses' and necessary coordinates.
        norm: Normalization array.
        from_degree: Starting degree for masking.
        to_degree: Ending degree for masking.

    Returns:
        Peak responses with reshaped and transposed dimensions.
    """
    config = dataset.attrs['config']
    from_degree = from_degree if from_degree is not None else config['offsets'][0] * 2.25
    to_degree = to_degree if to_degree is not None else (config['offsets'][1] - 1) * 2.25

    # Generate time masks
    masks = get_time_masks(
        dataset, from_column=from_degree / 5.8, to_column=to_degree / 5.8
    )

    # Apply masks to responses and rectify
    responses = dataset['responses']
    masked = responses.where(masks, other=0)
    rectified = masked.clip(min=0)  # Rectify: max(0, response)

    # Normalize if provided
    if norm is not None:
        rectified = rectified / norm

    # Compute peak (maximum over 'frame')
    peak = rectified.max(dim='frame')
    return peak


def get_time_masks(
    dataset: xr.Dataset, from_column: float = -1.5, to_column: float = 1.5
) -> xr.DataArray:
    """
    Generate time masks for each sample based on speed and column range.

    Args:
        dataset: Input dataset containing 'speed' and 'time' coordinates.
        from_column: Start of the column range.
        to_column: End of the column range.

    Returns:
        Boolean mask with dimensions ('sample', 'frame').
    """
    speeds = dataset['speed'].values
    unique_speeds = np.unique(speeds)
    config = dataset.attrs['config']
    start, end = config['offsets']
    times = dataset['time'].values

    # Precompute masks for unique speeds
    mask_dict = {}
    for speed in unique_speeds:
        t_start, t_end = time_window(
            speed, from_column=from_column, to_column=to_column, start=start, end=end
        )
        mask_dict[speed] = mask_between_seconds(t_start, t_end, times)

    # Map masks to each sample based on its speed
    masks = np.array([mask_dict[speed] for speed in speeds])

    # Create a DataArray for the masks
    mask_da = xr.DataArray(
        data=masks,
        dims=('sample', 'frame'),
        coords={'sample': dataset['sample'], 'frame': dataset['frame']},
    )

    return mask_da


def peak_responses_angular(
    dataset: xr.Dataset,
    norm: xr.DataArray = None,
    from_degree: float = None,
    to_degree: float = None,
) -> xr.DataArray:
    """
    Compute peak responses and make them complex over angles.

    Args:
        dataset: Input dataset.
        norm: Normalization array.
        from_degree: Starting degree for masking.
        to_degree: Ending degree for masking.

    Returns:
        Complex-valued peak responses.
    """
    peak = peak_responses(
        dataset, norm=norm, from_degree=from_degree, to_degree=to_degree
    )

    # Make complex over angles
    angles = peak['angle'].values
    radians = np.deg2rad(angles)
    # Expand dimensions to match broadcasting
    radians = radians[np.newaxis, :, np.newaxis]
    complex_peak = peak * np.exp(1j * radians)

    return complex_peak


def direction_selectivity_index(
    dataset: xr.Dataset,
    average: bool = True,
    norm: xr.DataArray = None,
    from_degree: float = None,
    to_degree: float = None,
) -> xr.DataArray:
    """
    Compute Direction Selectivity Index (DSI).

    Args:
        dataset: Input dataset.
        average: Whether to average over 'width' and 'speed'.
        norm: Normalization array.
        from_degree: Starting degree for masking.
        to_degree: Ending degree for masking.

    Returns:
        Direction Selectivity Index.
    """
    view = peak_responses_angular(
        dataset, norm=norm, from_degree=from_degree, to_degree=to_degree
    )
    view = view.set_index(sample=["angle", "width", "intensity", "speed"]).unstack(
        "sample"
    )

    # Compute vector sum over 'angle'
    vector_sum = view.sum(dim='angle')
    vector_length = np.abs(vector_sum)

    # Normalization: sum of absolute responses
    normalization = np.abs(view).sum(dim='angle').max(dim='intensity')
    dsi = vector_length / (normalization + 1e-15)

    if average:
        # Average over 'width' and 'speed'
        dsi = dsi.mean(dim=['width', 'speed'])

    return dsi.squeeze()


# -- plot ------------


def prepare_dsi_data(
    dsis, cell_types, sorted_type_list, known_on_off_first, sort_descending
):
    """
    Prepare DSI data for plotting.

    Args:
        dsis: Array of DSI values.
        cell_types: Array of cell type labels.
        sorted_type_list: List of cell types in desired order.
        known_on_off_first: Whether to sort known ON/OFF types first.
        sort_descending: Whether to sort DSIs in descending order.

    Returns:
        Tuple of prepared DSIs and cell types.
    """
    if known_on_off_first:
        sorted_type_list = nodes_edges_utils.nodes_list_sorting_on_off_unknown(cell_types)

    if sorted_type_list is not None:
        dsis = nodes_edges_utils.sort_by_mapping_lists(
            cell_types, sorted_type_list, dsis, axis=0
        )
        cell_types = np.array(sorted_type_list)

    if sort_descending:
        medians = np.median(dsis, axis=(-2, -1))
        index = np.argsort(medians)[::-1]
        dsis = dsis[index]
        cell_types = cell_types[index]

    return dsis, cell_types


def dsi_violins(
    dsis,
    cell_types,
    scatter_best=False,
    scatter_all=True,
    cmap=None,
    colors=None,
    color="b",
    figsize=[10, 1],
    fontsize=6,
    showmeans=False,
    showmedians=True,
    sorted_type_list=None,
    sort_descending=False,
    known_on_off_first=True,
    scatter_kwargs={},
    **kwargs,
):
    """
    Create violin plots for Direction Selectivity Index (DSI) data.

    Args:
        dsis: Array of DSI values.
        cell_types: Array of cell type labels.
        scatter_best: Whether to scatter the best points.
        scatter_all: Whether to scatter all points.
        cmap: Colormap for the violins.
        colors: Specific colors for the violins.
        color: Default color if colors is None and cmap is None.
        figsize: Figure size.
        fontsize: Font size for labels.
        showmeans: Whether to show means on violins.
        showmedians: Whether to show medians on violins.
        sorted_type_list: List of cell types in desired order.
        sort_descending: Whether to sort DSIs in descending order.
        known_on_off_first: Whether to sort known ON/OFF types first.
        **kwargs: Additional keyword arguments for violin_groups.

    Returns:
        Tuple of (figure, axis, colors, prepared DSIs)
    """
    dsis, cell_types = prepare_dsi_data(
        dsis, cell_types, sorted_type_list, known_on_off_first, sort_descending
    )

    if len(dsis.shape) == 1:
        dsis = dsis[None, None, :]
    elif len(dsis.shape) == 2:
        dsis = dsis[:, None]

    if colors is None and cmap is None and color is not None:
        colors = (color,)

    fig, ax, colors = violin_groups(
        dsis,
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
        **kwargs,
    )

    if dsis.shape[1] == 1:
        plt_utils.scatter_on_violins_with_best(
            dsis.T.squeeze(), ax, scatter_best, scatter_all, **scatter_kwargs
        )

    return fig, ax, colors, dsis


def dsi_violins_on_and_off(
    dsis: xr.DataArray,
    cell_types: xr.DataArray,
    scatter_best=False,
    scatter_all=True,
    bold_output_type_labels=False,
    output_cell_types=None,
    known_on_off_first=True,
    sorted_type_list=None,
    figsize=[10, 1],
    ylim=(0, 1),
    color_known_types=True,
    fontsize=6,
    fig=None,
    axes=None,
    **kwargs,
):
    """
    Plot Direction Selectivity Index (DSI) for ON and OFF intensities.

    Args:
        dsis: DataArray of DSI values.
        cell_types: DataArray of cell type labels.
        scatter_best: Whether to scatter the best points.
        scatter_all: Whether to scatter all points.
        bold_output_type_labels: Whether to bold output type labels.
        output_cell_types: Cell types to output.
        known_on_off_first: Whether to sort known ON/OFF types first.
        sorted_type_list: List of cell types in desired order.
        figsize: Figure size.
        ylim: Y-axis limits.
        color_known_types: Whether to color known cell types.
        fontsize: Font size for labels.
        fig: Existing figure to use.
        axes: Existing axes to use.
        **kwargs: Additional keyword arguments for dsi_violins.

    Returns:
        Tuple of (figure, (ax1, ax2))
    """
    if len(dsis.shape) == 2:
        dsis = dsis[None, :]

    if fig is None or axes is None:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        plt_utils.rm_spines(axes[0], spines=("bottom",))

    for i, intensity in enumerate([1, 0]):
        color = ON if intensity == 1 else OFF
        _, ax, *_ = dsi_violins(
            dsis=dsis.sel(intensity=intensity).values.T,
            cell_types=cell_types.values,
            color=color,
            fig=fig,
            ax=axes[i],
            fontsize=fontsize,
            sorted_type_list=sorted_type_list,
            scatter_best=scatter_best,
            scatter_all=scatter_all,
            known_on_off_first=known_on_off_first,
            **kwargs,
        )

        ax.grid(False)
        ax.set_ylim(*ylim)
        plt_utils.trim_axis(ax)
        plt_utils.set_spine_tick_params(
            ax, tickwidth=0.5, ticklength=3, ticklabelpad=2, spinewidth=0.5
        )

        if bold_output_type_labels:
            plt_utils.boldify_labels(output_cell_types, ax)

        if color_known_types:
            plt_utils.color_labels(["T4a", "T4b", "T4c", "T4d"], ON, ax)
            plt_utils.color_labels(["T5a", "T5b", "T5c", "T5d"], OFF, ax)

    # axes[0].set_xticks([])
    axes[0].set_yticks(np.arange(0, 1.2, 0.5))
    axes[1].set_yticks(np.arange(0, 1.2, 0.5))
    axes[1].invert_yaxis()

    return fig, axes


def dsi_correlation_to_known(
    dsis: xr.DataArray, max_aggregate_dims=("intensity",)
) -> xr.DataArray:
    """
    Compute the correlation between predicted DSIs and known DSIs.

    Args:
        dsis: DataArray containing DSIs for ON and OFF intensities.
            Should have dimensions including 'intensity' and 'neuron',
            and a coordinate 'cell_type'.
        max_aggregate_dims: Dimensions to max-aggregate before computing correlation.

    Returns:
        Correlation between predicted and known DSIs.

    Note:
        Known DSIs are binary (0 or 1) based on whether the cell type
        is known to be motion-tuned.
    """
    # Ensure the 'intensity' dimension has length 2
    assert dsis.sizes['intensity'] == 2, "Dimension 'intensity' should have length 2"

    # Retrieve ground truth motion tuning information
    motion_tuning = groundtruth_utils.motion_tuning
    known_dsi_types = groundtruth_utils.known_dsi_types

    # Select dsis for known cell types
    dsis_for_known = dsis.where(dsis['cell_type'].isin(known_dsi_types), drop=True).max(
        dim=max_aggregate_dims
    )

    # Construct ground truth motion tuning array
    groundtruth_mt = xr.DataArray(
        [
            1.0 if ct in motion_tuning else 0.0
            for ct in dsis_for_known['cell_type'].values
        ],
        coords={'neuron': dsis_for_known['neuron']},
        dims=['neuron'],
    )

    # Compute correlation along 'neuron' dimension
    corr_dsi = xr.corr(dsis_for_known, groundtruth_mt, dim='neuron')

    return corr_dsi


def correlation_to_known_tuning_curves(
    dataset: xr.Dataset, absmax: bool = False
) -> xr.DataArray:
    """
    Compute correlation between predicted and known tuning curves.

    Args:
        dataset: Input dataset.
        absmax: If True, maximize magnitude of correlation regardless of sign.

    Returns:
        Correlation values for each cell type.
    """
    tuning = peak_responses(dataset)
    gt_tuning = get_known_tuning_curves(
        ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"], np.arange(0, 360, 30)
    )

    tuning = (
        tuning.set_index(sample=["angle", "intensity", "width", "speed"])
        .unstack("sample")
        .fillna(0.0)
        .custom.where(cell_type=["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"])
    )

    # reset the neuron axis to make it compatible with the ground truth tuning curves
    tuning["neuron"] = np.arange(tuning.coords["neuron"].size)

    correlation = xr.corr(tuning, gt_tuning, dim="angle")
    correlation = correlation.fillna(0.0)
    if absmax:
        # take speed and width that maximize the magnitude of the correlation, regardless
        # of the sign
        argmax = np.abs(correlation).argmax(dim=("speed", "width"))
    else:
        # take speed and width that maximize the correlation as an experimentalist
        # would do
        argmax = correlation.argmax(dim=("speed", "width"))
    correlation = correlation.isel(argmax)
    return correlation


def get_known_tuning_curves(cell_types: List[str], angles: np.ndarray) -> xr.DataArray:
    """
    Retrieve ground truth tuning curves for specified cell types.

    Args:
        cell_types: List of cell type names.
        angles: Array of angles to interpolate curves to.

    Returns:
        DataArray of interpolated ground truth tuning curves.
    """
    gt_angles = np.arange(0, 360, 30)
    tuning_curves = []

    for cell_type in cell_types:
        gt_tuning = groundtruth_utils.tuning_curves[cell_type]
        interp_func = interp1d(
            gt_angles, gt_tuning, kind='cubic', fill_value="extrapolate"
        )
        gt_tuning = interp_func(angles)
        tuning_curves.append(gt_tuning)

    dataset = xr.DataArray(
        np.array(tuning_curves),
        dims=['neuron', 'angle'],
        coords={'cell_type': ("neuron", cell_types), 'angle': angles},
    )

    return dataset


def preferred_direction(
    dataset: xr.Dataset,
    average: bool = True,
    norm: xr.DataArray = None,
    from_degree: float = None,
    to_degree: float = None,
) -> xr.DataArray:
    """
    Compute the preferred direction based on peak responses.

    Args:
        dataset: Input dataset.
        average: Whether to average over certain dimensions.
        norm: Normalization array.
        from_degree: Starting degree for masking.
        to_degree: Ending degree for masking.

    Returns:
        Preferred direction angles in radians.
    """
    view = peak_responses_angular(
        dataset, norm=norm, from_degree=from_degree, to_degree=to_degree
    )
    view = view.set_index(sample=["angle", "width", "intensity", "speed"]).unstack(
        "sample"
    )

    # Compute vector sum over 'angle'
    vector_sum = view.sum(dim='angle')
    theta_pref = np.angle(vector_sum)

    if average:
        # Sum over 'width' and 'speed' before computing angle
        vector_sum = view.sum(dim=['width', 'speed', 'angle'])
        theta_pref = np.angle(vector_sum)

    vector_sum.data = theta_pref
    return vector_sum


def angular_distance_to_known(pds: xr.DataArray) -> xr.DataArray:
    """
    Compute angular distance between predicted and known preferred directions for T4/T5.

    Args:
        pds: Preferred directions for cells.

    Returns:
        Angular distances to known preferred directions.
    """
    t4s = pds.custom.where(cell_type=["T4a", "T4b", "T4c", "T4d"], intensity=1)
    t4_distances = angular_distances(t4s, np.array([np.pi, 0, np.pi / 2, 3 * np.pi / 2]))
    t5s = pds.custom.where(cell_type=["T5a", "T5b", "T5c", "T5d"], intensity=0)
    t5_distances = angular_distances(t5s, np.array([np.pi, 0, np.pi / 2, 3 * np.pi / 2]))
    # concatenate both xarrays again in the neuron dimension, drop intensity
    return xr.concat(
        [t4_distances.drop('intensity'), t5_distances.drop('intensity')], dim='neuron'
    )


def angular_distances(x: xr.DataArray, y: np.array, upper: float = np.pi) -> xr.DataArray:
    """
    Compute angular distances between two sets of angles.

    Args:
        x: First set of angles.
        y: Second set of angles.
        upper: Upper bound for distance calculation.

    Returns:
        Angular distances.
    """
    assert x.neuron.size == len(y)
    y_da = xr.DataArray(y, dims=['neuron'], coords={'neuron': x.coords['neuron']})

    result = xr.apply_ufunc(
        simple_angle_distance,
        x,
        y_da,
        input_core_dims=[['neuron'], ['neuron']],
        output_core_dims=[['neuron']],
        vectorize=True,
        kwargs={'upper': upper},
    )

    return result


def simple_angle_distance(
    a: np.ndarray, b: np.ndarray, upper: float = np.pi
) -> np.ndarray:
    """
    Calculate element-wise angle distance between 0 and pi radians.

    Args:
        a: First set of angles in radians.
        b: Second set of angles in radians.
        upper: Upper bound for distance calculation.

    Returns:
        Distance between 0 and pi radians.
    """
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)

    # make all angles positive between 0 and 2 * pi
    a = a % (2 * np.pi)
    b = b % (2 * np.pi)

    y = np.zeros_like(a)
    # subtract the smaller angle from the larger one
    mask = a >= b
    y[mask] = a[mask] - b[mask]
    y[~mask] = b[~mask] - a[~mask]

    # map distances between pi and 2 pi to 0 and pi
    y[y > np.pi] = 2 * np.pi - y[y > np.pi]

    # map distances between 0 and pi to 0 and upper
    return y / np.pi * upper


def plot_angular_tuning(
    dataset: xr.Dataset,
    cell_type: int,
    intensity: int,
    figsize: Tuple[float, float] = (1, 1),
    fontsize: int = 5,
    linewidth: float = 1,
    anglepad: float = -7,
    xlabelpad: float = -1,
    groundtruth: bool = True,
    groundtruth_linewidth: float = 1.0,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    peak_responses_da: xr.DataArray = None,
    weighted_average: xr.DataArray = None,
    average_models: bool = False,
    colors: str = None,
    zorder: Union[int, Iterable] = 0,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot angular tuning for a specific cell type and intensity.

    Args:
        dataset: Input dataset.
        cell_type: Neuron index to plot.
        intensity: Intensity level (0 or 1).
        figsize: Figure size.
        fontsize: Font size.
        linewidth: Line width.
        anglepad: Angle padding.
        xlabelpad: X-label padding.
        groundtruth: Whether to plot ground truth.
        groundtruth_linewidth: Line width for ground truth.
        fig: Existing figure.
        ax: Existing axes.
        peak_responses_da: Precomputed peak responses.
        weighted_average: Weights for averaging models.
        average_models: Whether to average across models.
        colors: Color for the plot.
        zorder: Z-order for plotting.
        **kwargs: Additional keyword arguments for plotting.

    Returns:
        The figure and axes objects.
    """
    if peak_responses_da is None:
        peak_responses_da = peak_responses(dataset)

    peak_responses_da = peak_responses_da.set_index(
        sample=["angle", "width", "intensity", "speed"]
    ).unstack("sample")

    # Select the specific cell type
    peak = peak_responses_da.custom.where(cell_type=cell_type, intensity=intensity)

    # Squeeze irrelevant dimensions
    # peak = peak.squeeze(dim=['width', 'intensity', 'speed'], drop=True)

    # Average over speeds
    average_tuning = peak.mean(dim=('speed', 'width'))

    # Average over models if specified
    if average_models and weighted_average is not None:
        average_tuning = average_tuning.weighted(weighted_average).mean(dim='network_id')
    elif average_models:
        average_tuning = average_tuning.mean(dim='network_id')

    color = (ON if intensity == 1 else OFF) if colors is None else colors

    average_tuning = average_tuning / average_tuning.max()

    angles = average_tuning['angle'].values
    fig, ax = polar(
        angles,
        average_tuning.data.squeeze().T,
        figsize=figsize,
        fontsize=fontsize,
        linewidth=linewidth,
        anglepad=anglepad,
        xlabelpad=xlabelpad,
        color=color,
        fig=fig,
        ax=ax,
        zorder=zorder,
        **kwargs,
    )

    if groundtruth and cell_type in groundtruth_utils.tuning_curves:
        r_gt = np.array(groundtruth_utils.tuning_curves[cell_type])
        r_gt = r_gt / np.max(np.abs(r_gt))
        theta_gt = np.arange(0, 360, 360 / len(r_gt))
        polar(
            theta_gt,
            r_gt,
            figsize=figsize,
            fontsize=fontsize,
            linewidth=groundtruth_linewidth,
            anglepad=anglepad,
            xlabelpad=xlabelpad,
            color="k",
            fig=fig,
            ax=ax,
            # **kwargs,
        )

    return fig, ax


def plot_T4_tuning(dataset: xr.Dataset) -> None:
    """
    Plot tuning curves for T4 cells.

    Args:
        dataset: Input dataset.
    """
    fig, axes, _ = plt_utils.get_axis_grid(
        range(4),
        projection="polar",
        aspect_ratio=4,
        figsize=[2.95, 0.83],
        wspace=0.25,
    )
    for i, cell_type in enumerate(["T4a", "T4b", "T4c", "T4d"]):
        plot_angular_tuning(
            dataset,
            cell_type,
            intensity=1,
            fig=fig,
            ax=axes[i],
            groundtruth=True,
            aggregate_models="mean",
            linewidth=1.0,
        )
        axes[i].set_xlabel(cell_type)

    for ax in axes:
        ax.xaxis.label.set_fontsize(5)
        [i.set_linewidth(0.5) for i in ax.spines.values()]
        ax.grid(True, linewidth=0.5)


def plot_T5_tuning(dataset: xr.Dataset) -> None:
    """
    Plot tuning curves for T5 cells.

    Args:
        dataset: Input dataset.
    """
    fig, axes, _ = plt_utils.get_axis_grid(
        range(4),
        projection="polar",
        aspect_ratio=4,
        figsize=[2.95, 0.83],
        wspace=0.25,
    )
    for i, cell_type in enumerate(["T5a", "T5b", "T5c", "T5d"]):
        plot_angular_tuning(
            dataset,
            cell_type,
            intensity=0,
            fig=fig,
            ax=axes[i],
            groundtruth=True,
            aggregate_models="mean",
            linewidth=1.0,
        )
        axes[i].set_xlabel(cell_type)

    for ax in axes:
        ax.xaxis.label.set_fontsize(5)
        [i.set_linewidth(0.5) for i in ax.spines.values()]
        ax.grid(True, linewidth=0.5)


def mask_between_seconds(
    t_start: float,
    t_end: float,
    time: np.ndarray = None,
    t_pre: float = None,
    t_stim: float = None,
    t_post: float = None,
    dt: float = None,
) -> np.ndarray:
    """
    Create a boolean mask for time values between t_start and t_end.

    Args:
        t_start: Start time for the mask.
        t_end: End time for the mask.
        time: Array of time values. If None, it will be generated using other parameters.
        t_pre: Time before stimulus onset.
        t_stim: Stimulus duration.
        t_post: Time after stimulus offset.
        dt: Time step.

    Returns:
        Boolean mask array.

    Note:
        If 'time' is not provided, it will be generated using t_pre, t_stim, t_post,
        and dt.
    """
    time = time if time is not None else np.arange(-t_pre, t_stim + t_post - dt, dt)
    return (time >= t_start) & (time <= t_end)


def time_window(
    speed: float,
    from_column: float = -1.5,
    to_column: float = 1.5,
    start: float = -10,
    end: float = 11,
) -> tuple[float, float]:
    """
    Calculate start and end time when the bar passes from_column to to_column.

    Args:
        speed: Speed in columns/s (5.8deg/s).
        from_column: Starting column in 5.8deg units.
        to_column: Ending column in 5.8deg units.
        start: Starting position in LED units (2.25deg).
        end: Ending position in LED units (2.25deg).

    Returns:
        Tuple containing start and end times.

    Note:
        The function adjusts the to_column by adding a single LED width (2.25 deg)
        to make it symmetric around the central column.
    """
    start_in_columns = start * 2.25 / 5.8  # in 5.8deg
    end_in_columns = end * 2.25 / 5.8  # in 5.8deg

    # Make it symmetric around the central column by adding a single LED width
    to_column += 2.25 / 5.8

    assert abs(start_in_columns) >= abs(from_column)
    assert abs(end_in_columns) >= abs(to_column)

    # Calculate when the edge is at the from_column
    t_start = (abs(start_in_columns) - abs(from_column)) / speed
    # Calculate when it's at the to_column
    t_end = t_start + (to_column - from_column) / speed
    return t_start, t_end
