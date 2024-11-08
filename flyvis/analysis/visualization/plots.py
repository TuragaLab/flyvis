import logging
from contextlib import suppress
from dataclasses import dataclass
from functools import wraps
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps as cm
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import RegularPolygon
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from numpy.typing import NDArray

from flyvis import utils
from flyvis.utils import hex_utils

from . import plt_utils

logging = logging.getLogger(__name__)


def heatmap(
    matrix: np.ndarray,
    xlabels: List[str],
    ylabels: Optional[List[str]] = None,
    size_scale: Union[str, float] = "auto",
    cmap: mpl.colors.Colormap = cm.get_cmap("seismic"),
    origin: Literal["upper", "lower"] = "upper",
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    symlog: Optional[bool] = None,
    cbar_label: str = "",
    log: Optional[bool] = None,
    cbar_height: float = 0.5,
    cbar_width: float = 0.01,
    title: str = "",
    figsize: Tuple[float, float] = [5, 4],
    fontsize: int = 4,
    midpoint: Optional[float] = None,
    cbar: bool = True,
    grid_linewidth: float = 0.5,
    **kwargs,
) -> Tuple[Figure, Axes, Optional[Colorbar], np.ndarray]:
    """
    Create a heatmap scatter plot of the matrix.

    Args:
        matrix: 2D matrix to be plotted.
        xlabels: List of x-axis labels.
        ylabels: List of y-axis labels. If not provided, xlabels will be used.
        size_scale: Size scale of the scatter points. If "auto",
            uses 0.005 * prod(figsize).
        cmap: Colormap for the heatmap.
        origin: Origin of the matrix. Either "upper" or "lower".
        ax: Existing Matplotlib Axes object to plot on.
        fig: Existing Matplotlib Figure object to use.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        symlog: Whether to use symmetric log normalization.
        cbar_label: Label for the colorbar.
        log: Whether to use logarithmic color scaling.
        cbar_height: Height of the colorbar.
        cbar_width: Width of the colorbar.
        title: Title of the plot.
        figsize: Size of the figure.
        fontsize: Font size for labels and ticks.
        midpoint: Midpoint for diverging colormaps.
        cbar: Whether to show the colorbar.
        grid_linewidth: Width of the grid lines.
        **kwargs: Additional keyword arguments.

    Returns:
        A tuple containing the Figure, Axes, Colorbar (if shown), and the input matrix.

    Note:
        This function creates a heatmap scatter plot with various customization options.
        The size of scatter points can be scaled based on the absolute value of the matrix
        elements.
    """
    y, x = np.nonzero(matrix)
    value = matrix[y, x]

    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax=ax, fig=fig, offset=0)

    norm = plt_utils.get_norm(
        symlog=symlog,
        vmin=vmin if vmin is not None else np.nanmin(matrix),
        vmax=vmax if vmax is not None else np.nanmax(matrix),
        log=log,
        midpoint=midpoint,
    )

    size = np.abs(value) * (
        size_scale if size_scale != "auto" else 0.005 * np.prod(figsize)
    )

    ax.scatter(
        x=x,
        y=matrix.shape[0] - y - 1 if origin == "upper" else y,
        s=size,
        c=value,
        cmap=cmap,
        norm=norm,
        marker="s",
        edgecolors="none",
    )

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=fontsize)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ylabels = ylabels if ylabels is not None else xlabels
    ax.set_yticklabels(ylabels[::-1] if origin == "upper" else ylabels, fontsize=fontsize)

    ax.grid(False, "major")
    ax.grid(True, "minor", linewidth=grid_linewidth)
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()[:-1]], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()[:-1]], minor=True)

    ax.set_xlim([-0.5, matrix.shape[1]])
    ax.set_ylim([-0.5, matrix.shape[0]])
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="y", which="minor", left=False)

    cbar_obj = None
    if cbar:
        cbar_obj = plt_utils.add_colorbar_to_fig(
            fig,
            height=cbar_height,
            width=cbar_width,
            cmap=cmap,
            norm=norm,
            fontsize=fontsize,
            label=cbar_label,
            x_offset=15,
        )

    return fig, ax, cbar_obj, matrix


# ---- hex scatter


def hex_scatter(
    u: NDArray,
    v: NDArray,
    values: NDArray,
    max_extent: Optional[int] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (1, 1),
    title: str = "",
    title_y: Optional[float] = None,
    fontsize: int = 5,
    label: str = "",
    labelxy: Union[str, Tuple[float, float]] = "auto",
    label_color: str = "black",
    edgecolor: Optional[str] = None,
    edgewidth: float = 0.5,
    alpha: float = 1,
    fill: Union[bool, int] = False,
    scalarmapper: Optional[mpl.cm.ScalarMappable] = None,
    norm: Optional[mpl.colors.Normalize] = None,
    radius: float = 1,
    origin: Literal["lower", "upper"] = "lower",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    midpoint: Optional[float] = None,
    mode: str = "default",
    orientation: float = np.radians(30),
    cmap: mpl.colors.Colormap = cm.get_cmap("seismic"),
    cbar: bool = True,
    cbar_label: str = "",
    cbar_height: Optional[float] = None,
    cbar_width: Optional[float] = None,
    cbar_x_offset: float = 0.05,
    annotate: bool = False,
    annotate_coords: bool = False,
    annotate_indices: bool = False,
    frame: bool = False,
    frame_hex_width: int = 1,
    frame_color: Optional[Union[str, Tuple[float, float, float, float]]] = None,
    nan_linestyle: str = "-",
    text_color_hsv_threshold: float = 0.8,
    **kwargs,
) -> Tuple[Figure, Axes, Tuple[Optional[Line2D], mpl.cm.ScalarMappable]]:
    """
    Plot a hexagonally arranged data points with coordinates u, v, and coloring color.

    Args:
        u: Array of hex coordinates in u direction.
        v: Array of hex coordinates in v direction.
        values: Array of pixel values per point (u_i, v_i).
        fill: Whether to fill the hex grid around u, v, values.
        max_extent: Maximum extent of the hex lattice shown. When fill=True, the hex
            grid is padded to the maximum extent when above the extent of u, v.
        fig: Matplotlib Figure object.
        ax: Matplotlib Axes object.
        figsize: Size of the figure.
        title: Title of the plot.
        title_y: Y-position of the title.
        fontsize: Font size for text elements.
        label: Label for the plot.
        labelxy: Position of the label. Either "auto" or a tuple of (x, y) coordinates.
        label_color: Color of the label.
        edgecolor: Color of the hexagon edges.
        edgewidth: Width of the hexagon edges.
        alpha: Alpha value for transparency.
        scalarmapper: ScalarMappable object for color mapping.
        norm: Normalization for color mapping.
        radius: Radius of the hexagons.
        origin: Origin of the plot. Either "lower" or "upper".
        vmin: Minimum value for color mapping.
        vmax: Maximum value for color mapping.
        midpoint: Midpoint for color mapping.
        mode: Hex coordinate system mode.
        orientation: Orientation of the hexagons in radians.
        cmap: Colormap for the plot.
        cbar: Whether to show a colorbar.
        cbar_label: Label for the colorbar.
        cbar_height: Height of the colorbar.
        cbar_width: Width of the colorbar.
        cbar_x_offset: X-offset of the colorbar.
        annotate: Whether to annotate hexagons with values.
        annotate_coords: Whether to annotate hexagons with coordinates.
        annotate_indices: Whether to annotate hexagons with indices.
        frame: Whether to add a frame around the plot.
        frame_hex_width: Width of the frame in hexagon units.
        frame_color: Color of the frame.
        nan_linestyle: Line style for NaN values.
        text_color_hsv_threshold: Threshold for text color in HSV space.
        **kwargs: Additional keyword arguments.

    Returns:
        A tuple containing the Figure, Axes, and a tuple of (label_text, scalarmapper).
    """

    def init_plot_and_validate_input(fig, ax):
        nonlocal values, u, v
        fig, ax = plt_utils.init_plot(
            figsize, title, fontsize, ax=ax, fig=fig, title_y=title_y, **kwargs
        )
        ax.set_aspect("equal")
        values = values * np.ones_like(u) if not isinstance(values, Iterable) else values
        if u.shape != v.shape or u.shape != values.shape:
            raise ValueError("shape mismatch of hexal values and coordinates")
        u, v, values = hex_utils.sort_u_then_v(u, v, values)
        return fig, ax

    def apply_max_extent():
        nonlocal u, v, values
        extent = hex_utils.get_extent(u, v) or 1
        if fill:
            u, v, values = hex_utils.pad_to_regular_hex(u, v, values, extent=extent)
        if max_extent is not None and extent > max_extent:
            u, v, values = hex_utils.crop_to_extent(u, v, values, max_extent)
        elif max_extent is not None and extent < max_extent and fill:
            u, v, values = hex_utils.pad_to_regular_hex(u, v, values, extent=max_extent)

    def setup_color_mapping(scalarmapper, norm):
        nonlocal vmin, vmax
        if np.any(values):
            vmin = vmin - 1e-10 if vmin is not None else np.nanmin(values) - 1e-10
            vmax = vmax + 1e-10 if vmax is not None else np.nanmax(values) + 1e-10
        else:
            vmin = 0
            vmax = 1

        if (
            midpoint == 0
            and np.isclose(vmin, vmax, atol=1e-10)
            and np.sign(vmin) == np.sign(vmax)
        ):
            sign = np.sign(vmax)
            if sign > 0:
                vmin = -vmax
            elif sign < 0:
                vmax = -vmin
            else:
                raise ValueError

        if midpoint == 0 and np.isnan(values).all():
            vmin = 0
            vmax = 0

        scalarmapper, norm = plt_utils.get_scalarmapper(
            scalarmapper=scalarmapper,
            cmap=cmap,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            midpoint=midpoint,
        )
        return scalarmapper.to_rgba(values), scalarmapper, norm

    def apply_frame():
        nonlocal u, v, values, color_rgba
        if frame:
            extent = hex_utils.get_extent(u, v) or 1
            _u, _v = hex_utils.get_hex_coords(extent + frame_hex_width)
            framed_color = np.zeros([len(_u)])
            framed_color_rgba = np.zeros([len(_u), 4])
            uv = np.stack((_u, _v), 1)
            _rings = (
                abs(0 - uv[:, 0]) + abs(0 + 0 - uv[:, 0] - uv[:, 1]) + abs(0 - uv[:, 1])
            ) / 2
            mask = np.where(_rings <= extent, True, False)
            framed_color[mask] = values
            framed_color[~mask] = 0.0
            framed_color_rgba[mask] = color_rgba
            framed_color_rgba[~mask] = (
                frame_color if frame_color else np.array([0, 0, 0, 1])
            )
            u, v, color_rgba, values = _u, _v, framed_color_rgba, framed_color

    def draw_hexagons():
        x, y = hex_utils.hex_to_pixel(u, v, mode=mode)
        if origin == "upper":
            y = y[::-1]
        c_mask = np.ma.masked_invalid(values)
        for i, (_x, _y, fc) in enumerate(zip(x, y, color_rgba)):
            if c_mask.mask[i]:
                _hex = RegularPolygon(
                    (_x, _y),
                    numVertices=6,
                    radius=radius,
                    linewidth=edgewidth,
                    orientation=orientation,
                    edgecolor=edgecolor,
                    facecolor="white",
                    alpha=alpha,
                    ls=nan_linestyle,
                )
            else:
                _hex = RegularPolygon(
                    (_x, _y),
                    numVertices=6,
                    radius=radius,
                    linewidth=edgewidth,
                    orientation=orientation,
                    edgecolor=edgecolor or fc,
                    facecolor=fc,
                    alpha=alpha,
                )
            ax.add_patch(_hex)
        return x, y, c_mask

    def add_colorbar():
        if cbar:
            plt_utils.add_colorbar_to_fig(
                fig,
                label=cbar_label,
                width=cbar_width or 0.03,
                height=cbar_height or 0.5,
                x_offset=cbar_x_offset or -2,
                cmap=cmap,
                norm=norm,
                fontsize=fontsize,
                tick_length=1,
                tick_width=0.25,
                rm_outline=True,
            )

    def set_plot_limits(x, y):
        extent = hex_utils.get_extent(u, v) or 1
        if fill:
            u_cs, v_cs = hex_utils.get_hex_coords(extent)
            x_cs, y_cs = hex_utils.hex_to_pixel(u_cs, v_cs, mode=mode)
            if origin == "upper":
                y_cs = y_cs[::-1]
            xmin, xmax = plt_utils.get_lims(x_cs, 1 / extent)
            ymin, ymax = plt_utils.get_lims(y_cs, 1 / extent)
        else:
            xmin, xmax = plt_utils.get_lims(x, 1 / extent)
            ymin, ymax = plt_utils.get_lims(y, 1 / extent)
        if xmin != xmax and ymin != ymax:
            ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax])

    def annotate_hexagons(x, y, c_mask):
        if annotate:
            for i, (_label, _x, _y) in enumerate(zip(values, x, y)):
                if not c_mask.mask[i] and not np.isnan(_label):
                    _textcolor = (
                        "black"
                        if mpl.colors.rgb_to_hsv(color_rgba[i][:-1])[-1]
                        > text_color_hsv_threshold
                        else "white"
                    )
                    ax.annotate(
                        f"{_label:.1F}",
                        fontsize=fontsize,
                        xy=(_x, _y),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha="center",
                        va="center",
                        color=_textcolor,
                    )
        if annotate_coords:
            for _x, _y, _u, _v in zip(x, y, u, v):
                ax.text(
                    _x - 0.45,
                    _y + 0.2,
                    _u,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                )
                ax.text(
                    _x + 0.45,
                    _y + 0.2,
                    _v,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                )
        if annotate_indices:
            for i, (_x, _y) in enumerate(zip(x, y)):
                ax.text(_x, _y, i, ha="center", va="center", fontsize=fontsize)

    def add_label():
        if labelxy == "auto":
            extent = hex_utils.get_extent(u, v) or 1
            u_cs, v_cs = hex_utils.get_hex_coords(extent)
            z = -u_cs + v_cs
            labelu, labelv = min(u_cs[z == 0]) - 1, min(v_cs[z == 0]) - 1
            labelx, labely = hex_utils.hex_to_pixel(labelu, labelv)
            ha = "right" if len(label) < 4 else "center"
            label_text = ax.annotate(
                label,
                (labelx, labely),
                ha=ha,
                va="bottom",
                fontsize=fontsize,
                zorder=1000,
                xycoords="data",
                color=label_color,
            )
        else:
            label_text = ax.text(
                labelxy[0],
                labelxy[1],
                label,
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=fontsize,
                zorder=100,
                color=label_color,
            )
        return label_text

    # Main execution
    fig, ax = init_plot_and_validate_input(fig, ax)
    apply_max_extent()
    color_rgba, scalarmapper, norm = setup_color_mapping(scalarmapper, norm)
    apply_frame()
    x, y, c_mask = draw_hexagons()
    add_colorbar()
    set_plot_limits(x, y)
    ax = plt_utils.rm_spines(ax, rm_xticks=True, rm_yticks=True)
    annotate_hexagons(x, y, c_mask)
    label_text = add_label()

    (xmin, ymin, xmax, ymax) = ax.dataLim.extents
    ax.set_xlim(plt_utils.get_lims((xmin, xmax), 0.01))
    ax.set_ylim(plt_utils.get_lims((ymin, ymax), 0.01))

    return fig, ax, (label_text, scalarmapper)


class SignError(Exception):
    """Raised when kernel signs are inconsistent."""

    pass


@wraps(hex_scatter)
def kernel(
    u: NDArray,
    v: NDArray,
    values: NDArray,
    fontsize: int = 5,
    cbar: bool = True,
    edgecolor: str = "k",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (1, 1),
    midpoint: float = 0,
    annotate: bool = True,
    alpha: float = 0.8,
    annotate_coords: bool = False,
    coord_fs: int = 8,
    cbar_height: float = 0.3,
    cbar_x_offset: float = -1,
    **kwargs,
) -> Tuple[Figure, Axes, Tuple[Optional[Line2D], mpl.cm.ScalarMappable]]:
    """Plot receptive fields with hex_scatter.

    Args:
        u: Array of hex coordinates in u direction.
        v: Array of hex coordinates in v direction.
        color: Array of pixel values per point (u_i, v_i).
        fontsize: Font size for text elements.
        cbar: Whether to show a colorbar.
        edgecolor: Color of the hexagon edges.
        fig: Matplotlib Figure object.
        ax: Matplotlib Axes object.
        figsize: Size of the figure.
        midpoint: Midpoint for color mapping.
        annotate: Whether to annotate hexagons with values.
        alpha: Alpha value for transparency.
        annotate_coords: Whether to annotate hexagons with coordinates.
        coord_fs: Font size for coordinate annotations.
        cbar_height: Height of the colorbar.
        cbar_x_offset: X-offset of the colorbar.
        **kwargs: Additional keyword arguments passed to hex_scatter.

    Returns:
        A tuple containing the Figure, Axes, and a tuple of (label_text, scalarmapper).

    Raises:
        SignError: If signs in the kernel are inconsistent.

    Note:
        Assigns `seismic` as colormap and checks that signs are consistent.
        All arguments except `cmap` can be passed to hex_scatter.
    """

    def check_sign_consistency(values: NDArray) -> None:
        non_zero_signs = set(np.sign(values[np.nonzero(values)]))
        if len(non_zero_signs) > 1:
            raise SignError(f"Inconsistent kernel with signs {non_zero_signs}")

    check_sign_consistency(values)

    hex_scatter_kwargs = {
        'u': u,
        'v': v,
        'values': values,
        'fontsize': fontsize,
        'cbar': cbar,
        'edgecolor': edgecolor,
        'fig': fig,
        'ax': ax,
        'figsize': figsize,
        'midpoint': midpoint,
        'annotate': annotate,
        'alpha': alpha,
        'annotate_coords': annotate_coords,
        'coord_fs': coord_fs,
        'cbar_height': cbar_height,
        'cbar_x_offset': cbar_x_offset,
        'cmap': cm.get_cmap("seismic"),
        **kwargs,
    }

    return hex_scatter(**hex_scatter_kwargs)


def hex_cs(
    extent: int = 5,
    mode: Literal["default", "flat"] = "default",
    annotate_coords: bool = True,
    edgecolor: str = "black",
    **kwargs,
) -> Tuple[Figure, Axes, Tuple[Optional[Line2D], mpl.cm.ScalarMappable]]:
    """Plot a hexagonal coordinate system.

    Args:
        extent: Extent of the hexagonal grid.
        mode: Hex coordinate system mode.
        annotate_coords: Whether to annotate hexagons with coordinates.
        edgecolor: Color of the hexagon edges.
        **kwargs: Additional keyword arguments passed to hex_scatter.

    Returns:
        A tuple containing the Figure, Axes, and a tuple of (label_text, scalarmapper).
    """
    u, v = hex_utils.get_hex_coords(extent)
    return hex_scatter(
        u,
        v,
        1,
        cmap=cm.get_cmap("binary_r"),
        annotate_coords=annotate_coords,
        vmin=0,
        vmax=1,
        edgecolor=edgecolor,
        cbar=False,
        mode=mode,
        **kwargs,
    )


def quick_hex_scatter(
    values: NDArray, cmap: mpl.colors.Colormap = cm.get_cmap("binary_r"), **kwargs
) -> Tuple[Figure, Axes, Tuple[Optional[Line2D], mpl.cm.ScalarMappable]]:
    """Create a hex scatter plot with implicit coordinates.

    Args:
        values: Array of pixel values.
        cmap: Colormap for the plot.
        **kwargs: Additional keyword arguments passed to hex_scatter.

    Returns:
        A tuple containing the Figure, Axes, and a tuple of (label_text, scalarmapper).
    """
    values = utils.tensor_utils.to_numpy(values.squeeze())
    u, v = hex_utils.get_hex_coords(hex_utils.get_hextent(len(values)))
    return hex_scatter(u, v, values, cmap=cmap, **kwargs)


# ------------------------- hex optic flow -------------------------


def hex_flow(
    u: NDArray,
    v: NDArray,
    flow: NDArray,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (1, 1),
    title: str = "",
    cmap: mpl.colors.Colormap = plt_utils.cm_uniform_2d,
    max_extent: Optional[int] = None,
    cwheelradius: float = 0.25,
    mode: Literal["default", "flat"] = "default",
    orientation: float = np.radians(30),
    origin: Literal["lower", "upper"] = "lower",
    fontsize: int = 5,
    cwheel: bool = True,
    cwheelxy: Tuple[float, float] = (),
    cwheelpos: str = "southeast",
    cwheellabelpad: float = -5,
    annotate_r: bool = False,
    annotate_theta: bool = False,
    annotate_coords: bool = False,
    coord_fs: int = 3,
    label: str = "",
    labelxy: Tuple[float, float] = (0, 1),
    vmin: float = -np.pi,
    vmax: float = np.pi,
    edgecolor: Optional[str] = None,
    **kwargs,
) -> Tuple[
    Figure,
    Axes,
    Tuple[
        Optional[Line2D],
        mpl.cm.ScalarMappable,
        Optional[mpl.colorbar.Colorbar],
        Optional[mpl.collections.PathCollection],
    ],
]:
    """Plot a hexagonal lattice with coordinates u, v, and flow.

    Args:
        u: Array of hex coordinates in u direction.
        v: Array of hex coordinates in v direction.
        flow: Array of flow per point (u_i, v_i), shape [2, len(u)].
        fig: Matplotlib Figure object.
        ax: Matplotlib Axes object.
        figsize: Size of the figure.
        title: Title of the plot.
        cmap: Colormap for the plot.
        max_extent: Maximum extent of the hex lattice.
        cwheelradius: Radius of the colorwheel.
        mode: Hex coordinate system mode.
        orientation: Orientation of hexagons in radians.
        origin: Origin of the plot.
        fontsize: Font size for text elements.
        cwheel: Whether to show a colorwheel.
        cwheelxy: Position of the colorwheel.
        cwheelpos: Position of the colorwheel.
        cwheellabelpad: Padding for colorwheel labels.
        annotate_r: Whether to annotate hexagons with magnitude.
        annotate_theta: Whether to annotate hexagons with angle.
        annotate_coords: Whether to annotate hexagons with coordinates.
        coord_fs: Font size for coordinate annotations.
        label: Label for the plot.
        labelxy: Position of the label.
        vmin: Minimum value for color mapping.
        vmax: Maximum value for color mapping.
        edgecolor: Color of the hexagon edges.
        **kwargs: Additional keyword arguments.

    Returns:
        A tuple containing the Figure, Axes, and a tuple of
            (label_text, scalarmapper, colorbar, scatter).

    Note:
        Works largely like hex_scatter, but with 2d-flow instead of 1d-intensities.
    """
    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax, fig)
    ax.set_aspect("equal")

    if max_extent:
        max_extent_index = hex_utils.max_extent_index(u, v, max_extent=max_extent)
        flow = flow[:, max_extent_index]
        u = u[max_extent_index]
        v = v[max_extent_index]

    r = np.linalg.norm(flow, axis=0)
    r /= r.max()
    theta = np.arctan2(flow[1], flow[0])

    vmin = vmin if vmin else theta.min()
    vmax = vmax if vmax else theta.max()
    scalarmapper, _ = plt_utils.get_scalarmapper(
        cmap=cmap, vmin=vmin, vmax=vmax, midpoint=0.0
    )
    color_rgba = scalarmapper.to_rgba(theta)
    color_rgba[:, -1] = r

    x, y = hex_utils.hex_to_pixel(u, v, mode=mode)
    if origin == "upper":
        y = y[::-1]

    def draw_hexagons():
        for _x, _y, c in zip(x, y, color_rgba):
            _hex = RegularPolygon(
                (_x, _y),
                numVertices=6,
                radius=1,
                linewidth=0.5,
                orientation=orientation,
                edgecolor=edgecolor or c,
                facecolor=c,
            )
            ax.add_patch(_hex)

    draw_hexagons()

    if cwheel:
        x_offset, y_offset = cwheelxy or (0, 0)
        cb, cs = plt_utils.add_colorwheel_2d(
            fig,
            [ax],
            radius=cwheelradius,
            pos=cwheelpos,
            sm=scalarmapper,
            fontsize=fontsize,
            x_offset=x_offset,
            y_offset=y_offset,
            N=1024,
            labelpad=cwheellabelpad,
        )

    extent = hex_utils.get_extent(u, v)
    ax.set_xlim(x.min() + x.min() / extent, x.max() + x.max() / extent)
    ax.set_ylim(y.min() + y.min() / extent, y.max() + y.max() / extent)

    ax = plt_utils.rm_spines(ax, rm_xticks=True, rm_yticks=True)

    if annotate_r:
        for _r, _x, _y in zip(r, x, y):
            ax.annotate(
                f"{_r:.2G}",
                fontsize=fontsize,
                xy=(_x, _y),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
            )

    if annotate_theta:
        for _theta, _x, _y in zip(np.degrees(theta), x, y):
            ax.annotate(
                f"{_theta:.2f}",
                fontsize=fontsize,
                xy=(_x, _y),
                xytext=(0, 0),
                textcoords="offset points",
                ha="center",
                va="center",
            )

    if annotate_coords:
        for _x, _y, _u, _v in zip(x, y, u, v):
            ax.annotate(
                _u,
                fontsize=coord_fs,
                xy=(_x, _y),
                xytext=np.array([-0.25, 0.25]),
                textcoords="offset points",
                ha="center",
                va="center",
            )
            ax.annotate(
                _v,
                fontsize=coord_fs,
                xy=(_x, _y),
                xytext=np.array([0.25, 0.25]),
                textcoords="offset points",
                ha="center",
                va="center",
            )

    label_text = None
    if label:
        label_text = ax.text(
            labelxy[0],
            labelxy[1],
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=fontsize,
        )

    if cwheel:
        return fig, ax, (label_text, scalarmapper, cb, cs)
    return fig, ax, (label_text, scalarmapper, None, None)


def quick_hex_flow(
    flow: NDArray, **kwargs
) -> Tuple[
    Figure,
    Axes,
    Tuple[
        Optional[Line2D],
        mpl.cm.ScalarMappable,
        Optional[mpl.colorbar.Colorbar],
        Optional[mpl.collections.PathCollection],
    ],
]:
    """Plot a flow field on a hexagonal lattice with implicit coordinates.

    Args:
        flow: Array of flow values.
        **kwargs: Additional keyword arguments passed to hex_flow.

    Returns:
        A tuple containing the Figure, Axes, and a tuple of
            (label_text, scalarmapper, colorbar, scatter).
    """
    flow = utils.tensor_utils.to_numpy(flow.squeeze())
    u, v = hex_utils.get_hex_coords(hex_utils.get_hextent(flow.shape[-1]))
    return hex_flow(u, v, flow, **kwargs)


# --- cartesian flow plots ---


def flow_to_rgba(flow: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Map cartesian flow to RGBA colors.

    Args:
        flow: Flow field of shape (2, h, w).

    Returns:
        RGBA color representation of the flow field.

    Note:
        The flow magnitude is mapped to the alpha channel, while the flow
        direction is mapped to the color using a uniform 2D colormap.
    """
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()

    X, Y = flow[0], flow[1]
    R = np.sqrt(X * X + Y * Y)
    PHI = np.arctan2(Y, X)
    scalarmapper, _ = plt_utils.get_scalarmapper(
        cmap=plt_utils.cm_uniform_2d, vmin=-np.pi, vmax=np.pi
    )
    rgba = scalarmapper.to_rgba(PHI)
    rgba[:, :, -1] = R / R.max()
    return rgba


def plot_flow(flow: Union[np.ndarray, torch.Tensor]) -> None:
    """Plot cartesian flow.

    Args:
        flow: Flow field of shape (2, h, w).

    Note:
        This function displays the flow field using matplotlib's imshow
        and immediately shows the plot.
    """
    rgba = flow_to_rgba(flow)
    plt.imshow(rgba)
    plt.show()


# ---- TRACES
def traces(
    trace: NDArray,
    x: Optional[NDArray] = None,
    contour: Optional[NDArray] = None,
    legend: Tuple[str, ...] = (),
    smooth: Optional[float] = None,
    stim_line: Optional[NDArray] = None,
    contour_cmap: mpl.colors.Colormap = cm.get_cmap("bone"),
    color: Optional[Union[str, List[str]]] = None,
    label: str = "",
    labelxy: Tuple[float, float] = (0, 1),
    linewidth: float = 1,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    title: str = "",
    highlight_mean: bool = False,
    figsize: Tuple[float, float] = (7, 4),
    fontsize: int = 10,
    ylim: Optional[Tuple[float, float]] = None,
    ylabel: str = "",
    xlabel: str = "",
    legend_frame_alpha: float = 0,
    contour_mode: Literal["full", "top", "bottom"] = "full",
    contour_y_rel: float = 0.06,
    fancy: bool = False,
    scale_pos: Optional[str] = None,
    scale_label: str = "100ms",
    null_line: bool = False,
    zorder_traces: Optional[int] = None,
    zorder_mean: Optional[int] = None,
    **kwargs,
) -> Tuple[Figure, Axes, NDArray, Optional[Line2D]]:
    """Create a line plot with optional contour and smoothing.

    Args:
        trace: 2D array (n_traces, n_points) of trace values.
        x: X-axis values.
        contour: Array of contour values.
        legend: Legend for each trace.
        smooth: Size of smoothing window in percent of #points.
        stim_line: Stimulus line data.
        contour_cmap: Colormap for the contour.
        color: Color(s) for the traces.
        label: Label for the plot.
        labelxy: Position of the label.
        linewidth: Width of the trace lines.
        ax: Matplotlib Axes object.
        fig: Matplotlib Figure object.
        title: Title of the plot.
        highlight_mean: Whether to highlight the mean trace.
        figsize: Size of the figure.
        fontsize: Font size for text elements.
        ylim: Y-axis limits.
        ylabel: Y-axis label.
        xlabel: X-axis label.
        legend_frame_alpha: Alpha value for the legend frame.
        contour_mode: Mode for contour plotting.
        contour_y_rel: Relative Y position for contour in "top" or "bottom" mode.
        fancy: Whether to use fancy styling.
        scale_pos: Position of the scale bar.
        scale_label: Label for the scale bar.
        null_line: Whether to draw a null line at y=0.
        zorder_traces: Z-order for traces.
        zorder_mean: Z-order for mean trace.
        **kwargs: Additional keyword arguments.

    Returns:
        A tuple containing the Figure, Axes, smoothed trace, and label text.

    Note:
        This function creates a line plot with various options for customization,
        including contour plotting and trace smoothing.
    """
    trace = np.atleast_2d(np.array(trace))

    if np.ma.masked_invalid(trace).mask.any():
        logging.debug("Invalid values encountered in trace.")

    # Smooth traces.
    if smooth:
        smooth = int(smooth * trace.shape[1])
        ylabel += " (smoothed)"
        trace = plt_utils.avg_pool(trace, smooth)
        if x is not None:
            x = x[0::smooth][: trace.shape[1]]

    shape = trace.shape

    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax=ax, fig=fig)

    legends = legend if len(legend) == shape[0] else ("",) * shape[0]

    if len(np.shape(color)) <= 1:
        colors = (color,) * shape[0]
    elif len(color) == shape[0]:
        colors = color
    else:
        colors = (None,) * shape[0]

    # Plot traces.
    iterations = np.arange(trace.shape[1]) if x is None else x
    for i, _trace in enumerate(trace):
        ax.plot(
            iterations,
            _trace,
            label=legends[i],
            c=colors[i],
            linewidth=linewidth,
            zorder=zorder_traces,
        )

    if highlight_mean:
        ax.plot(
            iterations,
            np.mean(trace, axis=0),
            linewidth=0.5,
            c="k",
            label="average",
            zorder=zorder_mean,
        )

    if contour is not None and contour_mode is not None:
        ylim = ylim or plt_utils.get_lims(
            np.array([
                min(contour.min(), trace.min()),
                max(contour.max(), trace.max()),
            ]),
            0.1,
        )

        _x = np.arange(len(contour)) if x is None or len(x) != len(contour) else x
        if contour_mode == "full":
            contour_y_range = (-20_000, 20_000)
        elif contour_mode == "top":
            yrange = ylim[1] - ylim[0]
            contour_y_range = (ylim[1], ylim[1] + yrange * contour_y_rel)
            ylim = (ylim[0], contour_y_range[1])
        elif contour_mode == "bottom":
            yrange = ylim[1] - ylim[0]
            contour_y_range = (ylim[0] - yrange * contour_y_rel, ylim[0])
            ylim = (contour_y_range[0], ylim[1])

        _y = np.linspace(*contour_y_range, 100)
        Z = np.tile(contour, (len(_y), 1))
        ax.contourf(
            _x,
            _y,
            Z,
            cmap=contour_cmap,
            levels=2,
            alpha=0.3,
            vmin=0,
            vmax=1,
        )

        if stim_line is not None:
            ax.plot(x, contour, color="k", linestyle="--")

    # Cosmetics.
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if null_line:
        ax.hlines(
            0,
            -20_000,
            20_000,
            color="0.5",
            zorder=-1,
            linewidth=0.5,
        )
    ax.set_xlim(*plt_utils.get_lims(iterations, 0.01))
    ax.tick_params(labelsize=fontsize)
    if legend:
        ax.legend(
            fontsize=fontsize,
            edgecolor="white",
            **dict(
                labelspacing=0.0,
                framealpha=legend_frame_alpha,
                borderaxespad=0.1,
                borderpad=0.1,
                handlelength=1,
                handletextpad=0.3,
            ),
        )
    if ylim is not None:
        ax.set_ylim(*ylim)

    label_text = None
    if label != "":
        label_text = ax.text(
            labelxy[0],
            labelxy[1],
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=fontsize,
        )

    if scale_pos and not any([isinstance(a, AnchoredSizeBar) for a in ax.artists]):
        scalebar = AnchoredSizeBar(
            ax.transData,
            size=0.1,
            label=scale_label,
            loc=scale_pos,
            pad=0.4,
            frameon=False,
            size_vertical=0.01 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
            fontproperties=dict(size=fontsize),
        )
        ax.add_artist(scalebar)

    if fancy:
        plt_utils.rm_spines(ax, ("left", "bottom"), rm_yticks=True, rm_xticks=True)

    return fig, ax, trace, label_text


def grouped_traces(
    trace_groups: List[np.ndarray],
    x: Optional[np.ndarray] = None,
    legend: Tuple[str, ...] = (),
    color: Optional[Union[str, List[str]]] = None,
    linewidth: float = 1,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    title: str = "",
    highlight_mean: bool = False,
    figsize: Tuple[float, float] = (7, 4),
    fontsize: int = 10,
    ylim: Optional[Tuple[float, float]] = None,
    ylabel: str = "",
    xlabel: str = "",
    legend_frame_alpha: float = 0,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Create a line plot with grouped traces.

    Args:
        trace_groups: List of 2D arrays, each containing trace values.
        x: X-axis values.
        legend: Legend for each trace group.
        color: Color(s) for the trace groups.
        linewidth: Width of the trace lines.
        ax: Matplotlib Axes object.
        fig: Matplotlib Figure object.
        title: Title of the plot.
        highlight_mean: Whether to highlight the mean trace.
        figsize: Size of the figure.
        fontsize: Font size for text elements.
        ylim: Y-axis limits.
        ylabel: Y-axis label.
        xlabel: X-axis label.
        legend_frame_alpha: Alpha value for the legend frame.
        **kwargs: Additional keyword arguments passed to traces().

    Returns:
        A tuple containing the Figure and Axes objects.
    """
    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax=ax, fig=fig)

    legends = legend if len(legend) == len(trace_groups) else ("",) * len(trace_groups)

    if color is None:
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = [color_cycle[i % len(color_cycle)] for i in range(len(trace_groups))]
    elif len(np.shape(color)) <= 1:
        colors = (color,) * len(trace_groups)
    elif len(color) == len(trace_groups) or (
        len(trace_groups) == 1 and len(color) == trace_groups[0].shape[0]
    ):
        colors = color
    else:
        raise ValueError(
            "`color` should be a single value, an iterable of length "
            f"`traces.shape[0]`, or None. Got {color} of shape {np.shape(color)}. "
            f"Expected {np.shape(trace_groups)}."
        )

    for i, _trace in enumerate(trace_groups):
        fig, ax, *_ = traces(
            trace=_trace,
            x=x,
            legend=(),
            color=colors[i],
            linewidth=linewidth,
            ax=ax,
            fig=fig,
            title=title,
            highlight_mean=highlight_mean,
            figsize=figsize,
            fontsize=fontsize,
            ylim=ylim,
            ylabel=ylabel,
            xlabel=xlabel,
            legend_frame_alpha=legend_frame_alpha,
            **kwargs,
        )
    if legend:
        custom_lines = [Line2D([0], [0], color=c) for c in colors]
        ax.legend(
            custom_lines,
            legends,
            fontsize=fontsize,
            edgecolor="white",
            **dict(
                labelspacing=0.0,
                framealpha=legend_frame_alpha,
                borderaxespad=0.1,
                borderpad=0.1,
                handlelength=1,
                handletextpad=0.3,
            ),
        )
    return fig, ax


# -- violins ----
def get_violin_x_locations(
    n_groups: int, n_random_variables: int, violin_width: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate x-axis locations for violin plots.

    Args:
        n_groups: Number of groups.
        n_random_variables: Number of random variables.
        violin_width: Width of each violin plot.

    Returns:
        A tuple containing:
        - np.ndarray: 2D array of violin locations.
        - np.ndarray: 1D array of first violin locations.
    """
    violin_locations = np.zeros([n_groups, n_random_variables])
    first_violins_location = np.arange(0, n_groups * n_random_variables, n_groups)
    for j in range(n_groups):
        violin_locations[j] = first_violins_location + j * violin_width

    return violin_locations, first_violins_location


@dataclass
class ViolinData:
    """
    Container for violin plot data.

    Attributes:
        data: np.ndarray
            The data used for creating violin plots.
        locations: np.ndarray
            The x-axis locations of the violin plots.
        colors: np.ndarray
            The colors used for the violin plots.
    """

    data: np.ndarray
    locations: np.ndarray
    colors: np.ndarray


def violin_groups(
    values: np.ndarray,
    xticklabels: Optional[List[str]] = None,
    pvalues: Optional[np.ndarray] = None,
    display_pvalues_kwargs: dict = {},
    legend: Union[bool, List[str]] = False,
    legend_kwargs: dict = {},
    as_bars: bool = False,
    colors: Optional[List[str]] = None,
    cmap: mpl.colors.Colormap = mpl.colormaps["tab10"],
    cstart: float = 0,
    cdist: float = 1,
    figsize: Tuple[float, float] = (10, 1),
    title: str = "",
    ylabel: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    rotation: float = 90,
    width: float = 0.7,
    fontsize: int = 6,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    showmeans: bool = False,
    showmedians: bool = True,
    grid: bool = False,
    scatter: bool = True,
    scatter_radius: float = 3,
    scatter_edge_color: Optional[str] = None,
    scatter_edge_width: float = 0.5,
    violin_alpha: float = 0.5,
    violin_marker_lw: float = 0.5,
    violin_marker_color: str = "k",
    color_by: Literal["groups", "experiments"] = "groups",
    zorder_mean_median: int = 5,
    zorder_min_max: int = 5,
    mean_median_linewidth: float = 0.5,
    mean_median_color: str = "k",
    mean_median_bar_length: Optional[float] = None,
    **kwargs,
) -> Tuple[Figure, Axes, ViolinData]:
    """
    Create violin plots or bar plots for grouped data.

    Args:
        values: Array of shape (n_random_variables, n_groups, n_samples).
        xticklabels: Labels for the x-axis ticks (random variables).
        pvalues: Array of p-values for statistical significance.
        display_pvalues_kwargs: Keyword arguments for displaying p-values.
        legend: If True or a list, display a legend with group labels.
        legend_kwargs: Keyword arguments for the legend.
        as_bars: If True, create bar plots instead of violin plots.
        colors: List of colors for the violins or bars.
        cmap: Colormap to use when colors are not provided.
        cstart: Starting point in the colormap.
        cdist: Distance between colors in the colormap.
        figsize: Size of the figure (width, height).
        title: Title of the plot.
        ylabel: Label for the y-axis.
        ylim: Limits for the y-axis (min, max).
        rotation: Rotation angle for x-axis labels.
        width: Width of the violins or bars.
        fontsize: Font size for labels and ticks.
        ax: Existing Axes object to plot on.
        fig: Existing Figure object to use.
        showmeans: If True, show mean lines on violins.
        showmedians: If True, show median lines on violins.
        grid: If True, display a grid.
        scatter: If True, scatter individual data points.
        scatter_radius: Size of scattered points.
        scatter_edge_color: Color of scattered point edges.
        scatter_edge_width: Width of scattered point edges.
        violin_alpha: Alpha (transparency) of violin plots.
        violin_marker_lw: Line width of violin markers.
        violin_marker_color: Color of violin markers.
        color_by: Whether to color by "groups" or "experiments".
        zorder_mean_median: Z-order for mean and median lines.
        zorder_min_max: Z-order for min and max lines.
        mean_median_linewidth: Line width for mean and median lines.
        mean_median_color: Color for mean and median lines.
        mean_median_bar_length: Length of mean and median bars.
        **kwargs: Additional keyword arguments.

    Returns:
        A tuple containing:
        - Figure: The matplotlib Figure object.
        - Axes: The matplotlib Axes object.
        - ViolinData: A custom object containing plot data.

    Raises:
        ValueError: If color specifications are invalid.

    Note:
        This function creates either violin plots or bar plots for grouped data,
        with options for customizing colors, scatter plots, and statistical annotations.
    """
    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax, fig)
    if grid:
        ax.yaxis.grid(zorder=-100)

    def plot_bar(X: float, values: np.ndarray, color: str) -> mpl.patches.Rectangle:
        handle = ax.bar(x=X, width=width, height=np.mean(values), color=color, zorder=1)
        return handle

    def plot_violin(
        X: float, values: np.ndarray, color: str
    ) -> mpl.collections.PolyCollection:
        if isinstance(values, np.ma.core.MaskedArray):
            values = values[~values.mask]

        parts = ax.violinplot(
            values,
            positions=[X],
            widths=width,
            showmedians=showmedians,
            showmeans=showmeans,
        )
        # Color the bodies.
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(violin_alpha)
            pc.set_zorder(0)
        # Color the lines.
        parts["cbars"].set_color(violin_marker_color)
        parts["cbars"].set_linewidth(violin_marker_lw)
        parts["cbars"].set_zorder(zorder_min_max)
        parts["cmaxes"].set_color(violin_marker_color)
        parts["cmaxes"].set_linewidth(violin_marker_lw)
        parts["cmaxes"].set_zorder(zorder_min_max)
        parts["cmins"].set_color(violin_marker_color)
        parts["cmins"].set_linewidth(violin_marker_lw)
        parts["cmins"].set_zorder(zorder_min_max)
        if "cmeans" in parts:
            parts["cmeans"].set_color(mean_median_color)
            parts["cmeans"].set_linewidth(mean_median_linewidth)
            parts["cmeans"].set_zorder(zorder_mean_median)
            if mean_median_bar_length is not None:
                (_, y0), (_, y1) = parts["cmeans"].get_segments()[0]
                (x0_vert, _), _ = parts["cbars"].get_segments()[0]
                parts["cmeans"].set_segments([
                    [
                        [x0_vert - mean_median_bar_length * width / 2, y0],
                        [x0_vert + mean_median_bar_length * width / 2, y1],
                    ]
                ])
        if "cmedians" in parts:
            parts["cmedians"].set_color(mean_median_color)
            parts["cmedians"].set_linewidth(mean_median_linewidth)
            parts["cmedians"].set_zorder(zorder_mean_median)
            if mean_median_bar_length is not None:
                (_, y0), (_, y1) = parts["cmedians"].get_segments()[0]
                (x0_vert, _), _ = parts["cbars"].get_segments()[0]
                parts["cmedians"].set_segments([
                    [
                        [x0_vert - mean_median_bar_length * width / 2, y0],
                        [x0_vert + mean_median_bar_length * width / 2, y1],
                    ]
                ])
        return parts["bodies"][0]

    shape = np.array(values).shape
    n_random_variables, n_groups = shape[0], shape[1]

    violin_locations, first_violins_location = get_violin_x_locations(
        n_groups, n_random_variables, violin_width=width
    )
    X = violin_locations.T

    if colors is None:
        if color_by == "groups":
            C = np.asarray([cmap(cstart + i * cdist) for i in range(n_groups)]).reshape(
                n_groups, 4
            )
        elif color_by == "experiments":
            C = np.asarray([
                cmap(cstart + i * cdist) for i in range(n_random_variables)
            ]).reshape(n_random_variables, 4)
        else:
            raise ValueError("Invalid color_by option")
    elif isinstance(colors, Iterable):
        if (
            color_by == "groups"
            and len(colors) == n_groups
            or color_by == "experiments"
            and len(colors) == n_random_variables
        ):
            C = colors
        else:
            raise ValueError("Invalid colors length")
    else:
        raise ValueError("Invalid colors specification")

    handles = []

    for i in range(n_random_variables):
        for j in range(n_groups):
            _color = C[i] if color_by == "experiments" else C[j]

            h = (
                plot_bar(X[i, j], values[i, j], _color)
                if as_bars
                else plot_violin(X[i, j], values[i, j], _color)
            )
            handles.append(h)

            if scatter:
                lims = plt_utils.get_lims(
                    (-width / (2 * n_groups), width / (2 * n_groups)), -0.05
                )
                xticks = np.ones_like(values[i][j]) * X[i, j]
                ax.scatter(
                    xticks + np.random.uniform(*lims, size=len(xticks)),
                    values[i][j],
                    facecolor="none",
                    edgecolor=scatter_edge_color or _color,
                    s=scatter_radius,
                    linewidth=scatter_edge_width,
                    zorder=2,
                )

    if legend:
        ax.legend(handles, legend, **legend_kwargs)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.tick_params(axis="both", which="major", labelsize=fontsize)

    if xticklabels is not None:
        ax.set_xticks(first_violins_location + (n_groups - 1) / 2 * width)
        ax.set_xticklabels(xticklabels, rotation=rotation)

    with suppress(ValueError):
        ax.set_xlim(np.min(X - width), np.max(X + width))

    ax.set_ylabel(ylabel or "", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    if pvalues is not None:
        plt_utils.display_pvalues(
            ax, pvalues, xticklabels, values, **display_pvalues_kwargs
        )

    return fig, ax, ViolinData(values, X, colors)


# ---- POLAR


def plot_complex(
    z: complex,
    marker: str = "s",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (1, 1),
    fontsize: int = 5,
) -> Tuple[Figure, Axes]:
    """
    Plot a complex number on a polar plot.

    Args:
        z: Complex number to plot.
        marker: Marker style for the point.
        fig: Existing figure to plot on.
        ax: Existing axes to plot on.
        figsize: Size of the figure.
        fontsize: Font size for text elements.

    Returns:
        A tuple containing the Figure and Axes objects.
    """
    fig, ax = plt_utils.init_plot(
        figsize=figsize, projection="polar", fontsize=fontsize, fig=fig, ax=ax
    )

    theta = np.angle(z)
    r = np.abs(z)

    ax.plot([0, theta], [0, r], marker=marker)
    return fig, ax


def plot_complex_vector(
    z0: complex,
    z1: complex,
    marker: str = "s",
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (1, 1),
    fontsize: int = 5,
) -> Tuple[Figure, Axes]:
    """
    Plot a vector between two complex numbers on a polar plot.

    Args:
        z0: Starting complex number.
        z1: Ending complex number.
        marker: Marker style for the points.
        fig: Existing figure to plot on.
        ax: Existing axes to plot on.
        figsize: Size of the figure.
        fontsize: Font size for text elements.

    Returns:
        A tuple containing the Figure and Axes objects.
    """
    fig, ax = plt_utils.init_plot(
        figsize=figsize, projection="polar", fontsize=fontsize, fig=fig, ax=ax
    )

    theta0 = np.angle(z0)
    r0 = np.abs(z0)

    theta = np.angle(z1)
    r = np.abs(z1)

    ax.plot([theta0, theta], [r0, r], marker=marker)
    return fig, ax


def polar(
    theta: NDArray,
    r: NDArray,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    color: Union[str, List[str]] = "b",
    linestyle: str = "-",
    marker: str = "",
    markersize: Optional[float] = None,
    label: Optional[str] = None,
    title: str = "",
    figsize: Tuple[float, float] = (5, 5),
    fontsize: int = 10,
    xlabel: str = "",
    fontweight: Literal[
        "normal", "bold", "light", "ultralight", "heavy", "black", "semibold"
    ] = "normal",
    anglepad: int = -2,
    xlabelpad: int = -3,
    linewidth: float = 2,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    stroke_kwargs: dict = {},
    yticks_off: bool = True,
    zorder: Union[int, List[int]] = 100,
    **kwargs,
) -> Tuple[Figure, Axes]:
    """
    Create a polar tuning plot.

    Args:
        theta: Array of angles in degrees.
        r: Array of radii.
        ax: Matplotlib Axes object.
        fig: Matplotlib Figure object.
        color: Color(s) for the plot.
        linestyle: Line style for the plot.
        marker: Marker style for data points.
        markersize: Size of markers.
        label: Label for the plot.
        title: Title of the plot.
        figsize: Size of the figure.
        fontsize: Font size for text elements.
        xlabel: X-axis label.
        fontweight: Font weight for labels.
        anglepad: Padding for angle labels.
        xlabelpad: Padding for x-axis label.
        linewidth: Width of the plot lines.
        ymin: Minimum y-axis value.
        ymax: Maximum y-axis value.
        stroke_kwargs: Keyword arguments for stroke effects.
        yticks_off: Whether to turn off y-axis ticks.
        zorder: Z-order for plot elements.
        **kwargs: Additional keyword arguments.

    Returns:
        A tuple containing the Figure and Axes objects.

    Note:
        This function creates a polar plot with various customization options.
        It supports multiple traces and custom styling.
    """
    fig, ax = plt_utils.init_plot(
        figsize=figsize,
        title=title,
        fontsize=fontsize,
        ax=ax,
        fig=fig,
        projection="polar",
    )

    if sum(theta) < 100:
        logging.warning("Using radians instead of degrees?")

    closed = theta[-1] % 360 == theta[0]
    theta = theta * np.pi / 180
    if not closed:
        theta = np.append(theta, theta[0])

    r = np.asarray(r)
    if not closed:
        r = np.append(r, np.expand_dims(r[0], 0), axis=0)

    line_effects = None
    if stroke_kwargs:
        line_effects = [
            path_effects.Stroke(**stroke_kwargs),
            path_effects.Normal(),
        ]

    zorder = plt_utils.extend_arg(zorder, int, r, default=0, dim=-1)

    if r.ndim == 2:
        for i, _r in enumerate(r.T):
            if isinstance(color, Iterable):
                if isinstance(color, str) and color.startswith("#"):
                    _color = color
                elif len(color) == r.shape[1]:
                    _color = color[i]
                else:
                    _color = color
            else:
                _color = color

            ax.plot(
                theta,
                _r,
                linewidth=linewidth,
                color=_color,
                linestyle=linestyle,
                marker=marker,
                label=label,
                path_effects=line_effects,
                zorder=zorder[i],
                markersize=markersize,
            )
    elif r.ndim == 1:
        ax.plot(
            theta,
            r,
            linewidth=linewidth,
            color=color,
            linestyle=linestyle,
            marker=marker,
            label=label,
            path_effects=line_effects,
            zorder=zorder,
            markersize=markersize,
        )

    ax.tick_params(axis="both", which="major", labelsize=fontsize, pad=anglepad)
    if yticks_off:
        ax.set_yticks([])
        ax.set_yticklabels([])
    ax.set_xticks([
        0,
        np.pi / 4,
        np.pi / 2,
        3 / 4 * np.pi,
        np.pi,
        5 / 4 * np.pi,
        3 / 2 * np.pi,
        7 / 4 * np.pi,
    ])
    ax.set_xticklabels(["0°", "45°", "90°", "", "", "", "", ""])

    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=xlabelpad, fontweight=fontweight)
    if all((val is not None for val in (ymin, ymax))):
        ax.set_ylim((ymin, ymax))
    plt.setp(ax.spines.values(), color="grey", linewidth=1)
    return fig, ax


def multi_polar(
    theta: np.ndarray,
    r: np.ndarray,
    ax: Optional[Axes] = None,
    fig: Optional[Figure] = None,
    mean_color: str = "b",
    norm: bool = False,
    std: bool = False,
    color: Union[str, List[str], np.ndarray] = "b",
    mean: bool = False,
    linestyle: str = "-",
    marker: str = "",
    label: Union[str, List[str]] = "",
    legend: bool = False,
    title: str = "",
    figsize: Tuple[float, float] = (0.98, 2.38),
    fontsize: int = 5,
    xlabel: str = "",
    fontweight: str = "bold",
    alpha: float = 1,
    anglepad: int = -6,
    xlabelpad: int = -3,
    linewidth: float = 0.75,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    zorder: Optional[Union[int, List[int], np.ndarray]] = None,
    legend_kwargs: Dict[str, Any] = dict(fontsize=5),
    rm_yticks: bool = True,
    **kwargs: Any,
) -> Tuple[Figure, Axes]:
    """
    Create a polar tuning plot.

    Args:
        theta: Angles in degrees.
        r: Radius values. Shape (n_samples, n_values).
        ax: Existing Axes object to plot on. Defaults to None.
        fig: Existing Figure object to use. Defaults to None.
        mean_color: Color for the mean line. Defaults to "b".
        norm: Whether to normalize the radius values. Defaults to False.
        std: Whether to plot standard deviation. Defaults to False.
        color: Color(s) for the lines. Defaults to "b".
        mean: Whether to plot the mean. Defaults to False.
        linestyle: Style of the lines. Defaults to "-".
        marker: Marker style for data points. Defaults to "".
        label: Label(s) for the lines. Defaults to "".
        legend: Whether to show a legend. Defaults to False.
        title: Title of the plot. Defaults to "".
        figsize: Size of the figure. Defaults to (0.98, 2.38).
        fontsize: Font size for text elements. Defaults to 5.
        xlabel: Label for the x-axis. Defaults to "".
        fontweight: Font weight for labels. Defaults to "bold".
        alpha: Alpha value for line transparency. Defaults to 1.
        anglepad: Padding for angle labels. Defaults to -6.
        xlabelpad: Padding for x-axis label. Defaults to -3.
        linewidth: Width of the lines. Defaults to 0.75.
        ymin: Minimum y-axis value. Defaults to None.
        ymax: Maximum y-axis value. Defaults to None.
        zorder: Z-order for drawing. Defaults to None.
        legend_kwargs: Additional keyword arguments for legend.
            Defaults to dict(fontsize=5).
        rm_yticks: Whether to remove y-axis ticks. Defaults to True.
        **kwargs: Additional keyword arguments.

    Returns:
        A tuple containing the Figure and Axes objects.

    Note:
        This function creates a polar plot with multiple traces, optionally showing
        mean and standard deviation.
    """
    fig, ax = plt_utils.init_plot(
        figsize=figsize,
        title=title,
        fontsize=fontsize,
        ax=ax,
        fig=fig,
        projection="polar",
    )
    r = np.atleast_2d(r)
    n_traces = r.shape[0]

    if norm:
        r = r / (r.max(axis=1, keepdims=True) + 1e-15)

    closed = theta[-1] % 360 == theta[0]
    theta = theta * np.pi / 180
    if not closed:
        theta = np.append(theta, theta[0])
        r = np.append(r, np.expand_dims(r[:, 0], 1), axis=1)

    color = [color] * n_traces if not isinstance(color, (list, np.ndarray)) else color
    label = [label] * n_traces if not isinstance(label, (list, np.ndarray)) else label
    zorder = [100] * n_traces if not isinstance(zorder, (list, np.ndarray)) else zorder

    for i, _r in enumerate(r):
        ax.plot(
            theta,
            _r,
            linewidth=linewidth,
            color=color[i],
            linestyle=linestyle,
            marker=marker,
            label=label[i],
            zorder=zorder[i],
            alpha=alpha,
        )

    if mean:
        ax.plot(
            theta,
            r.mean(0),
            linewidth=linewidth,
            color=mean_color,
            linestyle=linestyle,
            marker=marker,
            label="average",
            alpha=alpha,
        )

    if std:
        ax.fill_between(
            theta,
            r.mean(0) - r.std(0),
            r.mean(0) + r.std(0),
            color="0.8",
            alpha=0.5,
            zorder=-1,
        )

    ax.tick_params(axis="both", which="major", labelsize=fontsize, pad=anglepad)
    if rm_yticks:
        ax.set_yticks([])
        ax.set_yticklabels([])
    ax.set_xticks([
        0,
        np.pi / 4,
        np.pi / 2,
        3 / 4 * np.pi,
        np.pi,
        5 / 4 * np.pi,
        3 / 2 * np.pi,
        7 / 4 * np.pi,
    ])
    ax.set_xticklabels(["0°", "45°", "90°", "", "", "", "", ""])

    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=xlabelpad, fontweight=fontweight)
    if all((val is not None for val in (ymin, ymax))):
        ax.set_ylim((ymin, ymax))
    plt.setp(ax.spines.values(), color="grey", linewidth=0.5)

    if legend:
        ax.legend(**legend_kwargs)

    return fig, ax


def loss_curves(
    losses: List[np.ndarray],
    smooth: float = 0.05,
    subsample: int = 1,
    mean: bool = False,
    grid: bool = True,
    colors: Optional[List[str]] = None,
    cbar: bool = False,
    cmap: Optional[mpl.colors.Colormap] = None,
    norm: Optional[mpl.colors.Normalize] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plot loss traces.

    Args:
        losses: List of loss arrays, each of shape (n_iters,).
        smooth: Smoothing factor for the loss curves.
        subsample: Subsample factor for the loss curves.
        mean: Whether to plot the mean loss curve.
        grid: Whether to show grid lines.
        colors: List of colors for the loss curves.
        cbar: Whether to add a colorbar.
        cmap: Colormap for the loss curves.
        norm: Normalization for the colormap.
        fig: Existing figure to plot on.
        ax: Existing axes to plot on.
        xlabel: Label for the x-axis.
        ylabel: Label for the y-axis.

    Returns:
        A tuple containing the Figure and Axes objects.

    Note:
        This function plots loss curves for multiple models, with options for
        smoothing, subsampling, and various visual customizations.
    """
    losses = np.array([loss[::subsample] for loss in losses])

    max_n_iters = max(len(loss) for loss in losses)

    _losses = np.full((len(losses), max_n_iters), np.nan)
    for i, loss in enumerate(losses):
        n_iters = len(loss)
        _losses[i, :n_iters] = loss

    fig, ax, _, _ = traces(
        _losses[::-1],
        x=np.arange(max_n_iters) * subsample,
        fontsize=5,
        figsize=[1.2, 1],
        smooth=smooth,
        fig=fig,
        ax=ax,
        color=colors[::-1] if colors is not None else None,
        linewidth=0.5,
        highlight_mean=mean,
    )

    ax.set_ylabel(ylabel, fontsize=5)
    ax.set_xlabel(xlabel, fontsize=5)

    if cbar and cmap is not None and norm is not None:
        plt_utils.add_colorbar_to_fig(
            fig,
            cmap=cmap,
            norm=norm,
            label="min task error",
            fontsize=5,
            tick_length=1,
            tick_width=0.5,
            x_offset=2,
            y_offset=0.25,
        )

    if grid:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.grid(True, linewidth=0.5)

    return fig, ax


def histogram(
    array: np.ndarray,
    bins: Optional[Union[int, Sequence, str]] = None,
    fill: bool = False,
    histtype: Literal["bar", "barstacked", "step", "stepfilled"] = "step",
    figsize: Tuple[float, float] = (1, 1),
    fontsize: int = 5,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """
    Create a histogram plot.

    Args:
        array: Input data to plot.
        bins: Number of bins or bin edges. Defaults to len(array).
        fill: Whether to fill the bars. Defaults to False.
        histtype: Type of histogram to plot. Defaults to "step".
        figsize: Size of the figure. Defaults to (1, 1).
        fontsize: Font size for labels. Defaults to 5.
        fig: Existing figure to plot on. Defaults to None.
        ax: Existing axes to plot on. Defaults to None.
        xlabel: Label for x-axis. Defaults to None.
        ylabel: Label for y-axis. Defaults to None.

    Returns:
        A tuple containing the Figure and Axes objects.
    """
    fig, ax = plt_utils.init_plot(figsize=figsize, fontsize=fontsize, fig=fig, ax=ax)
    ax.hist(
        array,
        bins=bins if bins is not None else len(array),
        linewidth=0.5,
        fill=fill,
        histtype=histtype,
    )
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    return fig, ax


def violins(
    variable_names: List[str],
    variable_values: np.ndarray,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    max_per_ax: Optional[int] = 20,
    colors: Optional[Union[str, List[str]]] = None,
    cmap: plt.cm = plt.cm.viridis_r,
    fontsize: int = 5,
    violin_width: float = 0.7,
    legend: Optional[Union[str, List[str]]] = None,
    scatter_extent: List[float] = [-0.35, 0.35],
    figwidth: float = 10,
    fig: Optional[Figure] = None,
    axes: Optional[List[Axes]] = None,
    ylabel_offset: float = 0.2,
    **kwargs: Any,
) -> Tuple[Figure, List[Axes]]:
    """
    Create violin plots for multiple variables across groups.

    Args:
        variable_names: Names of the variables to plot.
        variable_values: Array of values for each variable and group.
        ylabel: Label for the y-axis.
        title: Title of the plot.
        max_per_ax: Maximum number of variables per axis.
        colors: Colors for the violin plots.
        cmap: Colormap to use if colors are not specified.
        fontsize: Font size for labels and ticks.
        violin_width: Width of each violin plot.
        legend: Legend labels for groups.
        scatter_extent: Extent of scatter points on violins.
        figwidth: Width of the figure.
        fig: Existing figure to plot on.
        axes: Existing axes to plot on.
        ylabel_offset: Offset for y-axis label.
        **kwargs: Additional keyword arguments for violin_groups function.

    Returns:
        A tuple containing the Figure and list of Axes objects.

    Note:
        This function creates violin plots for multiple variables, potentially
        across multiple groups, with optional scatter points on each violin.
    """
    variable_values = variable_values.T
    if len(variable_values.shape) == 2:
        variable_values = variable_values[:, None]

    n_variables, n_groups, n_samples = variable_values.shape
    if max_per_ax is None:
        max_per_ax = n_variables
    max_per_ax = min(max_per_ax, n_variables)
    n_axes = int(n_variables / max_per_ax)
    max_per_ax += int(np.ceil((n_variables % max_per_ax) / n_axes))

    fig, axes, _ = plt_utils.get_axis_grid(
        gridheight=n_axes,
        gridwidth=1,
        figsize=[figwidth, n_axes * 1.2],
        hspace=1,
        alpha=0,
        fig=fig,
        axes=axes,
    )

    for i in range(n_axes):
        ax_values = variable_values[i * max_per_ax : (i + 1) * max_per_ax]
        ax_names = variable_names[i * max_per_ax : (i + 1) * max_per_ax]

        fig, ax, C = violin_groups(
            ax_values,
            ax_names,
            rotation=90,
            scatter=False,
            fontsize=fontsize,
            width=violin_width,
            scatter_edge_color="white",
            scatter_radius=5,
            scatter_edge_width=0.25,
            cdist=100,
            colors=colors,
            cmap=cmap,
            showmedians=True,
            showmeans=False,
            violin_marker_lw=0.25,
            legend=(legend if legend else None if i == 0 else None),
            legend_kwargs=dict(
                fontsize=5,
                markerscale=10,
                loc="lower left",
                bbox_to_anchor=(0.75, 0.75),
            ),
            fig=fig,
            ax=axes[i],
            **kwargs,
        )

        violin_locations, _ = get_violin_x_locations(
            n_groups, len(ax_names), violin_width
        )

        for group in range(n_groups):
            plt_utils.scatter_on_violins_or_bars(
                ax_values[:, group].T,
                ax,
                xticks=violin_locations[group],
                facecolor="none",
                edgecolor="k",
                zorder=100,
                alpha=0.35,
                uniform=scatter_extent,
                marker="o",
                linewidth=0.5,
            )

        ax.grid(False)

        plt_utils.trim_axis(ax, yaxis=False)
        plt_utils.set_spine_tick_params(
            ax,
            tickwidth=0.5,
            ticklength=3,
            ticklabelpad=2,
            spinewidth=0.5,
        )

    lefts, bottoms, rights, tops = np.array([ax.get_position().extents for ax in axes]).T
    fig.text(
        lefts.min() - ylabel_offset * lefts.min(),
        (tops.max() - bottoms.min()) / 2,
        ylabel,
        rotation=90,
        fontsize=fontsize,
        ha="right",
        va="center",
    )

    axes[0].set_title(title, y=0.91, fontsize=fontsize)

    return fig, axes


def plot_strf(
    time: np.ndarray,
    rf: np.ndarray,
    hlines: bool = True,
    vlines: bool = True,
    time_axis: bool = True,
    fontsize: int = 6,
    fig: Optional[Figure] = None,
    axes: Optional[np.ndarray] = None,
    figsize: List[float] = [5, 1],
    wspace: float = 0,
    y_offset_time_axis: float = 0,
) -> Tuple[Figure, np.ndarray]:
    """
    Plot a Spatio-Temporal Receptive Field (STRF).

    Args:
        time: Array of time points.
        rf: Receptive field array.
        hlines: Whether to draw horizontal lines. Defaults to True.
        vlines: Whether to draw vertical lines. Defaults to True.
        time_axis: Whether to add a time axis. Defaults to True.
        fontsize: Font size for labels and ticks.
        fig: Existing figure to plot on.
        axes: Existing axes to plot on.
        figsize: Size of the figure as [width, height].
        wspace: Width space between subplots.
        y_offset_time_axis: Vertical offset for the time axis.

    Returns:
        A tuple containing the Figure and Axes objects.

    Note:
        This function creates a series of hexagonal plots representing the STRF
        at different time points.
    """
    max_extent = hex_utils.get_hextent(rf.shape[-1])
    t_steps = np.arange(0.0, 0.2, 0.01)[::2]

    u, v = hex_utils.get_hex_coords(max_extent)
    x, y = hex_utils.hex_to_pixel(u, v)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    elev = 0
    azim = 0

    if fig is None or axes is None:
        fig, axes = plt_utils.divide_figure_to_grid(
            np.arange(10).reshape(1, 10),
            wspace=wspace,
            as_matrix=True,
            figsize=figsize,
        )

    crange = np.abs(rf).max()

    for i, t in enumerate(t_steps):
        mask = np.where(np.abs(time - t) <= 1e-15, True, False)
        _rf = rf[mask]
        quick_hex_scatter(
            _rf,
            cmap=plt.cm.coolwarm,
            edgecolor=None,
            vmin=-crange,
            vmax=crange,
            midpoint=0,
            cbar=False,
            max_extent=max_extent,
            fig=fig,
            ax=axes[0, i],
            fill=True,
            fontsize=fontsize,
        )

        if hlines:
            axes[0, i].hlines(elev, xmin, xmax, color="grey", linewidth=0.25)
        if vlines:
            axes[0, i].vlines(azim, ymin, ymax, color="grey", linewidth=0.25)

    if time_axis:
        left = fig.transFigure.inverted().transform(
            axes[0, 0].transData.transform((0, 0))
        )[0]
        right = fig.transFigure.inverted().transform(
            axes[0, -1].transData.transform((0, 0))
        )[0]

        lefts, bottoms, rights, tops = np.array([
            ax.get_position().extents for ax in axes.flatten()
        ]).T
        time_axis = fig.add_axes((
            left,
            bottoms.min() + y_offset_time_axis * bottoms.min(),
            right - left,
            0.01,
        ))
        plt_utils.rm_spines(
            time_axis,
            ("left", "top", "right"),
            rm_yticks=True,
            rm_xticks=False,
        )

        data_centers_in_points = np.array([
            ax.transData.transform((0, 0)) for ax in axes.flatten()
        ])
        time_axis.tick_params(axis="both", labelsize=fontsize)
        ticks = time_axis.transData.inverted().transform(data_centers_in_points)[:, 0]
        time_axis.set_xticks(ticks)
        time_axis.set_xticklabels(np.arange(0, 200, 20))
        time_axis.set_xlabel("time (ms)", fontsize=fontsize, labelpad=2)
        plt_utils.set_spine_tick_params(
            time_axis,
            spinewidth=0.25,
            tickwidth=0.25,
            ticklength=3,
            ticklabelpad=2,
            spines=("top", "right", "bottom", "left"),
        )

    return fig, axes
