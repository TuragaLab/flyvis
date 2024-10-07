import logging
from contextlib import suppress
from dataclasses import dataclass
from numbers import Number
from typing import Iterable, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colormaps as cm
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import RegularPolygon
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from numpy.typing import NDArray

from flyvision import utils
from flyvision.utils import hex_utils

from . import plt_utils

logging = logging.getLogger(__name__)


def heatmap(
    matrix,
    xlabels,
    ylabels=None,
    size_scale="auto",
    cmap=cm.get_cmap("seismic"),
    origin="upper",
    ax=None,
    fig=None,
    vmin=None,
    vmax=None,
    # cbar_fontsize=16,
    symlog=None,
    cbar_label="",
    log=None,
    cbar_height=0.5,
    cbar_width=0.01,
    title="",
    figsize=[5, 4],
    fontsize=4,
    midpoint=None,
    cbar=True,
    grid_linewidth=0.5,
    **kwargs,
) -> Tuple[Figure, Axis, Colorbar, NDArray]:
    """
    Heatmap scatter of the matrix.

    Args:
        matrix (np.ndarray): 2D matrix
        xlabels (list): list of x labels
        ylabels (list): list of y labels. Optional. If not provided, xlabels will be
            used.
        scale (bool): whether to scale the size of the scatter points with the value.
            If True, the sizes of the scatter points will be |value| * size_scale
            or |value| * 0.005 * (size_scale or prod(figsize)).
            If False, the sizes of the scatter points will be 100.
        size_scale (float): size scale of the scatter points. Optional.
            If not provided, prod(figsize) will be used.
        origin (str): origin of the matrix. Either "upper" or "lower".
        size_transform (callable): optional function to transform the values to the
            size of the scatter points.

    Returns:
        fig, ax, cbar, matrix
    """
    y, x = np.nonzero(matrix)
    value = matrix[y, x]

    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax=ax, fig=fig, offset=0)

    norm = plt_utils.get_norm(
        symlog=symlog,
        vmin=vmin if vmin is not None else np.nanmin(matrix),
        vmax=vmax if vmin is not None else np.nanmax(matrix),
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

    cbar = (
        plt_utils.add_colorbar_to_fig(
            fig,
            height=cbar_height,
            width=cbar_width,
            cmap=cmap,
            norm=norm,
            fontsize=fontsize,
            label=cbar_label,
            x_offset=15,
        )
        if cbar
        else None
    )

    return fig, ax, cbar, matrix


# ---- hex scatter


def hex_scatter(
    u,
    v,
    color,
    max_extent=None,
    fig=None,
    ax=None,
    figsize=(1, 1),
    title="",
    title_y=None,
    fontsize=5,
    label="",
    labelxy="auto",
    label_color="black",
    edgecolor=None,
    edgewidth=0.5,
    alpha=1,
    fill=False,
    scalarmapper=None,
    norm=None,
    radius=1,
    origin="lower",
    vmin=None,
    vmax=None,
    midpoint=None,
    mode="default",
    orientation=np.radians(30),
    cmap=cm.get_cmap("seismic"),
    cbar=True,
    cbar_label="",
    cbar_height=None,
    cbar_width=None,
    cbar_x_offset=0.05,
    annotate=False,
    annotate_coords=False,
    annotate_indices=False,
    frame=False,
    frame_hex_width=1,
    frame_color=None,
    nan_linestyle="-",
    text_color_hsv_threshold=0.8,
    **kwargs,
):
    """Plots a hexagonal lattice with coordinates u, v, and coloring color.

    Args:
        u (array): array of hex coordinates in u direction.
        v (array): array of hex coordinates in v direction.
        color (array): array of pixel values per point (u_i, v_i).
        max_extent (tuple, optional): maximum extent of the hex lattice.
            Defaults to None.
        label (str, optional): a label positioned on the axis. In the upper left
            corner per default. Defaults to "".
        labelxy (tuple, optional): position of the label. Defaults to (0, 1).
        edgecolor (str, optional): color of the edges. Defaults to None.
        alpha (float, optional): alpha of the hexagon faces. Defaults to 0.8.
        fill (bool, optional): automatic filling of smallest hex grid around
            u, v, c. Defaults to False.
        scalarmapper (Scalarmappable, optional): Defaults to None.
        norm (Norm, optional): Defaults to None.
        vmin (float, optional): Defaults to None.
        vmax (float, optional): Defaults to None.
        midpoint (float, optional): color midpoint. Defaults to None.
        cmap (Colormap, optional): Defaults to cm.get_cmap("seismic").
        cbar (bool, optional): plots a colorbar. Defaults to True.
        annotate (bool, optional): annotates a rounded value. Defaults to False.
        annotate_coords (bool, optional): annotates (u_i, v_i).
            Defaults to False.
        mode (str, optional): hex coordinate system. Defaults to "default".
        orientation (float, optional): orientation of the hexagons in rad.
            Defaults to np.radians(30).


    Returns:
        [type]: [description]
    """
    fig, ax = plt_utils.init_plot(
        figsize, title, fontsize, ax, fig, title_y=title_y, **kwargs
    )
    ax.set_aspect("equal")
    color = color * np.ones_like(u) if not isinstance(color, Iterable) else color
    if u.shape != v.shape or u.shape != color.shape:
        raise ValueError("shape mismatch of hexal values and coordinates")

    u, v, color = hex_utils.sort_u_then_v(u, v, color)
    if max_extent:
        extent = hex_utils.get_extent(u, v) or 1

        if fill and extent < max_extent:
            _u, _v = hex_utils.get_hex_coords(max_extent)
            _color = np.ones_like(_u, dtype=float) * np.nan
            UV = np.stack((_u, _v), axis=1)
            uv = np.stack((u, v), axis=1)
            mask = utils.tensor_utils.matrix_mask_by_sub(uv, UV)
            _color[mask] = color
            u, v, color = _u, _v, _color

        extent_condition = (
            (-max_extent <= u)
            & (u <= max_extent)
            & (-max_extent <= v)
            & (v <= max_extent)
            & (-max_extent <= u + v)
            & (u + v <= max_extent)
        )
        u = u[extent_condition]
        v = v[extent_condition]
        color = color[extent_condition]

    vmin = vmin - 1e-10 if vmin is not None else np.nanmin(color) - 1e-10
    vmax = vmax + 1e-10 if vmax is not None else np.nanmax(color) + 1e-10

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

    scalarmapper, norm = plt_utils.get_scalarmapper(
        scalarmapper=scalarmapper,
        cmap=cmap,
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        midpoint=midpoint,
    )
    color_rgba = scalarmapper.to_rgba(color)

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
        framed_color[mask] = color
        framed_color[~mask] = 0.0
        # c_mask = np.ma.masked_invalid(framed_color)
        # framed_color[c_mask.mask] = 0.0
        framed_color_rgba[mask] = color_rgba
        framed_color_rgba[~mask] = frame_color if frame_color else np.array([0, 0, 0, 1])
        u, v, color_rgba, color = _u, _v, framed_color_rgba, framed_color

    extent = hex_utils.get_extent(u, v) or 1
    c_mask = np.ma.masked_invalid(color)
    # color[c_mask.mask] = 0.0

    _valid_patches_start_index = 0
    if fill:
        if isinstance(fill, bool):
            # smallest hexagonal coordinate system around that
            u_cs, v_cs = hex_utils.get_hex_coords(extent)
            x_cs, y_cs = hex_utils.hex_to_pixel(u_cs, v_cs, mode=mode)
            # if hex_utils.get_hextent(len(color)) != extent:
            if origin == "upper":
                y_cs = y_cs[::-1]
            for _x, _y in zip(x_cs, y_cs):
                _hex = RegularPolygon(
                    (_x, _y),
                    numVertices=6,
                    radius=radius,
                    linewidth=edgewidth,
                    orientation=orientation,
                    edgecolor=edgecolor,
                    facecolor="white",
                    alpha=alpha,
                    ls="-",
                )
                # Adding fill to patches not allows to loop through actual patches.
                # We assign new colors to them in animation's update functions.
                # Requires tracking actual and filled hexals if neurons are strided
                # as used in LayerActivityGrid. Simply store start index for the actual
                # valid patches as attribute of the axis.
                ax.add_patch(_hex)
                _valid_patches_start_index += 1
        elif isinstance(fill, Number):
            fill = int(fill)
            extent = fill
            # smallest hexagonal coordinate system around that
            u_cs, v_cs = hex_utils.get_hex_coords(fill)
            x_cs, y_cs = hex_utils.hex_to_pixel(u_cs, v_cs, mode=mode)
            # if hex_utils.get_hextent(len(color)) != fill:
            if origin == "upper":
                y_cs = y_cs[::-1]
            for _x, _y in zip(x_cs, y_cs):
                _hex = RegularPolygon(
                    (_x, _y),
                    numVertices=6,
                    radius=radius,
                    linewidth=edgewidth,
                    orientation=orientation,
                    edgecolor=edgecolor,
                    facecolor="white",
                    alpha=alpha,
                    ls="-",
                )
                # Adding fill to patches not allows to loop through actual patches.
                # We assign new colors to them in animation's update functions.
                # Requires tracking actual and filled hexals if neurons are strided
                # as used in LayerActivityGrid. Simply store start index for the actual
                # valid patches as attribute of the axis.
                ax.add_patch(_hex)
                _valid_patches_start_index += 1
    ax._valid_patches_start_index = _valid_patches_start_index

    # Add coloured hexagons on top
    x, y = hex_utils.hex_to_pixel(u, v, mode=mode)
    if origin == "upper":
        y = y[::-1]
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

    if cbar:
        cbar = plt_utils.add_colorbar_to_fig(
            fig,
            label=cbar_label,
            width=cbar_width or 0.03,
            height=cbar_height or 0.5,
            x_offset=cbar_x_offset or -2,  # (1 - 1 / extent),
            cmap=cmap,
            norm=norm,
            fontsize=fontsize,
            tick_length=1,
            tick_width=0.25,
            rm_outline=True,
        )

    if fill:
        xmin, xmax = plt_utils.get_lims(x_cs, 1 / extent)
        ymin, ymax = plt_utils.get_lims(y_cs, 1 / extent)
        if xmin != xmax and ymin != ymax:
            ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax])
    else:
        xmin, xmax = plt_utils.get_lims(x, 1 / extent)
        ymin, ymax = plt_utils.get_lims(y, 1 / extent)
        if xmin != xmax and ymin != ymax:
            ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax])

    ax = plt_utils.rm_spines(ax, rm_xticks=True, rm_yticks=True)
    # cmap_fc = cm.get_cmap("binary")
    if annotate:
        # annotate hexagons with activity value
        for i, (_label, _x, _y) in enumerate(zip(color, x, y)):
            if not c_mask.mask[i] and not np.isnan(_label):
                # fontcolor = cmap_fc(_label)
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
            # elif c_mask.mask[i]:
            #     _textcolor = "black"
            #     ax.annotate(
            #         f"nan",
            #         fontsize=fontsize,
            #         xy=(_x, _y),
            #         xytext=(0, 0),
            #         textcoords="offset points",
            #         ha="center",
            #         va="center",
            #         color=_textcolor,
            #     )

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

    if labelxy == "auto":
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

    (xmin, ymin, xmax, ymax) = ax.dataLim.extents
    ax.set_xlim(plt_utils.get_lims((xmin, xmax), 0.01))
    ax.set_ylim(plt_utils.get_lims((ymin, ymax), 0.01))

    return fig, ax, (label_text, scalarmapper)


class SignError(Exception):
    pass


def kernel(
    u,
    v,
    color,
    fontsize=5,
    cbar=True,
    edgecolor="k",
    fig=None,
    ax=None,
    figsize=(1, 1),
    midpoint=0,
    annotate=True,
    alpha=0.8,
    annotate_coords=False,
    coord_fs=8,
    cbar_height=0.3,
    cbar_x_offset=-1,
    **kwargs,
):
    """Plotting a receptive fields with hex_scatter.

    Note, assigns `seismic` as colormap and checks that signs are consistent.

    Args:
        see hex_scatter, all args but cmap can be passed.
        cmap is `seismic`.

    Returns:
        see hex_scatter

    Raises:
        SignError: if signs are inconsistent
    """
    sign = set(np.sign(color[np.nonzero(color)]))
    if len(sign) == 1:
        sign = sign.pop()
    elif len(sign) == 0:
        sign = 1
    else:
        raise SignError(f"inconsistent kernel with signs {sign}")
    cmap = cm.get_cmap("seismic")
    _kwargs = locals()
    _kwargs.update(_kwargs["kwargs"])
    _kwargs.pop("kwargs")
    return hex_scatter(**_kwargs)


def hex_cs(extent=5, mode="default", annotate_coords=True, edgecolor="black", **kwargs):
    """Convenience function for plotting a hexagonal coordinate system."""
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


def quick_hex_scatter(color, cmap=cm.get_cmap("binary_r"), **kwargs):
    """Convenience function for a hex scatter plot with implicit coordinates."""
    color = utils.tensor_utils.to_numpy(color.squeeze())
    u, v = hex_utils.get_hex_coords(hex_utils.get_hextent(len(color)))
    return hex_scatter(u, v, color, cmap=cmap, **kwargs)


# ------------------------- hex optic flow -------------------------


def hex_flow(
    u,
    v,
    flow,
    fig=None,
    ax=None,
    figsize=(1, 1),
    title="",
    cmap=plt_utils.cm_uniform_2d,
    max_extent=None,
    marker="H",
    alpha=0.7,
    cwheelradius=0.25,
    mode="default",
    orientation=np.radians(30),
    origin="lower",
    markerscale=1,
    fontsize=5,
    cwheel=True,
    cwheelxy=(),
    cwheelpos="southeast",
    cwheellabelpad=-5,
    annotate_r=False,
    annotate_theta=False,
    annotate_coords=False,
    coord_fs=3,
    label="",
    labelxy=(0, 1),
    vmin=-np.pi,
    vmax=np.pi,
    edgecolor=None,
    **kwargs,
):
    """Plots a hexagonal lattice with coordinates u, v, and coloring color.

    Args:
        u (array): array of hex coordinates in u direction.
        v (array): array of hex coordinates in v direction.
        flow (array): array of flow per point (u_i, v_i), i.e. shape [2, len(u)].

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

    # Add some coloured hexagons
    x, y = hex_utils.hex_to_pixel(u, v, mode=mode)
    if origin == "upper":
        y = y[::-1]
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
        # annotate hexagons with magnitude and angle
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


def quick_hex_flow(flow, **kwargs):
    """Convenience function for plotting a flow field on a hexagonal lattice with
    implicit coordinates."""
    flow = utils.tensor_utils.to_numpy(flow.squeeze())
    u, v = hex_utils.get_hex_coords(hex_utils.get_hextent(flow.shape[-1]))
    return hex_flow(u, v, flow, **kwargs)


# --- cartesian flow plots ---


def flow_to_rgba(flow):
    """Map cartesian flow to rgba colors.

    Args:
        flow of shape (2, h, w)
    """
    if isinstance(flow, torch.Tensor):
        flow = flow.cpu().numpy()

    X, Y = flow[0], flow[1]
    R = np.sqrt(X * X + Y * Y)
    PHI = np.arctan2(Y, X)  # + np.pi
    scalarmapper, norm = plt_utils.get_scalarmapper(
        cmap=plt_utils.cm_uniform_2d, vmin=-np.pi, vmax=np.pi
    )
    rgba = scalarmapper.to_rgba(PHI)
    rgba[:, :, -1] = R / R.max()
    return rgba


def plot_flow(flow):
    """Plot cartesian flow.

    Args:
        flow of shape (2, h, w)
    """
    rgba = flow_to_rgba(flow)
    plt.imshow(rgba)
    plt.show()


# ---- TRACES


def traces(
    trace,
    x=None,
    contour=None,
    legend=(),
    smooth=None,
    stim_line=None,
    contour_cmap=cm.get_cmap("bone"),
    color=None,
    label="",
    labelxy=(0, 1),
    linewidth=1,
    ax=None,
    fig=None,
    title="",
    highlight_mean=False,
    figsize=(7, 4),
    fontsize=10,
    ylim=None,
    ylabel="",
    xlabel="",
    legend_frame_alpha=0,
    contour_mode="full",
    contour_y_rel=0.06,
    fancy=False,
    scale_pos=None,
    scale_label="100ms",
    null_line=False,
    zorder_traces=None,
    zorder_mean=None,
    **kwargs,
):
    """Simple line plot with optional contour e.g. to visualize stimuli and
    optional smoothing.

    Args:
        trace (array): 2D array (#traces, #points).
        x (array, optional): x-axis values. Defaults to None.
        contour (array, optional): (#points) of contour values.
        legend (list, optional): legend for each trace. Defaults to [].
        smooth (float, optional): size of smoothing window in percent of #points.
            Default is 0.05.

    Returns:
        fig, ax, trace (smoothed), label
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

    if highlight_mean is True:
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
            # print(contour_y_range)
            # print(ylim)
            ylim = (contour_y_range[0], ylim[1])
            # print(ylim)

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
    if null_line is True:
        ax.hlines(
            0,
            -20_000,
            20_000,
            color="0.5",  # "0.7",
            # linestyle="--",
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
    trace_groups: list[np.ndarray],
    x=None,
    legend=(),
    color=None,
    linewidth=1,
    ax=None,
    fig=None,
    title="",
    highlight_mean=False,
    figsize=(7, 4),
    fontsize=10,
    ylim=None,
    ylabel="",
    xlabel="",
    legend_frame_alpha=0,
    **kwargs,
):
    """Line plot with"""
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
            f"`traces.shape[0]`, or None. Got {color} of shape {np.shape(color)}."
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


def get_violin_x_locations(n_groups, n_random_variables, violin_width):
    violin_locations = np.zeros([n_groups, n_random_variables])
    # n_variable ticks are n_groups distance apart so that each violins width
    # is between 0 and 1 in x-space
    first_violins_location = np.arange(0, n_groups * n_random_variables, n_groups)
    for j in range(n_groups):
        # step by violin_width along x
        violin_locations[j] = first_violins_location + j * violin_width

    return violin_locations, first_violins_location


@dataclass
class ViolinData:
    # fig: Figure
    # ax: Axis
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
    color_by: str = "groups",
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

    def plot_bar(X, values, color):
        handle = ax.bar(x=X, width=width, height=np.mean(values), color=color, zorder=1)
        return handle

    def plot_violin(X, values, color):
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

    # Create matrix for x position for each bar.
    violin_locations, first_violins_location = get_violin_x_locations(
        n_groups, n_random_variables, violin_width=width
    )
    X = violin_locations.T

    if colors is None:
        # Create matrix of colors.
        if color_by == "groups":
            C = np.asarray([cmap(cstart + i * cdist) for i in range(n_groups)]).reshape(
                n_groups, 4
            )
        if color_by == "experiments":
            C = np.asarray([
                cmap(cstart + i * cdist) for i in range(n_random_variables)
            ]).reshape(n_random_variables, 4)
    elif isinstance(colors, Iterable):
        if color_by == "groups":
            if len(colors) == n_groups:
                C = colors
            else:
                raise ValueError
        if color_by == "experiments":
            if len(colors) == n_random_variables:
                C = colors
            else:
                raise ValueError
    else:
        raise ValueError

    # Plot each violin or bar and optionally scatter.
    handles = []

    for i in range(n_random_variables):
        for j in range(n_groups):
            if color_by == "experiments":
                _color = C[i]
            elif color_by == "groups":
                _color = C[j]
            else:
                raise ValueError

            if as_bars:
                h = plot_bar(X[i, j], values[i, j], _color)
            else:
                h = plot_violin(X[i, j], values[i, j], _color)
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


def plot_complex(z, marker="s", fig=None, ax=None, figsize=[1, 1], fontsize=5):
    fig, ax = plt_utils.init_plot(
        figsize=figsize, projection="polar", fontsize=fontsize, fig=fig, ax=ax
    )

    theta = np.angle(z)
    r = np.abs(z)

    ax.plot([0, theta], [0, r], marker=marker)
    return fig, ax


def plot_complex_vector(
    z0, z1, marker="s", fig=None, ax=None, figsize=[1, 1], fontsize=5
):
    fig, ax = plt_utils.init_plot(
        figsize=figsize, projection="polar", fontsize=fontsize, fig=fig, ax=ax
    )

    theta0 = np.angle(z0)
    r0 = np.abs(z0)

    theta = np.angle(z1)
    r = np.abs(z1)

    ax.plot([theta0, theta], [r0, r], marker=marker)
    return fig, ax


def extend_arg(arg, argtype, r, default, dim=-1):
    """Extend an argument to the correct length for a given dimension."""
    r = np.asarray(r)

    if isinstance(arg, argtype) and r.ndim > 1:
        # Extend the arg to a list of r.shape[dim] times the same value
        return [arg] * r.shape[dim]
    elif isinstance(arg, Iterable) and len(arg) == r.shape[dim]:
        # If it's already a list of the correct length, return it unchanged
        return arg
    elif r.ndim == 1 and np.asarray(arg).size == 1:
        return arg
    elif r.ndim == 1:
        return default
    else:
        raise ValueError(
            f"arg must be either an integer or a list of length {r.shape[-1]}."
        )


def polar(
    theta,
    r,
    ax=None,
    fig=None,
    color="b",
    linestyle="-",
    marker="",
    markersize=None,
    label=None,
    title="",
    figsize=(5, 5),
    fontsize=10,
    xlabel="",
    fontweight="normal",
    anglepad=-2,
    xlabelpad=-3,
    linewidth=2,
    ymin=None,
    ymax=None,
    stroke_kwargs={},
    yticks_off=True,
    zorder=100,
    **kwargs,
):
    """Polar tuning plot.

    Args:
        theta (array): angles or x in degree!
        r (array): radius or y.

    Returns:
        [type]: [description]
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
        logging.warning("using radians instead of degree?")

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

    zorder = extend_arg(zorder, int, r, default=0, dim=-1)

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


def half_polar(theta, r, **kwargs):
    """To plot orientation tuning from oriented bars."""

    theta = np.append(theta, theta + 180)
    r = np.append(r, r)

    return polar(theta, r, **kwargs)


def multi_polar(
    theta,
    r,
    ax=None,
    fig=None,
    mean_color="b",
    norm=True,
    std=False,
    color="b",
    mean=False,
    linestyle="-",
    marker="",
    label="",
    legend=False,
    title="",
    figsize=(0.98, 2.38),
    fontsize=5,
    xlabel="",
    fontweight="bold",
    alpha=1,
    anglepad=-6,
    xlabelpad=-3,
    linewidth=0.75,
    ymin=None,
    ymax=None,
    zorder=None,
    legend_kwargs=dict(fontsize=5),
    rm_yticks=True,
    **kwargs,
):
    """Polar tuning plot.

    Args:
        theta (array): angles or x.
        r (array): radius or y, (nsamples, values).

    Returns:
        [type]: [description]
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

    if not isinstance(color, (list, np.ndarray)):
        color = n_traces * (color,)

    if not isinstance(label, (list, np.ndarray)):
        label = n_traces * (label,)

    if not isinstance(zorder, (list, np.ndarray)):
        zorder = n_traces * (100,)

    # why?
    if n_traces >= 1:
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


def half_multi_polar(
    theta,
    r,
    ax=None,
    fig=None,
    mean_color="b",
    norm=True,
    std=False,
    color="b",
    mean=False,
    linestyle="-",
    marker="",
    label="",
    legend=False,
    title="",
    figsize=(0.98, 2.38),
    fontsize=5,
    xlabel="",
    fontweight="bold",
    alpha=1,
    anglepad=-6,
    xlabelpad=-3,
    linewidth=0.75,
    ymin=None,
    ymax=None,
    zorder=None,
    legend_kwargs=dict(fontsize=5),
    rm_yticks=True,
    **kwargs,
):
    """Polar tuning plot.

    Args:
        theta (array): angles or x.
        r (array): radius or y, (nsamples, values).

    Returns:
        [type]: [description]
    """
    fig, ax = plt_utils.init_plot(
        figsize=figsize,
        title=title,
        fontsize=fontsize,
        ax=ax,
        fig=fig,
        projection="polar",
    )

    theta = np.append(theta, theta + 180)
    r = np.atleast_2d(r)
    r = np.append(r, r, axis=1)

    n_traces = r.shape[0]

    if norm:
        r = r / (r.max(axis=1, keepdims=True) + 1e-15)

    closed = theta[-1] % 360 == theta[0]
    theta = theta * np.pi / 180
    if not closed:
        theta = np.append(theta, theta[0])
        r = np.append(r, np.expand_dims(r[:, 0], 1), axis=1)

    if not isinstance(color, (list, np.ndarray)):
        color = n_traces * (color,)

    if not isinstance(label, (list, np.ndarray)):
        label = n_traces * (label,)

    if not isinstance(zorder, (list, np.ndarray)):
        zorder = n_traces * (100,)

    # why?
    if n_traces >= 1:
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
    plt.setp(ax.spines.values(), color="grey", linewidth=1)

    if legend:
        ax.legend(**legend_kwargs)

    return fig, ax


def stim_trace(time, stim, linewidth=1):
    fig, ax = plt.subplots(figsize=[2, 0.1])
    ax.plot(time, stim, "k", linewidth=linewidth)
    plt_utils.rm_spines(ax)
    return fig, ax


def loss_curves(
    losses,
    smooth=0.05,
    subsample=1,
    mean=False,
    grid=True,
    colors=None,
    cbar=False,
    cmap=None,
    norm=None,
    fig=None,
    ax=None,
    xlabel=None,
    ylabel=None,
):
    """Plot loss traces.

    Args:
        losses: tensor of shape (n_models, n_iters)
        smooth: smoothing factor
        subsample: subsample factor
        mean: plot mean
        grid: show grid
        colors: list of colors
        cbar: add colorbar
        cmap: colormap
        norm: normalization
        fig: figure
        ax: axis
    """
    losses = np.array([loss[::subsample] for loss in losses])

    max_n_iters = max([len(loss) for loss in losses])

    _losses = np.zeros([len(losses), max_n_iters]) * np.nan
    for i, loss in enumerate(losses):
        n_iters = len(loss)
        _losses[i, :n_iters] = loss[:]
    fig, ax, _, _ = traces(
        _losses[::-1],
        x=np.arange(max_n_iters) * subsample,
        fontsize=5,
        figsize=[1.2, 1],
        smooth=smooth,
        fig=fig,
        ax=ax,
        color=colors[::-1],
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
    array,
    bins=None,
    fill=False,
    histtype="step",
    figsize=[1, 1],
    fontsize=5,
    fig=None,
    ax=None,
    xlabel=None,
    ylabel=None,
):
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
    variable_names,
    variable_values,
    ylabel=None,
    title=None,
    max_per_ax=20,
    colors=None,
    cmap=plt.cm.viridis_r,
    fontsize=5,
    violin_width=0.7,
    legend=None,
    scatter_extent=[-0.35, 0.35],
    figwidth=10,
    fig=None,
    axes=None,
    ylabel_offset=0.2,
    **kwargs,
):
    """ """

    # variable first, samples second
    variable_values = variable_values.T
    if len(variable_values.shape) == 2:
        # add empty group dimension
        variable_values = variable_values[:, None]

    n_variables, n_groups, n_samples = variable_values.shape
    if max_per_ax is None:
        max_per_ax = n_variables
    max_per_ax = min(max_per_ax, n_variables)
    n_axes = int(n_variables / max_per_ax)
    max_per_ax += int(np.ceil((n_variables % max_per_ax) / n_axes))

    # breakpoint()
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

    # since axes are split, we need to manually add the ylabel
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

    # top axis gets the title
    axes[0].set_title(title, y=0.91, fontsize=fontsize)

    return fig, axes


def plot_strf(
    time,
    rf,
    hlines=True,
    vlines=True,
    time_axis=True,
    fontsize=6,
    fig=None,
    axes=None,
    figsize=[5, 1],
    wspace=0,
    y_offset_time_axis=0,
):
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
