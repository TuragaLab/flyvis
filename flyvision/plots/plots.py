from typing import Iterable, Tuple
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from numbers import Number
import logging
import torch

from matplotlib import colormaps as cm
from matplotlib.patches import RegularPolygon
import matplotlib as mpl
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.figure import Figure
from matplotlib.axis import Axis
from matplotlib.colorbar import Colorbar

from flyvision import utils
from flyvision.plots import plt_utils

logging = logging.getLogger()

# ---- connectivity matrix


def heatmap(
    matrix,
    xlabels,
    ylabels=None,
    scale=True,
    size_scale=None,
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
    xlabel="Postsynaptic",
    ylabel="Presynaptic",
    cbar_height=0.5,
    cbar_width=0.01,
    title="",
    figsize=[5, 4],
    fontsize=4,
    midpoint=None,
    cbar=True,
    grid_linewidth=0.5,
    size_transform=None,
    **kwargs,
) -> Tuple[Figure, Axis, Colorbar, NDArray]:
    """
    Heatmap scatter of the matrix.

    Args:
        matrix (np.ndarray): 2D matrix
        xlabels (list): list of x labels
        ylabels (list): list of y labels. Optional. If not provided, xlabels will be used.
        scale (bool): whether to scale the size of the scatter points with the value.
            If True, the sizes of the scatter points will be |value| * size_scale
            or |value| * 0.005 * (size_scale or prod(figsize)).
            If False, the sizes of the scatter points will be 100.
        size_scale (float): size scale of the scatter points. Optional.
            If not provided, prod(figsize) will be used.
        origin (str): origin of the matrix. Either "upper" or "lower".
        size_transform (callable): optional function to transform the values to the size of the scatter points.

    Returns:
        fig, ax, cbar, matrix
    """

    def x_y_value(matrix):
        x = np.arange(matrix.shape[1])
        y = np.arange(matrix.shape[0])
        table = []
        for _x in x:
            for i, _y in enumerate(y):
                val = matrix[_y, _x]
                if val:
                    table.append((_x, _y, val))
        return np.array(table).T

    x, y, value = x_y_value(matrix)

    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax=ax, fig=fig, offset=0)

    norm = plt_utils.get_norm(
        symlog=symlog,
        vmin=vmin if vmin is not None else np.nanmin(matrix),
        vmax=vmax if vmin is not None else np.nanmax(matrix),
        log=log,
        midpoint=midpoint,
    )

    # derive the sizes of the scatter points
    if size_transform is None:

        def default_size_transform(value):
            if scale:
                return np.abs(value) * (size_scale or 0.005 * np.prod(figsize))
            return 100

        size_transform = default_size_transform

    ax.scatter(
        x=x,
        y=matrix.shape[0] - y - 1 if origin == "upper" else y,
        s=size_transform(value),
        c=value,
        cmap=cmap,
        norm=norm,
        marker="s",
        edgecolors="none",
    )

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(xlabels, rotation=90, fontsize=fontsize)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ylabels = ylabels or xlabels
    ax.set_yticklabels(
        ylabels[::-1] if origin == "upper" else ylabels, fontsize=fontsize
    )

    ax.grid(False, "major")
    ax.grid(True, "minor", linewidth=grid_linewidth)
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()[:-1]], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()[:-1]], minor=True)

    ax.set_xlim([-0.5, max(x) + 0.5])
    ax.set_ylim([-0.5, max(y) + 0.5])
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
    labelxy=(0, 1),
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

    u, v, color = utils.hex_utils.sort_u_then_v(u, v, color)
    if max_extent:
        extent = utils.hex_utils.get_extent(u, v) or 1

        if fill and extent < max_extent:
            _u, _v = utils.hex_utils.get_hex_coords(max_extent)
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
        extent = utils.hex_utils.get_extent(u, v) or 1
        _u, _v = utils.hex_utils.get_hex_coords(extent + frame_hex_width)
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
        framed_color_rgba[~mask] = (
            frame_color if frame_color else np.array([0, 0, 0, 1])
        )
        u, v, color_rgba, color = _u, _v, framed_color_rgba, framed_color

    extent = utils.hex_utils.get_extent(u, v) or 1
    c_mask = np.ma.masked_invalid(color)
    # color[c_mask.mask] = 0.0

    _valid_patches_start_index = 0
    if fill:
        if isinstance(fill, bool):
            # smallest hexagonal coordinate system around that
            u_cs, v_cs = utils.hex_utils.get_hex_coords(extent)
            x_cs, y_cs = utils.hex_utils.hex_to_pixel(u_cs, v_cs, mode=mode)
            # if utils.get_hextent(len(color)) != extent:
            if origin == "upper":
                y_cs = y_cs[::-1]
            for i, (_x, _y) in enumerate(zip(x_cs, y_cs)):
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
                # Adding the fill to the patches not allows to simply loop through the actual patches
                # and assigning new colors to them in the animation's update functions if neurons are strided
                # as used in LayerActivityGrid. Therefore, store the start index for the actually valid patches as
                # attribute of the axis.
                ax.add_patch(_hex)
                _valid_patches_start_index += 1
        elif isinstance(fill, Number):
            fill = int(fill)
            extent = fill
            # smallest hexagonal coordinate system around that
            u_cs, v_cs = utils.hex_utils.get_hex_coords(fill)
            x_cs, y_cs = utils.hex_utils.hex_to_pixel(u_cs, v_cs, mode=mode)
            # if utils.get_hextent(len(color)) != fill:
            if origin == "upper":
                y_cs = y_cs[::-1]
            for i, (_x, _y) in enumerate(zip(x_cs, y_cs)):
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
                # Adding the fill to the patches not allows to simply loop through the actual patches
                # and assigning new colors to them in the animation's update functions if neurons are strided
                # as used in LayerActivityGrid. Therefore, store the start index for the actually valid patches as
                # attribute of the axis.
                ax.add_patch(_hex)
                _valid_patches_start_index += 1
    ax._valid_patches_start_index = _valid_patches_start_index

    # Add coloured hexagons on top
    x, y = utils.hex_utils.hex_to_pixel(u, v, mode=mode)
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

    # left, bottom, right, top = ax.get_position().extents

    label_text = ax.text(
        labelxy[0],
        labelxy[1],
        label,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=fontsize,
        zorder=100,
    )

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
    u, v = utils.get_hex_coords(extent)
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
    u, v = utils.hex_utils.get_hex_coords(utils.hex_utils.get_hextent(len(color)))
    return hex_scatter(u, v, color, cmap=cmap, **kwargs)


# ------------------------- hex optic flow -------------------------


def hex_flow(
    u,
    v,
    flow,
    fig=None,
    ax=None,
    figsize=(7, 7),
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
    fontsize=10,
    cwheel=True,
    cwheelxy=(),
    cwheelpos="southeast",
    cwheellabelpad=-5,
    annotate_r=False,
    annotate_theta=False,
    annotate_coords=False,
    coord_fs=8,
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
        max_extent_index = utils.max_extent_index(u, v, max_extent=max_extent)
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
    x, y = utils.hex_utils.hex_to_pixel(u, v, mode=mode)
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

    extent = utils.hex_utils.get_extent(u, v)
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
    flow = utils.to_numpy(flow.squeeze())
    u, v = utils.get_hex_coords(utils.get_hextent(flow.shape[-1]))
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
            np.array(
                [
                    min(contour.min(), trace.min()),
                    max(contour.max(), trace.max()),
                ]
            ),
            0.1,
        )

        if x is None or len(x) != len(contour):
            _x = np.arange(len(contour))
        else:
            _x = x
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

    if scale_pos:
        if not any([isinstance(a, AnchoredSizeBar) for a in ax.artists]):
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
