from typing import Iterable
from numbers import Number
from itertools import product
import colorsys
import torch
import torch.nn.functional as F

import numpy as np

# import seaborn as sns
# sns.set(font="Arial")
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, SymLogNorm, TwoSlopeNorm
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.transforms import Bbox
from matplotlib import cm

# import dvs.utils
# from dvs.plots.decoration import *
# from dvs.utils.color_utils import get_alpha_colormap


def default_save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0, transparent=True, format="pdf")


def rebuild_font_manager():
    doc = """Sometimes matplotlib does not stick to arial because it falls back to DejaVuSans.

    Workaround:
    in the file built by the fontmanager,
    the fallback option can be changed manually to Arial.
    Or this could be coded up here. But once we changed that file manually, we actually don't want
    to rebuild the font manager.
    """
    print(doc)


#     plt.style.use('~/.config/matplotlib/matplotlibrc')
#     matplotlib.font_manager._rebuild()
#     plt.style.use('~/.config/matplotlib/matplotlibrc')


def cm_to_inch(*args):
    if len(args) == 1:
        width = args[0][0]
        height = args[0][1]
    elif len(args) == 2:
        width = args[0]
        height = args[1]
    return width / 2.54, height / 2.54


class Arrow3D(FancyArrowPatch):
    def __init__(self, p0, p1, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = p0, p1

    def draw(self, renderer):
        p0, p1 = self._verts3d
        xs3d, ys3d, zs3d = (p0[0], p1[0]), (p0[1], p1[1]), (p0[2], p1[2])
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        p0, p1 = self._verts3d
        xs3d, ys3d, zs3d = (p0[0], p1[0]), (p0[1], p1[1]), (p0[2], p1[2])
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def figure(
    figsize,
    hspace=0.3,
    wspace=0.1,
    left=0.125,
    right=0.9,
    top=0.9,
    bottom=0.1,
    frameon=None,
):
    fig = plt.figure(figsize=figsize, frameon=frameon)
    plt.subplots_adjust(
        hspace=hspace,
        wspace=wspace,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
    )
    return fig


def subplot(
    title,
    grid=(1, 1),
    location=(0, 0),
    colspan=1,
    rowspan=1,
    projection=None,
    sharex=None,
    sharey=None,
    xlabel="",
    ylabel="",
    despine=True,
    face_alpha=1.0,
    offset=5,
    trim=False,
    title_fs=15,
    title_pos="center",
    position=None,
    **kwargs,
):
    ax = plt.subplot2grid(
        grid,
        location,
        colspan,
        rowspan,
        sharex=sharex,
        sharey=sharey,
        projection=projection,
    )
    if position:
        ax.set_position(position)
    ax.patch.set_alpha(face_alpha)

    ax.set_title(title, fontsize=title_fs, loc=title_pos)

    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)

    # sns.set(context='paper', style='ticks', font_scale=1.5)
    # sns.axes_style({'axes.edgecolor': '.6', 'axes.linewidth': 5.0})
    # if despine is True and not projection:
    #     sns.despine(ax=ax, offset=offset, trim=trim)

    return ax


class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        # TwoSlopeNorm
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0.0, 0.5, 1.0]
        return np.ma.masked_array(np.interp(value, x, y))


def init_plot(
    figsize=[1, 1],
    title="",
    fontsize=10,
    ax=None,
    fig=None,
    projection=None,
    set_axis_off=False,
    transparent=False,
    face_alpha=0,
    position=None,
    title_pos="center",
    title_y=None,
    **kwargs,
):
    """Creates or returns existing fig and ax object with title being set.

    Args:
        figsize (list, optional): width and height of the figure.
            Defaults to [10, 10].
        title (str, optional): axis title. Defaults to ''.
        fontsize (int, optional): Defaults to 10.
        ax (Axes, optional): Defaults to None.
        fig (Figure, optional): Defaults to None.
        projection (str, optional): e.g. polar. Defaults to None.

    Returns:
        fig, ax
    """
    # mpl.rc("font", **{"family": font_family, "sans-serif": font_type})
    # mpl.rc("figure", dpi=dpi)

    # initialize figure and ax
    if fig is None:
        fig = plt.figure(figsize=figsize)
        # fig = figure(figsize)
    if ax is not None:
        ax.set_title(title, fontsize=fontsize, loc=title_pos, y=title_y)
    else:
        ax = fig.add_subplot(projection=projection)
        if position is not None:
            ax.set_position(position)
        ax.patch.set_alpha(face_alpha)
        ax.set_title(title, fontsize=fontsize, loc=title_pos, y=title_y)

        # ax = subplot(
        #     title, title_fs=fontsize, projection=projection, **kwargs
        # )
        if set_axis_off:
            ax.set_axis_off()
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    if transparent:
        ax.patch.set_alpha(0)
    return fig, ax


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def get_colors(num_colors):
    rnd = np.random.RandomState(7)
    colors = []
    k = 0
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue = i / 360.0
        if k == 0:
            lightness = (80 + rnd.rand() * 10) / 100.0
            saturation = (50 + rnd.rand() * 10) / 100.0
            k += 1
        elif k == 1:
            lightness = (50 + rnd.rand() * 10) / 100.0
            saturation = (80 + rnd.rand() * 10) / 100.0
            k += 1
        elif k == 2:
            lightness = (50 + rnd.rand() * 10) / 100.0
            saturation = (50 + rnd.rand() * 10) / 100.0
            k += 1
        elif k == 3:
            lightness = (80 + rnd.rand() * 10) / 100.0
            saturation = (80 + rnd.rand() * 10) / 100.0
            k = 0
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def rm_spines(
    ax,
    spines=("top", "right", "bottom", "left"),
    visible=False,
    rm_xticks=True,
    rm_yticks=True,
):
    for spine in spines:
        ax.spines[spine].set_visible(visible)
    if ("top" in spines or "bottom" in spines) and rm_xticks:
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position("none")
    if ("left" in spines or "right" in spines) and rm_yticks:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks_position("none")
    return ax


def standalone_legend(
    labels,
    colors,
    legend_elements=None,
    alpha=1,
    fontsize=6,
    fig=None,
    ax=None,
    lw=4,
    labelspacing=0.5,
    handlelength=2.0,
):

    if legend_elements is None:
        from matplotlib.lines import Line2D

        legend_elements = []
        for i, label in enumerate(labels):
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=colors[i],
                    lw=lw,
                    label=label,
                    alpha=alpha,
                    solid_capstyle="round",
                )
            )
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=[0.1, 0.1 * len(labels)])
    ax.legend(
        handles=legend_elements,
        loc="center",
        edgecolor="white",
        framealpha=1,
        fontsize=fontsize,
        bbox_to_anchor=(0, 1, 0, 1),
        labelspacing=labelspacing,
        handlelength=handlelength,
    )
    rm_spines(ax, rm_yticks=True, rm_xticks=True)
    return fig, ax


def add_legend(
    ax,
    labels,
    colors,
    legend_elements=None,
    alpha=1,
    fontsize=6,
    lw=4,
    labelspacing=0.5,
    handlelength=2.0,
    bbox_to_anchor=(1.1, 0.5),
    edgecolor=None,
    edgewidth=None,
    loc="center",
    override_alpha=False,
):

    if legend_elements is None:
        from matplotlib.lines import Line2D

        legend_elements = []
        for i, label in enumerate(labels):
            if len(colors[i]) == 4:
                if not override_alpha:
                    alpha = None
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=colors[i],
                    lw=lw,
                    label=label,
                    alpha=alpha,
                    solid_capstyle="round",
                    markeredgecolor=edgecolor,
                    markeredgewidth=edgewidth,
                )
            )
    ax.legend(
        handles=legend_elements,
        loc=loc,
        edgecolor="white",
        framealpha=1,
        fontsize=fontsize,
        bbox_to_anchor=bbox_to_anchor,
        labelspacing=labelspacing,
        handlelength=handlelength,
    )


def get_ax_centers(axes):
    return get_ax_positions(axes)[1][0]


def get_ax_lbwh(axes):
    (left, bottom, _, _), (_, width, height) = get_ax_positions(axes)
    return left, bottom, width, height


def get_ax_x0y0x1y1(axes):
    return get_ax_positions(axes)[0]


def get_ax_positions(axes):
    """
    Args:
        single ax or iterable of axes

    Returns:
        tuple ((lefts, bottoms, rights, tops), (centers, widths, heights))
    """
    axes = np.atleast_1d(axes)
    lefts, bottoms, rights, tops = np.atleast_2d(
        np.array([ax.get_position().extents for ax in axes])
    ).T
    widths = rights - lefts
    heights = tops - bottoms
    centers = np.array([lefts + widths / 2, bottoms + heights / 2])
    return (lefts, bottoms, rights, tops), (centers, widths, heights)


def standalone_marker_legend(
    labels,
    markers,
    colors="#6CA2CA",
    alpha=1,
    fontsize=6,
    markersize=10,
    fig=None,
    ax=None,
):
    from matplotlib.lines import Line2D

    legend_elements = []

    if isinstance(colors, str):
        colors = (colors,) * len(labels)

    for i, label in enumerate(labels):
        legend_elements.append(
            Line2D(
                [],
                [],
                color=colors[i],
                marker=markers[i],
                linestyle="None",
                label=label,
                alpha=alpha,
                markersize=markersize,
            )
        )
    if fig is None:
        fig, ax = plt.subplots(figsize=[0.1, 0.1 * len(labels)])
        bbox_to_anchor = (0, 1, 0, 1)
    else:
        if ax is None:
            ax = fig.add_axes([0, 0, 1, 1])
        left, bottom, right, top = ax.get_position().extents
        bbox_to_anchor = (left, right - left, bottom, top - bottom)

    ax.legend(
        handles=legend_elements,
        loc="center",
        edgecolor="white",
        framealpha=1,
        fontsize=fontsize,
        bbox_to_anchor=bbox_to_anchor,
    )
    rm_spines(ax, rm_yticks=True, rm_xticks=True)
    return fig, ax


def add_marker_legend(
    fig,
    labels,
    markers,
    colors="#1A599D",
    pos="right",
    width=0.05,
    height=0.5,
    x_offset=0,
    y_offset=0,
    alpha=1,
    fontsize=5,
    markersize=5,
):

    from matplotlib.lines import Line2D

    if isinstance(colors, str):
        colors = (colors,) * len(labels)

    legend_elements = [
        Line2D(
            [],
            [],
            color=colors[i],
            marker=markers[i],
            linestyle="None",
            label=label,
            alpha=alpha,
            markersize=markersize,
        )
        for i, label in enumerate(labels)
    ]

    position = derive_position_for_supplementary_ax(
        fig,
        pos=pos,
        width=width,
        height=height,
        x_offset=x_offset,
        y_offset=y_offset,
    )
    # print(position)
    legend_ax = fig.add_axes(position, label="marker legend", alpha=0)
    legend_ax.legend(
        handles=legend_elements,
        loc="center",
        edgecolor="white",
        framealpha=0,
        fontsize=fontsize,
        # bbox_to_anchor=(0, 1, 0, 1),
    )
    rm_spines(legend_ax, rm_yticks=True, rm_xticks=True)
    return legend_ax


def trim_axis(ax, xaxis=True, yaxis=True):

    if xaxis:
        xticks = np.array(ax.get_xticks())
        minor_xticks = np.array(ax.get_xticks(minor=True))
        all_ticks = np.sort(np.concatenate((minor_xticks, xticks)))
        if hasattr(xticks, "size"):
            firsttick = np.compress(all_ticks >= min(ax.get_xlim()), all_ticks)[0]
            lasttick = np.compress(all_ticks <= max(ax.get_xlim()), all_ticks)[-1]
            ax.spines["top"].set_bounds(firsttick, lasttick)
            ax.spines["bottom"].set_bounds(firsttick, lasttick)
            new_minor_ticks = minor_xticks.compress(minor_xticks <= lasttick)
            new_minor_ticks = new_minor_ticks.compress(new_minor_ticks >= firsttick)
            newticks = xticks.compress(xticks <= lasttick)
            newticks = newticks.compress(newticks >= firsttick)
            ax.set_xticks(newticks)
            ax.set_xticks(new_minor_ticks, minor=True)

    if yaxis:
        yticks = np.array(ax.get_yticks())
        minor_yticks = np.array(ax.get_yticks(minor=True))
        all_ticks = np.sort(np.concatenate((minor_yticks, yticks)))
        if hasattr(yticks, "size"):
            firsttick = np.compress(all_ticks >= min(ax.get_ylim()), all_ticks)[0]
            lasttick = np.compress(all_ticks <= max(ax.get_ylim()), all_ticks)[-1]
            ax.spines["left"].set_bounds(firsttick, lasttick)
            ax.spines["right"].set_bounds(firsttick, lasttick)
            new_minor_ticks = minor_yticks.compress(minor_yticks <= lasttick)
            new_minor_ticks = new_minor_ticks.compress(new_minor_ticks >= firsttick)
            newticks = yticks.compress(yticks <= lasttick)
            newticks = newticks.compress(newticks >= firsttick)
            ax.set_yticks(newticks)
            ax.set_yticks(new_minor_ticks, minor=True)


def get_polar_colormap(
    fig, ax, size=0.2, xy=(), cmap=plt.cm.twilight_shifted, fontsize=10
):
    """Plots a new axis with polar colormap next to a figure.

    Note: can only do 1d colorwheel. Seadd.
    """
    x0, y0, x1, y1 = ax.get_position().extents
    left = xy[0] if xy else x1 + 0.1 * x1
    bottom = xy[1] if xy else y1 - y1 / 2
    cb = fig.add_axes([left, bottom, size, size], polar=True)
    norm = plt.Normalize(0, 2 * np.pi)
    n = 200  # the number of secants for the mesh
    t = np.linspace(0, 2 * np.pi, n)  # theta values
    r = np.linspace(0, 1, 2)  # radius values change 0.6 to 0 for full circle
    rg, tg = np.meshgrid(r, t)  # create a r,theta meshgrid
    # plot the colormesh on axis with colormap
    im = cb.pcolormesh(t, r, tg.T, cmap=plt.cm.twilight_shifted, norm=norm)
    cb.set_yticklabels([])  # turn of radial tick labels (yticks)
    cb.tick_params(labelsize=fontsize)  # cosmetic changes to tick labels
    cb.spines["polar"].set_visible(False)
    return cb


def complex_to_rgb(complex_data, invert=False):
    """Computes RGB data from a complex array based on sinusoidal phase encoding."""
    phase = np.angle(complex_data)
    amplitude = np.abs(complex_data)
    amplitude = amplitude / amplitude.max()
    A = np.zeros([complex_data.shape[0], complex_data.shape[1], 4])
    A[:, :, 0] = 0.5 * (np.cos(phase) + 1) * amplitude
    A[:, :, 1] = 0.5 * (-np.cos(phase) + 1) * amplitude
    A[:, :, 2] = 0.8  # *(np.cos(phase)+1)*amplitude
    if invert:
        A = 1 - A
        A[:, :, -1] = amplitude
        return A
    else:
        A[:, :, -1] = amplitude
        return A


def polar_to_cmap(
    r, theta, invert=True, cmap=plt.cm.twilight_shifted, norm=None, sm=None
):
    """Computes RGB data from a complex array based on phase encoding by
    indicated colormap or scalarmappable.
    """
    sm = sm if sm else plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    r = r / r.max()
    A = np.zeros([theta.shape[0], theta.shape[1], 4])
    RGBA = sm.to_rgba(theta)
    A[:, :, 0] = RGBA[:, :, 0]
    A[:, :, 1] = RGBA[:, :, 1]
    A[:, :, 2] = RGBA[:, :, 2]
    if invert:
        A = 1 - A
        A[:, :, -1] = r  # amplitude
        return A
    else:
        A[:, :, -1] = r  # amplitude
        return A


def coordinate_system_with_motion_direction(norm=None, sm=None):
    fig, _, ax = standalone_colorwheel_2d(
        figsize=cm_to_inch([1, 1]),
        cmap=cm_uniform_2d,
        labelpad=-17,
        fontsize=5,
        ticks=[0, 90, 180, 270],
        mode="1d",
        norm=norm,
        sm=sm,
    )
    ax.set_title("motion\ndirection", fontsize=5, pad=0)
    cs_ax = fig.add_axes([0, 0, 1, 1], label="cs_ax")
    coordinate_system(fig=fig, ax=cs_ax)
    return fig


def add_coordinate_system_with_motion_direction(
    fig, title="", pos="east", radius=0.25, x_offset=0, y_offset=0, fontsize=5
):
    cwheel_ax, _ = add_colorwheel_2d(
        fig,
        pos=pos,
        radius=radius,
        x_offset=x_offset,
        y_offset=y_offset,
        fontsize=fontsize,
        labelpad=-17,
        mode="1d",
        ticks=[0, 90, 180, 270],
    )
    x0, y0, x1, y1 = cwheel_ax.get_position().extents
    cs_ax = fig.add_axes([x0, y0, x1 - x0, y1 - y0], label="cs_ax")
    coordinate_system(fig=fig, ax=cs_ax, xlabel="front", ylabel="up", fontsize=fontsize)
    cs_ax.set_title(title, fontsize=fontsize)
    return fig
    fig, _, ax = standalone_colorwheel_2d(
        figsize=cm_to_inch([1, 1]),
        cmap=cm_uniform_2d,
        labelpad=-17,
        fontsize=5,
        ticks=[0, 90, 180, 270],
        mode="1d",
        norm=norm,
        sm=sm,
    )
    ax.set_title("motion\ndirection", fontsize=5, pad=0)
    cs_ax = fig.add_axes([0, 0, 1, 1], label="cs_ax")


def coordinate_system(
    fig=None, ax=None, xlabel="posterior", ylabel="dorsal", fontsize=5
):
    fig, ax = init_plot(figsize=[0.3, 0.3], fontsize=fontsize, fig=fig, ax=ax)
    rm_spines(ax, ("top", "right"))
    ax.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.spines["left"].set_position(("data", 0))
    ax.spines["bottom"].set_position(("data", 0))
    ax.plot(
        1,
        0,
        ">k",
        transform=ax.get_yaxis_transform(),
        ms=1,
        linewidth=0.5,
        clip_on=False,
    )
    ax.plot(
        0,
        1,
        "^k",
        transform=ax.get_xaxis_transform(),
        ms=1,
        linewidth=0.5,
        clip_on=False,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel, fontsize=fontsize, loc="bottom", labelpad=2)
    ax.set_xlabel(xlabel, fontsize=fontsize, loc="left", labelpad=2)
    ax.set_aspect("equal")
    return fig


def add_coordinate_system(
    fig,
    pos="southwest",
    radius=0.1,
    x_offset=0,
    y_offset=0,
    xlabel="posterior",
    ylabel="dorsal",
    fontsize=5,
):
    ax = add_positioned_axis_hex(
        fig, radius=radius, pos=pos, x_offset=x_offset, y_offset=y_offset
    )
    _ = coordinate_system(
        fig=fig, ax=ax, xlabel=xlabel, ylabel=ylabel, fontsize=fontsize
    )


def add_colorwheel_2d(
    fig,
    axes=None,
    pos="southeast",
    radius=0.25,
    x_offset=0,
    y_offset=0,
    sm=None,
    cmap="cm_uniform_2d",
    norm=None,
    fontsize=6,
    N=512,
    labelpad=0,
    invert=False,
    mode="2d",
    ticks=[0, 60, 120],
):
    """
    Args:
        pos: 'southeast', 'east', ...
        radius: radius in percentage of the ax radius
        x_offset: offset in percentage of cbar diameter
        y_offset: offset in percentage of cbar diameter
    """
    cmap = plt.get_cmap(cmap)

    pos = derive_position_for_supplementary_ax_hex(
        fig,
        axes=axes,
        pos=pos,
        radius=radius,
        x_offset=x_offset,
        y_offset=y_offset,
    )
    cb = fig.add_axes(pos, alpha=0)
    cb.patch.set_alpha(0)

    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X * X + Y * Y)
    circular_mask = R < 1
    R[~circular_mask] = 0
    if mode == "1d":
        R[circular_mask] = 1

    PHI = np.arctan2(Y, X)  # + np.pi
    cb.imshow(
        polar_to_cmap(R, PHI, invert=invert, cmap=cmap, norm=norm, sm=sm),
        origin="lower",
    )
    cb.set_axis_off()

    cs = fig.add_axes(pos, polar=True, label="annotation", alpha=0)
    cs.set_facecolor("none")
    cs.set_yticks([])
    cs.set_yticklabels([])  # turn of radial tick labels (yticks)

    # cosmetic changes to tick labels
    cs.tick_params(pad=labelpad, labelsize=fontsize)
    cs.set_xticks(
        np.radians(ticks)
    )  # , np.radians(180), np.radians(240), np.radians(300)])  # , np.pi])
    cs.set_xticklabels(
        [x + "°" for x in np.array(ticks).astype(int).astype(str)]
    )  # , r"180$^\circ$", r"-120$^\circ$", r"-60$^\circ$"])  # , "$\pi$"])

    plt.setp(cs.spines.values(), color="white", linewidth=2)

    return cb, cs


def add_positioned_axis(
    fig,
    pos="right",
    width=0.05,
    height=0.5,
    x_offset=0,
    y_offset=0,
):
    pos = derive_position_for_supplementary_ax(
        fig, pos, width, height, x_offset, y_offset
    )
    new_ax = fig.add_axes(pos, alpha=0)
    new_ax.patch.set_alpha(0)
    return new_ax


def add_positioned_axis_hex(
    fig,
    pos="southwest",
    radius=0.25,
    x_offset=0,
    y_offset=0,
):
    pos = derive_position_for_supplementary_ax_hex(
        fig, pos=pos, radius=radius, x_offset=x_offset, y_offset=y_offset
    )
    new_ax = fig.add_axes(pos, alpha=0)
    new_ax.patch.set_alpha(0)
    return new_ax


def standalone_colorbar(
    cmap,
    norm,
    figsize=[0.1, 0.4],
    orientation="vertical",
    ticks=None,
    alpha=1,
    fontsize=5,
    label="",
    tick_width=1,
    tick_length=2,
    pos="right",
    style="",
    use_math_text=False,
    scilimits=None,
    tick_pad=0,
    rm_outline=True,
    plain=False,
    fig=None,
    ax=None,
):

    if fig is None or ax is None:
        fig = plt.figure(figsize=figsize)
        cbax = fig.add_axes([0, 0, 1, 1], label="cbar")
    else:
        cbax = ax
    cbar = mpl.colorbar.ColorbarBase(
        cbax,
        cmap=cmap,
        norm=norm,
        orientation=orientation,
        ticks=ticks,
        alpha=alpha,
    )
    cbar.set_label(fontsize=fontsize, label=label)
    cbar.ax.tick_params(
        labelsize=fontsize, length=tick_length, width=tick_width, pad=tick_pad
    )

    if pos in ("left", "right"):
        scalarformatter = isinstance(
            cbar.ax.yaxis.get_major_formatter(), mpl.ticker.ScalarFormatter
        )
        cbar.ax.yaxis.set_ticks_position(pos)
        cbar.ax.yaxis.set_label_position(pos)
        cbar.ax.yaxis.get_offset_text().set_fontsize(fontsize)
        cbar.ax.yaxis.get_offset_text().set_horizontalalignment("left")
        cbar.ax.yaxis.get_offset_text().set_verticalalignment("bottom")
    else:
        scalarformatter = isinstance(
            cbar.ax.xaxis.get_major_formatter(), mpl.ticker.ScalarFormatter
        )
        cbar.ax.xaxis.set_ticks_position(pos)
        cbar.ax.xaxis.set_label_position(pos)
        cbar.ax.xaxis.get_offset_text().set_fontsize(fontsize)
        cbar.ax.xaxis.get_offset_text().set_verticalalignment("top")
        cbar.ax.xaxis.get_offset_text().set_horizontalalignment("left")

    if scalarformatter:
        cbar.ax.ticklabel_format(
            style=style, useMathText=use_math_text, scilimits=scilimits
        )

    if rm_outline:
        cbar.outline.set_visible(False)

    if plain:
        cbar.set_ticks([])
        # rm_spines(cbar.ax, rm_xticks=True, rm_yticks=True)

    return fig, cbar


def derive_position_for_supplementary_ax(
    fig, pos="right", width=0.04, height=0.5, x_offset=0, y_offset=0, axes=None
):
    """Returns a position for a supplementary ax.

    pos: right, left, top, or bottom referring to the edge in cartesian space
        that the ax will be placed on.

    Returns: tuple: left, bottom, width, height
    """
    axes = axes if axes is not None else fig.get_axes()
    x0, y0, x1, y1 = np.array([ax.get_position().extents for ax in axes]).T
    x0, y0, x1, y1 = x0.min(), y0.min(), x1.max(), y1.max()
    ax_width = x1 - x0
    ax_height = y1 - y0
    positions = {
        "right": [
            x1 + ax_width * width / 2 + ax_width * width * x_offset,  # left
            y0 + (1 - height) * ax_height / 2 + ax_height * height * y_offset,  # bottom
            ax_width * width,  # width
            ax_height * height,  # height
        ],
        "left": [
            x0 - 3 / 2 * ax_width * width + ax_width * width * x_offset,  # left
            y0 + (1 - height) * ax_height / 2 + y_offset,  # bottom
            ax_width * width,  # width
            ax_height * height,  # height
        ],
        "top": [
            x1
            - ax_width * width
            + ax_width * width * x_offset,  # x0 + (1 - width) * ax_width/2
            y1 + ax_height * height / 2 + ax_height * height * y_offset,
            ax_width * width,
            ax_height * height,
        ],
        # 'top_right': [x1 - x0 + (1 - width) * ax_width/2 + ax_width * width * x_offset,
        #               y1 + ax_height * height / 2 + ax_height * height * y_offset,
        #               ax_width * width,
        #               ax_height * height],
        "bottom": [
            x0 + (1 - width) * ax_width / +ax_width * width * x_offset,
            y0 - 3 / 2 * ax_height * height + ax_height * height * y_offset,
            ax_width * width,
            ax_height * height,
        ],
    }
    return positions[pos]


def derive_position_for_supplementary_ax_hex(
    fig,
    axes=None,
    pos="southwest",
    radius=0.25,
    x_offset=0,
    y_offset=0,
):
    """Returns a position for a supplementary ax.

    pos: southeast, east, northeast, north, northwest, west, southwest, south
        referring to the edge in regular hex space that the ax will be placed on.
    """
    axes = axes if axes is not None else fig.get_axes()
    x0, y0, x1, y1 = np.array([ax.get_position().extents for ax in axes]).T
    x0, y0, x1, y1 = x0.min(), y0.min(), x1.max(), y1.max()
    axes_width = x1 - x0
    axes_height = y1 - y0
    axes_radius = (axes_width + axes_height) / 4
    new_ax_radius = axes_radius * radius
    position = {
        "southeast": [
            x1 + x_offset * 2 * new_ax_radius,
            y1
            - axes_height
            - 2 * new_ax_radius
            + y_offset * 2 * new_ax_radius,  # + radius ** 2 * axes_height,
            2 * new_ax_radius,
            2 * new_ax_radius,
        ],
        "east": [
            x1 + x_offset * 2 * new_ax_radius,
            y1
            - axes_height / 2
            - new_ax_radius
            + y_offset * 2 * new_ax_radius,  # + radius ** 2 * axes_height,
            2 * new_ax_radius,
            2 * new_ax_radius,
        ],
        "northeast": [
            x1 + x_offset * 2 * new_ax_radius,
            y1 + y_offset * 2 * new_ax_radius,  # + radius ** 2 * axes_height,
            2 * new_ax_radius,
            2 * new_ax_radius,
        ],
        "north": [
            x1 - axes_width / 2 - new_ax_radius + x_offset * 2 * new_ax_radius,
            y1 + y_offset * 2 * new_ax_radius,  # + radius ** 2 * axes_height,
            2 * new_ax_radius,
            2 * new_ax_radius,
        ],
        "northwest": [
            x1 - axes_width - 2 * new_ax_radius + x_offset * 2 * new_ax_radius,
            y1 + y_offset * 2 * new_ax_radius,  # + radius ** 2 * axes_height,
            2 * new_ax_radius,
            2 * new_ax_radius,
        ],
        "west": [
            x1 - axes_width - 2 * new_ax_radius + x_offset * 2 * new_ax_radius,
            y1
            - axes_height / 2
            - new_ax_radius
            + y_offset * 2 * new_ax_radius,  # + radius ** 2 * axes_height,
            2 * new_ax_radius,
            2 * new_ax_radius,
        ],
        "southwest": [
            x1 - axes_width - 2 * new_ax_radius + x_offset * 2 * new_ax_radius,
            y1
            - axes_height
            - 2 * new_ax_radius
            + y_offset * 2 * new_ax_radius,  # + radius ** 2 * axes_height,
            2 * new_ax_radius,
            2 * new_ax_radius,
        ],
        "south": [
            x1 - axes_width / 2 - new_ax_radius + x_offset * 2 * new_ax_radius,
            y1
            - axes_height
            - 2 * new_ax_radius
            + y_offset * 2 * new_ax_radius,  # + radius ** 2 * axes_height,
            2 * new_ax_radius,
            2 * new_ax_radius,
        ],
        "origin": [
            x0 + x_offset,
            y0 + y_offset,
            2 * new_ax_radius,
            2 * new_ax_radius,
        ],
    }
    return position[pos]


def add_colorbar_to_fig(
    fig,
    axes=None,
    pos="right",
    width=0.04,
    height=0.5,
    x_offset=0,
    y_offset=0,
    cmap=cm.get_cmap("binary"),
    fontsize=10,
    tick_length=1.5,
    tick_width=0.75,
    rm_outline=True,
    ticks=None,
    formatter=None,
    norm=None,
    label="",
    plain=False,
    use_math_text=False,
    scilimits=None,
    style="",
    alpha=1,
    n_ticks=9,  # only effective if norm is TwoSlopeNorm
    discrete=False,
    n_discrete=None,
    discrete_labels=None,
    n_decimals=2,
):
    """
    Args:
        pos: either 'right', 'left', 'top', or 'bottom'
        width: cbar width in percentage of ax_width
        height: cbar height in percentage of ax_height
        x_offset: offset in percentage of cbar width
        y_offset: offset in percentage of cbar height
    """

    _orientation = "vertical" if pos in ("left", "right") else "horizontal"

    position = derive_position_for_supplementary_ax(
        fig=fig,
        pos=pos,
        width=width,
        height=height,
        x_offset=x_offset,
        y_offset=y_offset,
        axes=axes,
    )
    cbax = fig.add_axes(position, label="cbar")
    # sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    # # breakpoint()
    # cbar = plt.colorbar(
    #     sm, cax=cbax, ticks=ticks, orientation=_orientation, alpha=alpha
    # )
    cbar = mpl.colorbar.ColorbarBase(
        cbax,
        cmap=cmap,
        norm=norm,
        orientation=_orientation,
        ticks=ticks,
        alpha=alpha,
    )
    cbar.set_label(fontsize=fontsize, label=label)
    cbar.ax.tick_params(labelsize=fontsize, length=tick_length, width=tick_width)
    if pos in ("left", "right"):
        scalarformatter = isinstance(
            cbar.ax.yaxis.get_major_formatter(), mpl.ticker.ScalarFormatter
        )
        cbar.ax.yaxis.set_ticks_position(pos)
        cbar.ax.yaxis.set_label_position(pos)
        cbar.ax.yaxis.get_offset_text().set_fontsize(fontsize)
        cbar.ax.yaxis.get_offset_text().set_horizontalalignment("left")
        cbar.ax.yaxis.get_offset_text().set_verticalalignment("bottom")
    else:
        scalarformatter = isinstance(
            cbar.ax.xaxis.get_major_formatter(), mpl.ticker.ScalarFormatter
        )
        cbar.ax.xaxis.set_ticks_position(pos)
        cbar.ax.xaxis.set_label_position(pos)
        cbar.ax.xaxis.get_offset_text().set_fontsize(fontsize)
        cbar.ax.xaxis.get_offset_text().set_verticalalignment("top")
        cbar.ax.xaxis.get_offset_text().set_horizontalalignment("left")

    if scalarformatter:
        cbar.ax.ticklabel_format(
            style=style, useMathText=use_math_text, scilimits=scilimits
        )
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    if rm_outline:
        cbar.outline.set_visible(False)

    if isinstance(norm, TwoSlopeNorm):
        # if isinstance(norm, MidpointNormalize):
        # n_ticks = 7  # len(cbar.ax.yaxis.get_major_ticks())
        vmin = norm.vmin
        vmax = norm.vmax
        # vcenter = norm.midpoint  # norm.vcenter
        vcenter = norm.vcenter
        left_ticks = np.linspace(vmin, vcenter, n_ticks // 2)
        right_ticks = np.linspace(vcenter, vmax, n_ticks // 2)
        # print(left_ticks, right_ticks)
        # right_ticks = np.linspace(vmax / (n_ticks // 2), vmax, n_ticks // 2)
        ticks = ticks or [*left_ticks, *right_ticks[1:]]
        # print(ticks)
        # print(cbar.ax.get_ylim())
        cbar.set_ticks(
            ticks,
            labels=[f"{t:.{n_decimals}f}" for t in ticks],
            fontsize=fontsize,
        )
        # print(cbar.ax.get_ylim())

    if plain:
        cbar.set_ticks([])
        # rm_spines(cbar.ax, rm_xticks=True, rm_yticks=True)

    if discrete:
        # to put ticklabels for discrete colors in the middle
        if not n_discrete:
            raise ValueError(f"n_discrete {n_discrete}")
        lim = cbar.ax.get_ylim() if pos in ["left", "right"] else cbar.ax.get_xlim()
        color_width = (lim[1] - lim[0]) / n_discrete
        label_offset_to_center = color_width / 2
        labels = np.arange(n_discrete)
        loc = np.linspace(*lim, n_discrete, endpoint=False) + label_offset_to_center
        cbar.set_ticks(loc)
        cbar.set_ticklabels(discrete_labels or labels)

    return cbar


def add_colorbar(
    fig,
    ax,
    pos="right",
    width=0.03,
    height=0.5,
    x_offset=0,
    y_offset=0,
    cmap=cm.get_cmap("binary"),
    fontsize=10,
    tick_length=2,
    tick_width=1,
    rm_outline=True,
    ticks=None,
    formatter=None,
    norm=None,
    label="",
    plain=False,
    use_math_text=False,
    scilimits=None,
    style="",
    alpha=1,
    discrete=False,
    n_discrete=None,
    discrete_labels=None,
):
    """
    Args:
        pos: either 'right', 'left', 'top', or 'bottom'
        width: cbar width in percentage of ax_width
        height: cbar height in percentage of ax_height
        x_offset: offset in percentage of cbar width
        y_offset: offset in percentage of cbar height
    """
    kwargs = vars()
    kwargs.pop("ax")
    return add_colorbar_to_fig(**kwargs)


def cmap_map(function, cmap):
    """Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ("red", "green", "blue"):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step: np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(["red", "green", "blue"]):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j, i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1],), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return mpl.colors.LinearSegmentedColormap("colormap", cdict, 1024)


def nbAgg(animation):
    """Changes the matplotlib backend for animations in the notebook."""
    import functools

    @functools.wraps(animation)
    def decorator(*args, **kwargs):
        backend = mpl.get_backend()
        if backend != "nbAgg":
            mpl.pyplot.switch_backend("nbAgg")
        value = animation(*args, **kwargs)
        mpl.pyplot.switch_backend(backend)
        return value

    return decorator


def get_norm(norm=None, vmin=None, vmax=None, midpoint=None, log=None, symlog=None):
    """Returns a normalization object.

    Args:
        norm (Normalize, optional): A class which, when called, can normalize
            data into an interval [vmin, vmax]. Defaults to None.
        vmin (float, optional): Defaults to None.
        vmax (float, optional): Defaults to None.
        midpoint (float, optional): Midpoint value so that data is normalized
            around it. Defaults to None.
        log (bool, optional): if to normalize on a log-scale. Defaults to None.
        symlog (float, optional): if float, normalizes to symlog with linear range
            around the range (-symlog, symlog).

    Behaviour:
        1. returns existing norm if given.
        2. returns TwoSlopeNorm if vmin, vmax and midpoint is not None.
        3. returns LogNorm if vmin, vmax and log is not None.
        4. returns regular Normalize object if vmin and vmax is not None.

    Returns:
        Normalize object
    """
    if norm:
        return norm
    # TODO: compose
    if all(val is not None for val in (vmin, vmax)):
        vmin -= 1e-15
        vmax += 1e-15

    if all(val is not None for val in (vmin, vmax, midpoint)):
        if np.isclose(vmin, midpoint, atol=1e-9):
            vmin = midpoint - vmax
        if np.isclose(vmax, midpoint, atol=1e-9):
            vmax = midpoint - vmin
        # print(vmin, midpoint, vmax)
        return TwoSlopeNorm(vcenter=midpoint, vmin=vmin, vmax=vmax)
    elif all(val is not None for val in (vmin, vmax, log)):
        return mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    elif all(val is not None for val in (vmin, vmax, symlog)):
        v = max(np.abs(vmin), np.abs(vmax))
        return mpl.colors.SymLogNorm(symlog, vmin=-v, vmax=v)
    elif all(val is not None for val in (vmin, vmax)):
        return mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        pass


def get_scalarmapper(
    scalarmapper=None,
    cmap=None,
    norm=None,
    vmin=None,
    vmax=None,
    midpoint=None,
    log=None,
    symlog=None,
):
    """Returns scalarmappable with appropiate colornorm.

    Args:
        scalarmapper (Scalarmappable, optional): This is a mixin class to support
            scalar data to RGBA mapping. Defaults to None.
        cmap (Colormap, optional): Defaults to None.

    Returns:
        scalarmapper, norm
    """
    if scalarmapper:
        return scalarmapper, norm

    norm = get_norm(
        norm=norm,
        vmin=vmin,
        vmax=vmax,
        midpoint=midpoint,
        log=log,
        symlog=symlog,
    )
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap), norm


def get_discrete_color_map(array, cmap, vmin=None, vmax=None, midpoint=None):
    """Get a listed colormap with len(array) values of colors according to cmap
        and either linear or midpoint normalization.

    Note: iterate through it with indices instead of values from array.
    """
    from matplotlib.colors import ListedColormap

    sm, norm = get_scalarmapper(cmap=cmap, vmin=vmin, vmax=vmax, midpoint=midpoint)
    colors = sm.to_rgba(array)
    return ListedColormap(colors)


def get_lims(z, offset, min=None, max=None):
    def sub_nan(val, sub):
        if np.isnan(val):
            return sub
        else:
            return val

    if isinstance(z, (tuple, list)):
        z = list(map(lambda x: get_lims(x, offset), z))
    z = np.array(z)[~np.isinf(z)]
    if not z.any():
        return -1, 1
    _min, _max = np.nanmin(z), np.nanmax(z)
    _range = np.abs(_max - _min)
    _min -= _range * offset
    _max += _range * offset
    _min, _max = sub_nan(_min, 0), sub_nan(_max, 1)
    if min is not None:
        _min = np.min((min, _min))
    if max is not None:
        _max = np.max((max, _max))
    return _min, _max


def node_type_collection_ax_lims_per_batch(data, neuron_types=None, offset=0.1):
    """

    Args:
        data (Dict[str, array]): maps node types onto
                            activity of shape (#samples, #frames, ...).

    Returns:
        List[Tuple]: list of length #samples with ax min and ax max limits.
    """

    neuron_types = neuron_types or list(data.keys())

    # stack as #node_types, #samples, #frames, ...
    stacked_data = []
    for neuron_type in neuron_types:
        trace = dvs.utils.to_numpy(data[neuron_type])
        stacked_data.append(trace)
    stacked_data = np.array(stacked_data)

    lims = []
    for batch in range(stacked_data.shape[1]):
        lims.append(get_lims(stacked_data[:, batch], offset))

    return lims


def filter_trace(trace, N):
    """Filters trace with a bump of size N. Returns a masked array."""
    trace = np.ma.masked_invalid(trace)
    shape = trace.shape
    _data = smooth_gpu(trace, N)
    trace.data[:] = _data[..., : shape[1]].reshape(*shape)
    trace.mask[:, :N] = np.ones(N)
    trace.mask[:, -N:] = np.ones(N)
    return trace


def avg_pool(trace, N):
    """Smoothes (multiple) traces over the second dimension using the GPU.

    Args:
        trace (array): of shape (N, t).
    """
    shape = trace.shape
    trace = trace.reshape(np.prod(shape[:-1]), 1, shape[-1])
    with torch.no_grad():
        trace_smooth = F.avg_pool1d(torch.Tensor(trace), N, N).cpu().numpy()
    return trace_smooth.reshape(shape[0], -1)


def smooth_gpu(trace, N):
    """Smoothes (multiple) traces over the second dimension using the GPU.

    Args:
        trace (array): of shape (..., ..., t).
    """
    shape = trace.shape
    trace = trace.reshape(np.prod(shape[:-1]), shape[-1])
    with torch.no_grad():
        conv = torch.nn.Conv1d(1, 1, N, bias=False, padding=int(N / 2))
        conv.weight[:] = torch.exp(-torch.linspace(-1, 1, N) ** 2)
        conv.weight /= conv.weight.sum()
        # breakpoint()
        trace_smooth = conv(torch.Tensor(trace).unsqueeze(1)).cpu().numpy()

    return trace_smooth.squeeze()[..., : shape[-1]].reshape(*shape)


def width_n_height(N, aspect_ratio, max_width=None, max_height=None):

    if max_width is not None and max_height is not None:
        raise ValueError

    _sqrt = int(np.ceil(np.sqrt(N)))

    gridwidth = np.ceil(_sqrt * aspect_ratio).astype(int)
    gridheight = np.ceil(_sqrt / aspect_ratio).astype(int)

    gridwidth = max(1, min(N, gridwidth, np.ceil(N / gridheight)))
    gridheight = max(1, min(N, gridheight, np.ceil(N / gridwidth)))

    if max_width is not None and gridwidth > max_width:
        gridwidth = max_width
        gridheight = np.ceil(N / gridwidth)

    if max_height is not None and gridheight > max_height:
        gridheight = max_height
        gridwidth = np.ceil(N / gridheight)

    assert gridwidth * gridheight >= N

    return int(gridwidth), int(gridheight)


def plot_stim_contour(
    time, stim, ax, cmap=plt.cm.bone, alpha=0.3, vmin=0, vmax=1, levels=2
):
    y = np.linspace(-20_000, 20_000, 10)
    Z = np.tile(stim, (len(y), 1))
    ax.contourf(time, y, Z, cmap=cmap, levels=levels, alpha=alpha, vmin=vmin, vmax=vmax)


def get_axis_grid(
    alist=None,
    gridwidth=None,
    gridheight=None,
    max_width=None,
    max_height=None,
    fig=None,
    ax=None,
    axes=None,
    aspect_ratio=1,
    offset=5,
    figsize=None,
    scale=3,
    projection=None,
    as_matrix=False,
    fontsize=5,
    wspace=0.1,
    hspace=0.3,
    alpha=1,
    sharex=None,
    sharey=None,
    unmask_n=None,
):
    """Create a list of axes that are aligned in a grid.

    Args:
        alist: list of elements to create grid for.
        scale: figure scaling, figsize = scale*(num_rows, num_columns).
    """
    if alist is not None and (
        gridwidth is None or gridheight is None or gridwidth * gridheight != len(alist)
    ):
        gridwidth, gridheight = width_n_height(
            len(alist), aspect_ratio, max_width=max_width, max_height=max_height
        )
    elif gridwidth and gridheight:
        alist = range(gridwidth * gridheight)
    else:
        raise ValueError("Either specify alist or gridwidth and gridheight manually.")
    unmask_n = unmask_n or len(alist)
    if figsize is not None:
        pass
    elif isinstance(scale, Number):
        figsize = [scale * gridwidth, scale * gridheight]
    elif isinstance(scale, Iterable) and len(scale) == 2:
        figsize = [scale[0] * gridwidth, scale[1] * gridheight]

    if fig is None:
        fig = figure(figsize=figsize, hspace=hspace, wspace=wspace)

    if not isinstance(projection, Iterable) or isinstance(projection, str):
        projection = (projection,) * len(alist)

    if isinstance(ax, Iterable):
        assert len(ax) == len(alist)
        if isinstance(ax, dict):
            ax = list(ax.values())
        return fig, ax, (gridwidth, gridheight)

    if ax:
        # divide an existing ax in a figure
        matrix = np.ones(gridwidth * gridheight) * np.nan
        for i in range(len(alist)):
            matrix[i] = i
        axes = divide_axis_to_grid(
            ax,
            matrix=matrix.reshape(gridwidth, gridheight),
            wspace=wspace,
            hspace=hspace,
        )
        axes = list(axes.values())
    elif axes is None:
        # fill a figure with axes
        axes = []
        _sharex, _sharey = None, None

        for i, element in enumerate(alist):

            if i < unmask_n:
                ax = subplot(
                    "",
                    grid=(gridheight, gridwidth),
                    location=(int(i // gridwidth), int(i % gridwidth)),
                    rowspan=1,
                    colspan=1,
                    sharex=_sharex,
                    sharey=_sharey,
                    projection=projection[i],
                )

                if sharex is not None:
                    sharex = ax
                if sharey is not None:
                    sharey = ax

                axes.append(ax)
            else:
                axes.append(np.nan)

    for ax in axes:
        if isinstance(ax, Axes):
            set_label_size(ax, fontsize)
            ax.patch.set_alpha(alpha)

    if as_matrix:
        axes = np.array(axes).reshape(gridheight, gridwidth)

    return fig, axes, (gridwidth, gridheight)


def set_label_size(ax, fontsize=5):
    ax.tick_params(axis="both", which="major", labelsize=fontsize)


def divide_figure_to_grid(
    matrix=[
        [0, 1, 1, 1, 2, 2, 2],
        [3, 4, 5, 6, 2, 2, 2],
        [3, 7, 7, 7, 2, 2, 2],
        [3, 8, 8, 12, 2, 2, 2],
        [3, 10, 11, 12, 2, 2, 2],
    ],
    as_matrix=False,
    alpha=0,
    constrained_layout=False,
    fig=None,
    figsize=[10, 10],
    projection=None,
    wspace=0.1,
    hspace=0.3,
    no_spines=False,
    keep_nan_axes=False,
    fontsize=5,
    reshape_order="F",
):
    """
    Creates a figure grid specified by the arrangement of unique
    elements in a matrix.
    """

    def _array_to_slice(array):
        step = 1
        start = array.min()
        stop = array.max() + 1
        return slice(start, stop, step)

    fig = plt.figure(figsize=figsize) if fig is None else fig
    fig.set_constrained_layout(constrained_layout)
    matrix = np.ma.masked_invalid(matrix)
    rows, columns = matrix.shape

    gs = GridSpec(rows, columns, figure=fig, hspace=hspace, wspace=wspace)

    axes = {}
    for val in np.unique(matrix[~matrix.mask]):
        _row_ind, _col_ind = np.where(matrix == val)
        _row_slc, _col_slc = _array_to_slice(_row_ind), _array_to_slice(_col_ind)

        _projection = projection[val] if isinstance(projection, dict) else projection
        ax = fig.add_subplot(gs[_row_slc, _col_slc], projection=_projection)
        ax.patch.set_alpha(alpha)
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        if projection is None:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[val] = ax

    if keep_nan_axes:
        for _row_ind, _col_ind in np.array(np.where(np.isnan(matrix))).T:
            _row_slc, _col_slc = _array_to_slice(_row_ind), _array_to_slice(_col_ind)
            ax = fig.add_subplot(gs[_row_slc, _col_slc])
            ax.patch.set_alpha(alpha)
            rm_spines(ax, ("left", "right", "top", "bottom"))

    if no_spines:
        for ax in axes.values():
            rm_spines(ax, rm_xticks=True, rm_yticks=True)

    if as_matrix:

        if reshape_order == "special":
            ax_matrix = np.ones(matrix.shape, dtype=object) * np.nan
            for key, value in axes.items():
                _row_ind, _col_ind = np.where(matrix == key)
                ax_matrix[_row_ind, _col_ind] = value
            axes = ax_matrix
        else:
            _axes = np.ones(matrix.shape, dtype=object).flatten() * np.nan
            _axes[: len(axes)] = np.array(list(axes.values()), dtype=object)
            axes = _axes.reshape(matrix.shape, order=reshape_order)

    return fig, axes


def divide_axis_to_grid(
    ax,
    matrix=((0, 1, 2), (3, 3, 3)),
    wspace=0.1,
    hspace=0.1,
    projection=None,
    despine=True,
    offset=5,
    trim=False,
):
    """Divides an existing axis inside a figure to a grid specified by unique
        elements in a matrix.

    Args:
        ax: existing Axes object.
        matrix (array): grid matrix, where each unique element specifies
            a new axis.
        wspace (float): horizontal space between new axes.
        hspace (float): vertical space between new axes.

    Returns
        tuple: fig, axes (dict)


    Example:
        >>> fig = plt.figure()
        >>> ax = plt.subplot()
        >>> plt.tight_layout()
        >>> divide_axis_to_grid(ax, matrix=[[0, 1, 1, 1, 2, 2, 2],
                                            [3, 4, 5, 6, 2, 2, 2],
                                            [3, 7, 7, 7, 2, 2, 2],
                                            [3, 8, 8, 12, 2, 2, 2],
                                            [3, 10, 11, 12, 2, 2, 2]],
                                wspace=0.1, hspace=0.1)
    """

    # get position of original axis, and dispose it
    x0, y0, x1, y1 = ax.get_position().extents
    ax.set_axis_off()
    ax.patch.set_alpha(0)
    fig = ax.figure
    # ax.remove()

    # get grid shape
    n_row, n_col = np.array(matrix).shape

    # get geometry params
    width = x1 - x0
    height = y1 - y0
    height_per_row = height / n_row
    width_per_col = width / n_col

    _ax_pos = {}

    for i, row in enumerate(matrix):

        for j, _ax in enumerate(row):

            # get occurence of unique element per row and column
            _ax_per_row = sum([1 for _ in np.array(matrix).T[j] if _ == _ax])
            _ax_per_col = sum([1 for _ in row if _ == _ax])

            # compute positioning of _ax
            left = x0 + j * width_per_col + wspace / 2
            bottom = y0 + height - (i + _ax_per_row) * height_per_row + hspace / 2
            _width = width_per_col * _ax_per_col - min(
                wspace / 2, width_per_col * _ax_per_col
            )
            _height = height_per_row * _ax_per_row - min(
                hspace / 2, height_per_row * _ax_per_row
            )

            # store positioning
            if not _ax in _ax_pos and not np.isnan(_ax):
                _ax_pos[_ax] = [left, bottom, _width, _height]

    # add axis to existing figure and store in dict
    # sns.set(context='paper', style='ticks', font_scale=1.5)
    # sns.axes_style({'axes.edgecolor': '.6', 'axes.linewidth': 5.0})
    axes = {k: None for k in _ax_pos}
    for _ax, pos in _ax_pos.items():
        axes[_ax] = fig.add_axes(pos, projection=projection)
        # if despine is True:
        #     sns.despine(ax=axes[_ax], offset=offset, trim=trim)

    return axes


def merge_axes(fig, axes, fontsize=5):
    """To merge multiple axis to one to allow flexible adapations of an axis grid.

    Args:
        axes: list of axes to 'merge'.

    Note the merge is in fact superimposing a large ax on the maximal extents across
    all provided individual axes.

    Note the individual axes will be removed.

    Note the new ax will take the first properties found on the first ax.
    """

    # determine the minimal (x0, y0) and maximal (x1, y1) to place one large axes across
    x0, y0, x1, y1 = np.array([ax.get_position().extents for ax in axes]).T
    new_x0, new_y0, new_x1, new_y1 = min(x0), min(y0), max(x1), max(y1)
    new_width = new_x1 - new_x0
    new_height = new_y1 - new_y0
    superax = fig.add_axes([new_x0, new_y0, new_width, new_height])
    superax.tick_params(axis="both", which="major", labelsize=fontsize)
    for ax in axes:
        ax.remove()
    return superax


def set_aspect(ax, ratio=1):
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)


def regular_hex_ax_scatter(n_rows, n_columns, mode="pointy", **kwargs):
    x, y = dvs.utils.hex_rows(n_rows, n_columns, mode=mode)
    fig, axes, pos = dvs.plots.plt_utils.regular_ax_scatter(x, y, **kwargs)
    return fig, axes, pos


def regular_ax_scatter(
    x,
    y,
    fig=None,
    figsize=[7, 7],
    hspace=0,
    wspace=0,
    hpad=0.1,
    wpad=0.1,
    alpha=0,
    zorder=10,
    offset=0,
    trim=False,
    as_matrix=False,
    projection=None,
    labels=None,
):
    """Creates scattered axes in a given or new figure.

    Args:
        x (list or array): representing left coordinates of the axes.
        y (list or array): representing bottom coordinates of the axes.
        fig (None or maptlotlib.figure): Defaults to None.
        figsize (tuple or list): size of new figure. Without effect if a figure is provided. Defaults to [7, 7].
        hpad (float): spacing to the horizontal borders of the figure. Defaults to 0.1.
        wpad (float): spacing to the vertical borders of the figure. Defaults to 0.1.
        alpha (float): alpha value for the white background of each ax. Defaults to 0.

    Returns:
        fig
        list of axes for each point in (x, y)
        scaled version of x and y in figure coordinates
    """
    x, y = np.array(x), np.array(y)
    assert len(x) == len(y)

    # Min-Max Scale x-Positions.
    # breakpoint()
    width = (1 - 2 * wpad) / (2 * np.ceil(np.median(np.unique(x))))
    width = width - wspace * width  # len(np.unique(np.round(x, 2)))#))
    x = (x - np.min(x)) / (np.max(x) + np.min(x)) * (1 - width) * (1 - wpad * 2) + wpad

    # Min-Max Scale y-Positions.
    height = (1 - 2 * hpad) / (2 * np.ceil(np.median(np.unique(y))))
    height = (
        height - hspace * height
    )  # len(np.unique(np.round(y, 2))) #np.max(np.unique(np.round(y, 2), return_counts=True))
    y = (y - np.min(y)) / (np.max(y) + np.min(y)) * (1 - height) * (1 - hpad * 2) + hpad

    # Create axes in figure.
    fig = fig or plt.figure(figsize=figsize)
    axes = []
    for i, (_x, _y) in enumerate(zip(x, y)):
        ax = fig.add_axes(
            [_x, _y, width, height],
            projection=projection,
            label=labels[i] if labels is not None else None,
        )
        ax.set_zorder(zorder)
        ax.patch.set_alpha(alpha)
        # sns.despine(ax=ax, offset=offset, trim=trim)
        axes.append(ax)

    center = []
    for ax in axes:
        _, (_center, _, _) = get_ax_positions(ax)
        center.append(_center.flatten().tolist())
    return fig, axes, center


def black_or_white(background_rgb):
    """Based on the value in hsv colorspace of the background, return the font color of the text object."""
    if len(background_rgb) == 4:
        # assume rgba
        background_rgb = background_rgb[:-1]
    assert len(background_rgb) == 3
    return "white" if mpl.colors.rgb_to_hsv(background_rgb)[-1] < 0.5 else "black"


def fit_fontsize(text, width, height, fig=None, ax=None):
    """
    Adjusts the fontsize of a matplotlib text object to fit into a given width and height.
    """
    fig = fig or plt.gcf()
    ax = ax or plt.gca()
    renderer = fig.canvas.get_renderer()
    bbox_text = text.get_window_extent(renderer=renderer)
    bbox_text = Bbox(ax.transData.inverted().transform(bbox_text))
    fits_width = bbox_text.width < width
    fits_height = bbox_text.height < height
    if not all((fits_width, fits_height)):
        text.set_fontsize(text.get_fontsize() - 1)
        fit_fontsize(text, width, height, fig, ax)


# colormap from
# https://stackoverflow.com/questions/23712207/cyclic-colormap-without-visual-distortions-for-use-in-phase-angle-plots
cm_uniform_2d = np.array(
    [
        [0.91510904, 0.55114749, 0.67037311],
        [0.91696411, 0.55081563, 0.66264366],
        [0.91870995, 0.55055664, 0.65485881],
        [0.92034498, 0.55037149, 0.64702356],
        [0.92186763, 0.55026107, 0.63914306],
        [0.92327636, 0.55022625, 0.63122259],
        [0.9245696, 0.55026781, 0.62326754],
        [0.92574582, 0.5503865, 0.6152834],
        [0.92680349, 0.55058299, 0.6072758],
        [0.92774112, 0.55085789, 0.59925045],
        [0.9285572, 0.55121174, 0.59121319],
        [0.92925027, 0.551645, 0.58316992],
        [0.92981889, 0.55215808, 0.57512667],
        [0.93026165, 0.55275127, 0.56708953],
        [0.93057716, 0.5534248, 0.55906469],
        [0.93076407, 0.55417883, 0.55105838],
        [0.93082107, 0.55501339, 0.54307696],
        [0.93074689, 0.55592845, 0.53512681],
        [0.9305403, 0.55692387, 0.52721438],
        [0.93020012, 0.55799943, 0.51934621],
        [0.92972523, 0.55915477, 0.51152885],
        [0.92911454, 0.56038948, 0.50376893],
        [0.92836703, 0.56170301, 0.49607312],
        [0.92748175, 0.56309471, 0.48844813],
        [0.9264578, 0.56456383, 0.48090073],
        [0.92529434, 0.56610951, 0.47343769],
        [0.92399062, 0.56773078, 0.46606586],
        [0.92254595, 0.56942656, 0.45879209],
        [0.92095971, 0.57119566, 0.4516233],
        [0.91923137, 0.5730368, 0.44456642],
        [0.91736048, 0.57494856, 0.4376284],
        [0.91534665, 0.57692945, 0.43081625],
        [0.91318962, 0.57897785, 0.42413698],
        [0.91088917, 0.58109205, 0.41759765],
        [0.90844521, 0.58327024, 0.41120533],
        [0.90585771, 0.58551053, 0.40496711],
        [0.90312676, 0.5878109, 0.3988901],
        [0.90025252, 0.59016928, 0.39298143],
        [0.89723527, 0.5925835, 0.38724821],
        [0.89407538, 0.59505131, 0.38169756],
        [0.89077331, 0.59757038, 0.37633658],
        [0.88732963, 0.60013832, 0.37117234],
        [0.88374501, 0.60275266, 0.36621186],
        [0.88002022, 0.6054109, 0.36146209],
        [0.87615612, 0.60811044, 0.35692989],
        [0.87215369, 0.61084868, 0.352622],
        [0.86801401, 0.61362295, 0.34854502],
        [0.86373824, 0.61643054, 0.34470535],
        [0.85932766, 0.61926872, 0.3411092],
        [0.85478365, 0.62213474, 0.3377625],
        [0.85010767, 0.6250258, 0.33467091],
        [0.84530131, 0.62793914, 0.3318397],
        [0.84036623, 0.63087193, 0.32927381],
        [0.8353042, 0.63382139, 0.32697771],
        [0.83011708, 0.63678472, 0.32495541],
        [0.82480682, 0.63975913, 0.32321038],
        [0.81937548, 0.64274185, 0.32174556],
        [0.81382519, 0.64573011, 0.32056327],
        [0.80815818, 0.6487212, 0.31966522],
        [0.80237677, 0.65171241, 0.31905244],
        [0.79648336, 0.65470106, 0.31872531],
        [0.79048044, 0.65768455, 0.31868352],
        [0.78437059, 0.66066026, 0.31892606],
        [0.77815645, 0.66362567, 0.31945124],
        [0.77184076, 0.66657827, 0.32025669],
        [0.76542634, 0.66951562, 0.3213394],
        [0.75891609, 0.67243534, 0.32269572],
        [0.75231298, 0.67533509, 0.32432138],
        [0.74562004, 0.6782126, 0.32621159],
        [0.73884042, 0.68106567, 0.32836102],
        [0.73197731, 0.68389214, 0.33076388],
        [0.72503398, 0.68668995, 0.33341395],
        [0.7180138, 0.68945708, 0.33630465],
        [0.71092018, 0.69219158, 0.33942908],
        [0.70375663, 0.69489159, 0.34278007],
        [0.69652673, 0.69755529, 0.34635023],
        [0.68923414, 0.70018097, 0.35013201],
        [0.6818826, 0.70276695, 0.35411772],
        [0.67447591, 0.70531165, 0.3582996],
        [0.667018, 0.70781354, 0.36266984],
        [0.65951284, 0.71027119, 0.36722061],
        [0.65196451, 0.71268322, 0.37194411],
        [0.64437719, 0.71504832, 0.37683259],
        [0.63675512, 0.71736525, 0.38187838],
        [0.62910269, 0.71963286, 0.38707389],
        [0.62142435, 0.72185004, 0.39241165],
        [0.61372469, 0.72401576, 0.39788432],
        [0.60600841, 0.72612907, 0.40348469],
        [0.59828032, 0.72818906, 0.40920573],
        [0.59054536, 0.73019489, 0.41504052],
        [0.58280863, 0.73214581, 0.42098233],
        [0.57507535, 0.7340411, 0.42702461],
        [0.5673509, 0.7358801, 0.43316094],
        [0.55964082, 0.73766224, 0.43938511],
        [0.55195081, 0.73938697, 0.44569104],
        [0.54428677, 0.74105381, 0.45207286],
        [0.53665478, 0.74266235, 0.45852483],
        [0.52906111, 0.74421221, 0.4650414],
        [0.52151225, 0.74570306, 0.47161718],
        [0.5140149, 0.74713464, 0.47824691],
        [0.506576, 0.74850672, 0.48492552],
        [0.49920271, 0.74981912, 0.49164808],
        [0.49190247, 0.75107171, 0.4984098],
        [0.48468293, 0.75226438, 0.50520604],
        [0.47755205, 0.7533971, 0.51203229],
        [0.47051802, 0.75446984, 0.5188842],
        [0.46358932, 0.75548263, 0.52575752],
        [0.45677469, 0.75643553, 0.53264815],
        [0.45008317, 0.75732863, 0.5395521],
        [0.44352403, 0.75816207, 0.54646551],
        [0.43710682, 0.758936, 0.55338462],
        [0.43084133, 0.7596506, 0.56030581],
        [0.42473758, 0.76030611, 0.56722555],
        [0.41880579, 0.76090275, 0.5741404],
        [0.41305637, 0.76144081, 0.58104704],
        [0.40749984, 0.76192057, 0.58794226],
        [0.40214685, 0.76234235, 0.59482292],
        [0.39700806, 0.7627065, 0.60168598],
        [0.39209414, 0.76301337, 0.6085285],
        [0.38741566, 0.76326334, 0.6153476],
        [0.38298304, 0.76345681, 0.62214052],
        [0.37880647, 0.7635942, 0.62890454],
        [0.37489579, 0.76367593, 0.63563704],
        [0.37126045, 0.76370246, 0.64233547],
        [0.36790936, 0.76367425, 0.64899736],
        [0.36485083, 0.76359176, 0.6556203],
        [0.36209245, 0.76345549, 0.66220193],
        [0.359641, 0.76326594, 0.66873999],
        [0.35750235, 0.76302361, 0.67523226],
        [0.35568141, 0.76272903, 0.68167659],
        [0.35418202, 0.76238272, 0.68807086],
        [0.3530069, 0.76198523, 0.69441305],
        [0.35215761, 0.7615371, 0.70070115],
        [0.35163454, 0.76103888, 0.70693324],
        [0.35143685, 0.76049114, 0.71310742],
        [0.35156253, 0.75989444, 0.71922184],
        [0.35200839, 0.75924936, 0.72527472],
        [0.3527701, 0.75855647, 0.73126429],
        [0.3538423, 0.75781637, 0.73718884],
        [0.3552186, 0.75702964, 0.7430467],
        [0.35689171, 0.75619688, 0.74883624],
        [0.35885353, 0.75531868, 0.75455584],
        [0.36109522, 0.75439565, 0.76020396],
        [0.36360734, 0.75342839, 0.76577905],
        [0.36637995, 0.75241752, 0.77127961],
        [0.3694027, 0.75136364, 0.77670417],
        [0.37266493, 0.75026738, 0.7820513],
        [0.37615579, 0.74912934, 0.78731957],
        [0.37986429, 0.74795017, 0.79250759],
        [0.38377944, 0.74673047, 0.797614],
        [0.38789026, 0.74547088, 0.80263746],
        [0.3921859, 0.74417203, 0.80757663],
        [0.39665568, 0.74283455, 0.81243022],
        [0.40128912, 0.74145908, 0.81719695],
        [0.406076, 0.74004626, 0.82187554],
        [0.41100641, 0.73859673, 0.82646476],
        [0.41607073, 0.73711114, 0.83096336],
        [0.4212597, 0.73559013, 0.83537014],
        [0.42656439, 0.73403435, 0.83968388],
        [0.43197625, 0.73244447, 0.8439034],
        [0.43748708, 0.73082114, 0.84802751],
        [0.44308905, 0.72916502, 0.85205505],
        [0.44877471, 0.72747678, 0.85598486],
        [0.45453694, 0.72575709, 0.85981579],
        [0.46036897, 0.72400662, 0.8635467],
        [0.4662644, 0.72222606, 0.86717646],
        [0.47221713, 0.72041608, 0.87070395],
        [0.47822138, 0.71857738, 0.87412804],
        [0.4842717, 0.71671065, 0.87744763],
        [0.4903629, 0.71481659, 0.88066162],
        [0.49649009, 0.71289591, 0.8837689],
        [0.50264864, 0.71094931, 0.88676838],
        [0.50883417, 0.70897752, 0.88965898],
        [0.51504253, 0.70698127, 0.89243961],
        [0.52126981, 0.70496128, 0.8951092],
        [0.52751231, 0.70291829, 0.89766666],
        [0.53376652, 0.70085306, 0.90011093],
        [0.54002912, 0.69876633, 0.90244095],
        [0.54629699, 0.69665888, 0.90465565],
        [0.55256715, 0.69453147, 0.90675397],
        [0.55883679, 0.69238489, 0.90873487],
        [0.56510323, 0.69021993, 0.9105973],
        [0.57136396, 0.68803739, 0.91234022],
        [0.57761655, 0.68583808, 0.91396258],
        [0.58385872, 0.68362282, 0.91546336],
        [0.59008831, 0.68139246, 0.91684154],
        [0.59630323, 0.67914782, 0.9180961],
        [0.60250152, 0.67688977, 0.91922603],
        [0.60868128, 0.67461918, 0.92023033],
        [0.61484071, 0.67233692, 0.921108],
        [0.62097809, 0.67004388, 0.92185807],
        [0.62709176, 0.66774097, 0.92247957],
        [0.63318012, 0.66542911, 0.92297153],
        [0.63924166, 0.66310923, 0.92333301],
        [0.64527488, 0.66078227, 0.92356308],
        [0.65127837, 0.65844919, 0.92366082],
        [0.65725076, 0.65611096, 0.92362532],
        [0.66319071, 0.65376857, 0.92345572],
        [0.66909691, 0.65142302, 0.92315115],
        [0.67496813, 0.64907533, 0.92271076],
        [0.68080311, 0.64672651, 0.92213374],
        [0.68660068, 0.64437763, 0.92141929],
        [0.69235965, 0.64202973, 0.92056665],
        [0.69807888, 0.6396839, 0.91957507],
        [0.70375724, 0.63734122, 0.91844386],
        [0.70939361, 0.63500279, 0.91717232],
        [0.7149869, 0.63266974, 0.91575983],
        [0.72053602, 0.63034321, 0.91420578],
        [0.72603991, 0.62802433, 0.9125096],
        [0.7314975, 0.62571429, 0.91067077],
        [0.73690773, 0.62341425, 0.9086888],
        [0.74226956, 0.62112542, 0.90656328],
        [0.74758193, 0.61884899, 0.90429382],
        [0.75284381, 0.6165862, 0.90188009],
        [0.75805413, 0.61433829, 0.89932181],
        [0.76321187, 0.6121065, 0.89661877],
        [0.76831596, 0.6098921, 0.89377082],
        [0.77336536, 0.60769637, 0.89077786],
        [0.77835901, 0.6055206, 0.88763988],
        [0.78329583, 0.6033661, 0.88435693],
        [0.78817477, 0.60123418, 0.88092913],
        [0.79299473, 0.59912616, 0.87735668],
        [0.79775462, 0.59704339, 0.87363986],
        [0.80245335, 0.59498722, 0.86977904],
        [0.8070898, 0.592959, 0.86577468],
        [0.81166284, 0.5909601, 0.86162732],
        [0.81617134, 0.5889919, 0.8573376],
        [0.82061414, 0.58705579, 0.85290625],
        [0.82499007, 0.58515315, 0.84833413],
        [0.82929796, 0.58328538, 0.84362217],
        [0.83353661, 0.58145389, 0.83877142],
        [0.8377048, 0.57966009, 0.83378306],
        [0.8418013, 0.57790538, 0.82865836],
        [0.84582486, 0.57619119, 0.82339871],
        [0.84977422, 0.57451892, 0.81800565],
        [0.85364809, 0.57289, 0.8124808],
        [0.85744519, 0.57130585, 0.80682595],
        [0.86116418, 0.56976788, 0.80104298],
        [0.86480373, 0.56827749, 0.79513394],
        [0.86836249, 0.56683612, 0.789101],
        [0.87183909, 0.56544515, 0.78294645],
        [0.87523214, 0.56410599, 0.77667274],
        [0.87854024, 0.56282002, 0.77028247],
        [0.88176195, 0.56158863, 0.76377835],
        [0.88489584, 0.56041319, 0.75716326],
        [0.88794045, 0.55929505, 0.75044023],
        [0.89089432, 0.55823556, 0.74361241],
        [0.89375596, 0.55723605, 0.73668312],
        [0.89652387, 0.55629781, 0.72965583],
        [0.89919653, 0.55542215, 0.72253414],
        [0.90177242, 0.55461033, 0.71532181],
        [0.90425, 0.55386358, 0.70802274],
        [0.90662774, 0.55318313, 0.70064098],
        [0.90890408, 0.55257016, 0.69318073],
        [0.91107745, 0.55202582, 0.68564633],
        [0.91314629, 0.55155124, 0.67804225],
    ]
)

cm_uniform_2d = colors.ListedColormap(cm_uniform_2d)
try:
    plt.get_cmap("cm_uniform_2d")
except:
    plt.cm.register_cmap("cm_uniform_2d", cmap=cm_uniform_2d)


def standalone_colorwheel_2d(
    figsize=[1, 1],
    sm=None,
    cmap=cm_uniform_2d,
    norm=None,
    fontsize=6,
    N=512,
    labelpad=0,
    ticks=[0, 60, 120],
    mode="2d",
    invert_y=False,
):
    """
    Args:
        pos: 'southeast', 'east', ...
        radius: radius in percentage of the ax radius
        x_offset: offset in percentage of cbar diameter
        y_offset: offset in percentage of cbar diameter
    """
    fig = plt.figure(figsize=figsize)

    cb = fig.add_axes([0, 0, 1, 1])

    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    if invert_y:
        Y *= -1
    R = np.sqrt(X * X + Y * Y)
    circular_mask = R < 1
    R[~circular_mask] = 0
    if mode == "1d":
        R[circular_mask] = 1
    PHI = np.arctan2(Y, X)  # + np.pi
    cb.imshow(
        polar_to_cmap(R, PHI, invert=False, cmap=cmap, norm=norm, sm=sm),
        origin="lower",
        zorder=0,
    )
    cb.set_axis_off()

    cs = fig.add_axes([0, 0, 1, 1], polar=True, label="annotation")
    cs.set_facecolor("none")
    cs.set_yticks([])
    cs.set_yticklabels([])  # turn of radial tick labels (yticks)

    # cosmetic changes to tick labels
    cs.tick_params(pad=labelpad, labelsize=fontsize, zorder=5)
    cs.set_xticks(
        np.radians(ticks)
    )  # , np.radians(180), np.radians(240), np.radians(300)])  # , np.pi])
    cs.set_xticklabels(
        [x + "°" for x in np.array(ticks).astype(int).astype(str)],
    )  # , r"180$^\circ$", r"-120$^\circ$", r"-60$^\circ$"])  # , "$\pi$"])

    plt.setp(cs.spines.values(), color="white", linewidth=2, zorder=3)

    return fig, cb, cs


def clear_font_cache():
    import shutil
    from pathlib import Path

    shutil.rmtree(Path("~/.cache/matplotlib").expanduser())


def scatter_on_violins_or_bars(
    data,
    ax,
    xticks=None,
    indices=None,
    s=5,
    zorder=100,
    facecolor="none",
    edgecolor="k",
    linewidth=0.5,
    alpha=0.35,
    uniform=[-0.35, 0.35],
    seed=42,
    marker="o",
    **kwargs,
):
    """
    data (array): shape (n_samples, n_random_variables).
    indices (array, optional): selection along sample dimension.
    """
    random = np.random.RandomState(seed)

    if xticks is None:
        xticks = ax.get_xticks()
    indices = indices if indices is not None else range(data.shape[0])

    if (
        not isinstance(facecolor, Iterable)
        or len(facecolor) != len(data)
        or isinstance(facecolor, str)
    ):
        facecolor = (facecolor,) * len(indices)

    if (
        not isinstance(edgecolor, Iterable)
        or len(edgecolor) != len(data)
        or isinstance(edgecolor, str)
    ):
        edgecolor = (edgecolor,) * len(indices)

    for i, model_index in enumerate(indices):
        ax.scatter(
            xticks + random.uniform(*uniform, size=len(xticks)),
            data[model_index],
            s=s,
            zorder=zorder,
            facecolor=facecolor[i],
            edgecolor=edgecolor[i],
            linewidth=linewidth,
            alpha=alpha,
            marker=marker,
            **kwargs,
        )


def flash_response_color_labels(ax):
    on = [
        key for key, value in dvs.utils.groundtruth_utils.polarity.items() if value == 1
    ]
    off = [
        key
        for key, value in dvs.utils.groundtruth_utils.polarity.items()
        if value == -1
    ]
    color_labels(on, dvs.utils.color_utils.ON_FR, ax)
    color_labels(off, dvs.utils.color_utils.OFF_FR, ax)
    return ax


def color_labels(labels, color, ax):
    for t in ax.texts:
        if t.get_text() in labels:
            t.set_color(color)

    for tick in ax.xaxis.get_major_ticks():
        if tick.label1.get_text() in labels:
            tick.label1.set_color(color)

    for tick in ax.yaxis.get_major_ticks():
        if tick.label1.get_text() in labels:
            tick.label1.set_color(color)


def boldify_labels(labels, ax):
    for t in ax.texts:
        if t.get_text() in labels:
            t.set_weight("bold")

    for tick in ax.xaxis.get_major_ticks():
        if tick.label1.get_text() in labels:
            tick.label1.set_weight("bold")

    for tick in ax.yaxis.get_major_ticks():
        if tick.label1.get_text() in labels:
            tick.label1.set_weight("bold")


def patch_type_texts(ax):
    patch_ax_texts(ax, "CT1(M10)", "CT1(M10)")
    patch_ax_texts(ax, "CT1(Lo1)", "CT1(Lo1)")
    patch_ax_texts(ax, "TmY18", "TmY18")
    return ax


def patch_ax_texts(ax, original, replacement):

    title = ax.title
    if original in title.get_text():
        title.set_text(title.get_text().replace(original, replacement))

    for t in ax.texts:
        if original in t.get_text():
            t.set_text(t.get_text().replace(original, replacement))

    new = tuple()
    for t in ax.xaxis.get_ticklabels():
        if original in t.get_text():
            t.set_text(t.get_text().replace(original, replacement))
        new += (t,)
    ax.xaxis.set_ticklabels(new)

    new = tuple()
    for t in ax.yaxis.get_ticklabels():
        if original in t.get_text():
            t.set_text(t.get_text().replace(original, replacement))
        new += (t,)
    ax.yaxis.set_ticklabels(new)
    return ax


def add_panel_letter(letter, fig, fontsize=12, fontweight="bold"):
    axes = np.array(fig.axes)
    lefts, bottoms, rights, tops = np.atleast_2d(
        np.array([ax.get_position().extents for ax in axes])
    ).T
    #     #central coordinates
    #     centers_x = lefts + (rights - lefts) / 2
    #     centers_y = bottoms + (tops - bottoms) / 2

    text_positions = []
    for ax in axes:
        title = ax.title
        if title:
            text_positions.append(
                fig.transFigure.inverted().transform(
                    title.get_transform().transform(title.get_position())
                )
            )
        # for text in ax.texts:
        #     transform = text.get_transform()
        #     position = text.get_position()
        #     # print(transform, position)
        #     text_positions.append(fig.transFigure.inverted().transform(transform.transform(position)))

    for text in fig.texts:
        text_positions.append(text.get_position())

    tp_x, tp_y = 1e15, -1e15
    if text_positions:
        tp_x, tp_y = np.array(text_positions).T
    # breakpoint()
    topmost = max(tops.max(), np.max(tp_y))
    leftmost = min(lefts.min(), np.min(tp_x))
    fig.text(
        leftmost,
        topmost,
        letter,
        fontweight=fontweight,
        ha="right",
        va="center",
    )


def set_spine_tick_params(
    ax,
    spinewidth=0.25,
    tickwidth=0.25,
    ticklength=3,
    ticklabelpad=2,
    labelsize=6,
    spines=("top", "right", "bottom", "left"),
):
    """Set spine and tick widths and lengths."""
    for s in spines:
        ax.spines[s].set_linewidth(spinewidth)
    ax.tick_params(axis="both", width=tickwidth, length=ticklength, pad=ticklabelpad)


# def add_cluster_marker(
#     fig, ax, marker="o", color="#6CA2CA", s=20, xy=(0.15, 0.15), pad=0.1
# ):
#     left, bottom, width, height = ax.get_position().bounds
#     clm_ax = fig.add_axes(
#         (
#             left - left * pad,
#             bottom - bottom * pad,
#             width + width * pad,
#             height + height * pad,
#         ),
#         label=marker,
#     )
#     clm_ax.patch.set_alpha(0)
#     dvs.plots.plt_utils.rm_spines(
#         clm_ax, visible=False, rm_xticks=True, rm_yticks=True
#     )
#     clm_ax.set_ylim(0, 1)
#     clm_ax.set_xlim(0, 1)
#     clm_ax.scatter(*xy, marker=marker, s=s, color=color)


def add_cluster_marker(
    fig, ax, marker="o", marker_size=15, color="#4F73AE", x_offset=0, y_offset=0
):
    # make all axes transparent to see the marker regardless where on the figure
    # plane it is
    for _ax in fig.axes:
        _ax.patch.set_alpha(0)

    # create an invisible ax that spans the entire figure to scatter the marker on it
    overlay_ax = [ax for ax in fig.axes if ax.get_label() == "overlay"]
    overlay_ax = (
        overlay_ax[0] if overlay_ax else fig.add_axes([0, 0, 1, 1], label="overlay")
    )
    overlay_ax.set_ylim(0, 1)
    overlay_ax.set_xlim(0, 1)
    overlay_ax.patch.set_alpha(0)
    dvs.plots.plt_utils.rm_spines(
        overlay_ax, visible=False, rm_xticks=True, rm_yticks=True
    )

    # get where the axis is actually positioned, that will be annotated with the
    # marker
    left, bottom, width, height = ax.get_position().bounds

    # scatter the marker relative to that position of the ax
    overlay_ax.scatter(
        left + x_offset * width,
        bottom + y_offset * height,
        marker=marker,
        s=marker_size,
        color=color,
    )
