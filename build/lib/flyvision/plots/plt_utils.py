"""Plotting utils.""" ""
from typing import Iterable, Tuple
from numbers import Number

import torch
import torch.nn.functional as F

import numpy as np

import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.colors as colors

from matplotlib import colormaps as cm
from matplotlib.cm import ScalarMappable
from matplotlib.colors import hex2color, Normalize
from matplotlib.colors import ListedColormap


def init_plot(
    figsize=[1, 1],
    title="",
    fontsize=5,
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
    """Creates fig and axis object with certain default settings.

    Args:
        figsize (list, optional): Defaults to [1, 1].
        title (str, optional): Defaults to ''.
        fontsize (int, optional): Defaults to 5.
        fig (Figure, optional): If None, creates new figure. Defaults to None.
        ax (Axes, optional): If None, creates new axis. Defaults to None.
            `fontsize` and `transparent` effect existing axis.
        projection (str, optional): Defaults to None. E.g. `polar`.
        set_axis_off (bool, optional): Defaults to False.
        transparent (bool, optional): Defaults to False.
        face_alpha (float, optional): Defaults to 0.
        position (list, optional): Position for newly created axis.
            Defaults to None.
        title_pos (str, optional): Defaults to 'center'.
        title_y (float, optional): Defaults to None.

    Returns:
        fig, ax
    """

    # initialize figure and ax
    if fig is None:
        fig = plt.figure(figsize=figsize, layout="constrained")
    if ax is not None:
        ax.set_title(title, fontsize=fontsize, loc=title_pos, y=title_y)
    else:
        ax = fig.add_subplot(projection=projection)
        if position is not None:
            ax.set_position(position)
        ax.patch.set_alpha(face_alpha)
        ax.set_title(title, fontsize=fontsize, loc=title_pos, y=title_y)

        if set_axis_off:
            ax.set_axis_off()
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    if transparent:
        ax.patch.set_alpha(0)
    return fig, ax


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Truncate colormap."""
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)),
    )
    return new_cmap


def rm_spines(
    ax,
    spines=("top", "right", "bottom", "left"),
    visible=False,
    rm_xticks=True,
    rm_yticks=True,
):
    """Removes spines and ticks from axis."""
    for spine in spines:
        ax.spines[spine].set_visible(visible)
    if ("top" in spines or "bottom" in spines) and rm_xticks:
        ax.xaxis.set_ticklabels([])
        ax.xaxis.set_ticks_position("none")
    if ("left" in spines or "right" in spines) and rm_yticks:
        ax.yaxis.set_ticklabels([])
        ax.yaxis.set_ticks_position("none")
    return ax


def get_ax_positions(axes):
    """Returns the positions of the axes in the figure.

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


def is_hex(color):
    """Checks if color is hex.""" ""
    return "#" in color


def is_integer_rgb(color):
    """Checks if color is integer rgb."""
    try:
        return any([c > 1 for c in color])
    # if color is string
    except TypeError:
        return False


def get_alpha_colormap(saturated_color, number_of_shades):
    """Create a colormap from a color and a number of shades."""
    if is_hex(saturated_color):
        rgba = [*hex2color(saturated_color)[:3], 0]
    elif is_integer_rgb(saturated_color):
        rgba = [*list(np.array(saturated_color) / 255.0), 0]

    N = number_of_shades
    colors = []
    alphas = np.linspace(1 / N, 1, N)[::-1]
    for alpha in alphas:
        rgba[-1] = alpha
        colors.append(rgba.copy())

    return ListedColormap(colors)


def polar_to_cmap(
    r, theta, invert=True, cmap=plt.cm.twilight_shifted, norm=None, sm=None
):
    """Maps angle to rgb and amplitude to alpha and returns the resulting array."""
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
    """Adds a colorwheel to a figure.

    Args:
        pos: 'southeast', 'east', 'northeast', 'north', 'northwest', 'west', 'southwest', 'south', 'origin'.
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
    cs.set_yticklabels([])  # turn off radial tick labels (yticks)

    # cosmetic changes to tick labels
    cs.tick_params(pad=labelpad, labelsize=fontsize)
    cs.set_xticks(np.radians(ticks))
    cs.set_xticklabels([x + "°" for x in np.array(ticks).astype(int).astype(str)])

    plt.setp(cs.spines.values(), color="white", linewidth=2)

    return cb, cs


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

    Args:
        pos: 'southwest', 'southeast', 'northeast', 'northwest', 'north',
            'south', 'east', 'west', 'origin'

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
    """Adds a colorbar to a figure.

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

    if rm_outline:
        cbar.outline.set_visible(False)

    if isinstance(norm, TwoSlopeNorm):
        vmin = norm.vmin
        vmax = norm.vmax
        vcenter = norm.vcenter
        left_ticks = np.linspace(vmin, vcenter, n_ticks // 2)
        right_ticks = np.linspace(vcenter, vmax, n_ticks // 2)
        ticks = ticks or [*left_ticks, *right_ticks[1:]]
        cbar.set_ticks(
            ticks,
            labels=[f"{t:.{n_decimals}f}" for t in ticks],
            fontsize=fontsize,
        )

    if plain:
        cbar.set_ticks([])

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


def get_norm(
    norm=None, vmin=None, vmax=None, midpoint=None, log=None, symlog=None
) -> Normalize:
    """Returns a normalization object for color normalization.

    Args:
        norm (Normalize, optional): A class which, when called, can normalize
            data into an interval [vmin, vmax]. Defaults to None.
        vmin (float, optional): Defaults to None.
        vmax (float, optional): Defaults to None.
        midpoint (float, optional): Midpoint value so that data is normalized
            around it. Defaults to None.
        log (bool, optional): if to normalize on a log-scale. Defaults to None.
        symlog (float, optional): normalizes to symlog with linear range
            around the range (-symlog, symlog).

    Returns:
        1. existing norm if given.
        2. else TwoSlopeNorm if vmin, vmax and midpoint is not None.
        3. else LogNorm if vmin, vmax and log is not None.
        4. else SymLogNorm if vmin, vmax and symlog is not None.
        5. else regular Normalize object if vmin and vmax is not None.
        6. else None.
    """
    if norm:
        return norm

    if all(val is not None for val in (vmin, vmax)):
        vmin -= 1e-15
        vmax += 1e-15

    if all(val is not None for val in (vmin, vmax, midpoint)):
        if vmin > midpoint or np.isclose(vmin, midpoint, atol=1e-9):
            vmin = midpoint - vmax
        if vmax < midpoint or np.isclose(vmax, midpoint, atol=1e-9):
            vmax = midpoint - vmin
        return TwoSlopeNorm(vcenter=midpoint, vmin=vmin, vmax=vmax)
    elif all(val is not None for val in (vmin, vmax, log)):
        return mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    elif all(val is not None for val in (vmin, vmax, symlog)):
        v = max(np.abs(vmin), np.abs(vmax))
        return mpl.colors.SymLogNorm(symlog, vmin=-v, vmax=v)
    elif all(val is not None for val in (vmin, vmax)):
        return mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        return None


def get_scalarmapper(
    scalarmapper=None,
    cmap=None,
    norm=None,
    vmin=None,
    vmax=None,
    midpoint=None,
    log=None,
    symlog=None,
) -> Tuple[ScalarMappable, Normalize]:
    """Returns scalarmappable with norm from `get_norm` and cmap.

    Args:
        scalarmapper (Scalarmappable, optional): for data to RGBA mapping.
            Defaults to None.
        cmap (Colormap, optional): Defaults to None.
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


def get_lims(z, offset, min=None, max=None):
    """Get scalar bounds of Ndim-array-like structure with relative offset."""

    def sub_nan(val, sub):
        if np.isnan(val):
            return sub
        else:
            return val

    if isinstance(z, (tuple, list)):
        z = list(map(lambda x: get_lims(x, offset), z))
    if isinstance(z, torch.Tensor):
        z = z.detach().cpu().numpy()
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


def width_n_height(N, aspect_ratio, max_width=None, max_height=None):
    """Integer width and height for a grid of N plots with aspect ratio."""
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
    """Create axis grid for a list of elements or integer width and height.

    Args:
        alist: list of elements to create grid for.
        gridwidth: width of grid.
        gridheight: height of grid.
        max_width: maximum width of grid.
        max_height: maximum height of grid.
        fig: optional existing figure to use.
        ax: optional existing axis to use. This ax will be divided into a grid
            of axes with the same size as the grid.
        axes: optional existing axes to use.
        aspect_ratio: aspect ratio of grid.
        figsize: figure size.
        scale: scales figure size by this factor(s) times the grid width and height.
        projection: projection of axes.
        as_matrix: return axes as matrix.
        fontsize: fontsize of axes.
        wspace: width space between axes.
        hspace: height space between axes.
        alpha: alpha of axes.
        sharex: share x axis. Only effective if a new grid of axes is created.
        sharey: share y axis. Only effective if a new grid of axes is created.
        unmask_n: number of elements to unmask. If None, all elements are unmasked.
            If provided elements at indices >= unmask_n are padded with nans.

    Returns:
        fig, axes, (gridwidth, gridheight)
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
            ax.tick_params(axis="both", which="major", labelsize=fontsize)
            ax.patch.set_alpha(alpha)

    if as_matrix:
        axes = np.array(axes).reshape(gridheight, gridwidth)

    return fig, axes, (gridwidth, gridheight)


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
    """Create a figure with the given size and spacing."""
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
    face_alpha=1.0,
    fontisze=5,
    title_pos="center",
    position=None,
    **kwargs,
):
    """Create a subplot using subplot2grid with some extra options."""
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

    ax.set_title(title, fontsize=fontisze, loc=title_pos)

    plt.xlabel(xlabel, fontsize=fontisze)
    plt.ylabel(ylabel, fontsize=fontisze)

    return ax


def divide_axis_to_grid(
    ax,
    matrix=((0, 1, 2), (3, 3, 3)),
    wspace=0.1,
    hspace=0.1,
    projection=None,
):
    """Divides an existing axis inside a figure to a grid specified by unique
        elements in a matrix.

    Args:
        ax: existing Axes object.
        matrix (array): grid matrix, where each unique element specifies
            a new axis.
        wspace (float): horizontal space between new axes.
        hspace (float): vertical space between new axes.
        projection (str): projection of new axes.


    Returns
        dict: dictionary of new axes, where keys are unique elements in
            the matrix.

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
    axes = {k: None for k in _ax_pos}
    for _ax, pos in _ax_pos.items():
        axes[_ax] = fig.add_axes(pos, projection=projection)

    return axes


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
    projection=None,
    labels=None,
):
    """Creates scattered axes in a given or new figure.

    Args:
        x (list or array): representing left coordinates of the axes.
        y (list or array): representing bottom coordinates of the axes.
        fig (None or maptlotlib.figure): Defaults to None.
        figsize (tuple or list): size of new figure. Without effect if a figure is provided. Defaults to [7, 7].
        hspace (float): spacing between axes in horizontal direction. Defaults to 0.
        wspace (float): spacing between axes in vertical direction. Defaults to 0.
        hpad (float): spacing to the horizontal borders of the figure. Defaults to 0.1.
        wpad (float): spacing to the vertical borders of the figure. Defaults to 0.1.
        alpha (float): alpha value for the white background of each ax. Defaults to 0.
        zorder (int): zorder of the white background of each ax. Defaults to 10.
        projection (str): projection of the axes. Defaults to None.
        labels (list): list of labels for each ax. Defaults to None.

    Returns:
        fig
        list of axes for each point in (x, y)
        scaled version of x and y in figure coordinates
    """
    x, y = np.array(x), np.array(y)
    assert len(x) == len(y)

    # Min-Max Scale x-Positions.
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
        axes.append(ax)

    center = []
    for ax in axes:
        _, (_center, _, _) = get_ax_positions(ax)
        center.append(_center.flatten().tolist())
    return fig, axes, center


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
    cm.get_cmap("cm_uniform_2d")
except:
    cm.register(cm_uniform_2d, name="cm_uniform_2d")
