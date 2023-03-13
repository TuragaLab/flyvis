from dataclasses import dataclass

# from dvs.utils.hex_utils import sort_u_then_v, sort_u_then_v_index
from typing import Dict, Iterable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from numbers import Number
import logging
from copy import deepcopy

logging = logging.getLogger("dvs")
import pandas as pd
import torch

from matplotlib import colormaps as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogFormatter, ScalarFormatter, NullFormatter
from matplotlib.patches import RegularPolygon
import matplotlib as mpl
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patheffects as path_effects
from matplotlib.figure import Figure
from matplotlib.axis import Axis
from matplotlib.colorbar import Colorbar

from flyvision import utils
from flyvision.plots import plt_utils

# from dvs.plots import plots
# from dvs.plots import pairplot


def plot(*args, **kwargs):
    fig, ax = plt_utils.init_plot(figsize=[1, 0.75], fontsize=5)
    ax.plot(*args, **kwargs)
    return fig, ax


# ---- CONNECTIVITY MATRIX


def heatmap_uniform(
    matrix,
    nodes,
    ax=None,
    fig=None,
    cmap=cm.get_cmap("seismic"),
    vmin=None,
    vmax=None,  # cbar_ticks=None,
    symlog=None,
    grid=True,
    cbar_label="",
    log=None,
    xlabel="Postsynaptic",
    ylabel="Presynaptic",
    cbar_height=0.5,
    cbar_width=0.01,
    cbar=True,
    title="",
    figsize=(15, 15),
    fontsize=16,
    midpoint=None,
    **kwargs,
) -> Tuple[Figure, Axis, Colorbar, np.ndarray]:
    """Plots a connection matrix.

    Args:
        matrix (array): square image.
        nodes (list): list of node types.
        cmap (Colorbar, optional): Defaults to cm.get_cmap("seismic").
        symlog (float, optional): Symmetric log scaling with
            (-symlog, symlog) linear range around zero.
        grid (bool, optional): Plots a grid. Defaults to True.
        cbar_label (str, optional): Colorbar label. Defaults to ''.
        xlabel (str, optional): Defaults to 'Postsynaptic'.
        ylabel (str, optional): Defaults to 'Presynaptic'.
        midpoint (float, optional): Midpoint normalization. Defaults to 0.

    Returns:
        fig, ax, cbar, matrix
    """
    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax=ax, fig=fig)

    norm = plt_utils.get_norm(
        symlog=symlog,
        vmin=vmin if vmin is not None else matrix.min(),
        vmax=vmax if vmin is not None else matrix.max(),
        log=log,
        midpoint=midpoint,
    )
    # plot
    im = ax.pcolormesh(matrix[::-1], cmap=cmap, norm=norm)

    cbar = (
        plt_utils.add_colorbar(
            fig,
            ax,
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

    # cosmetics
    ax.set_xticks(np.arange(len(nodes)))
    ax.set_yticks(np.arange(len(nodes)))
    ax.set_xticklabels(labels=nodes, rotation=90, fontsize=fontsize)
    ax.set_yticklabels(labels=nodes[::-1], fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    ax.grid(False, "major")
    ax.grid(True, "minor")
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    # ax.set_xlim([-0.5, max([t for t in ax.get_xticks()]) + 0.5])
    # ax.set_ylim([-0.5, max([t for t in ax.get_yticks()]) + 0.5])
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="y", which="minor", left=False)

    return fig, ax, cbar, matrix


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
):
    """
    Heatmap scatter of the matrix.

    If scale is True,
    the sizes of the scatter points will be |value| * size_scale
    or |value| * 0.005 * prod(figsize) is size_scale is not specified.

    If scale is False,
    the sizes of the scatter points will be 100.

    Todo: simplify size scale logic.

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
            else:
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
        plt_utils.add_colorbar(
            fig,
            ax,
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


# ---


def histogram_grid(
    values, labels=None, share_lims=True, nbins=None, annotate=True, fontsize=5
):
    """
    values [values1, values2, values3, ...]
    """
    dim = len(values)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = colors if dim <= len(colors) else plt_utils.get_colors(dim)
    nbins = nbins or max([len(v) for v in values])
    if share_lims:
        limits = [plt_utils.get_lims(values, 0.1) for _ in values]
    else:
        limits = [plt_utils.get_lims(v, 0.1) for v in values]

    fig, axes, _ = plt_utils.get_axis_grid(
        gridwidth=dim,
        gridheight=dim,
        figsize=[1 * dim, 1 * dim],
        as_matrix=True,
        fontsize=fontsize,
        wspace=0.3,
        hspace=0.3,
    )

    for ax in axes.flatten():
        ax.patch.set_alpha(0)
        plt_utils.rm_spines(ax, spines=("left",), rm_yticks=True)

    tril_indices = np.stack(np.tril_indices(dim, k=-1), axis=1)
    for row, column in tril_indices:
        plt_utils.rm_spines(axes[row, column])

    for i, ax in enumerate(np.diag(axes)):
        ax.hist(
            values[i],
            bins=np.linspace(*limits[i], nbins),
            linewidth=0.5,
            fill=False,
            histtype="step",
            density=True,
            color=colors[i],
        )

        if labels is not None:
            ax.set_xlabel(labels[i], fontsize=fontsize)

        if annotate:
            ax.annotate(
                f"min:{values[i].min():.3G}\nmedian:{np.median(values[i]):.3G}\nmean:{values[i].mean():.3G}\nstd:{values[i].std():.2G}",
                xy=(1, 1),
                ha="right",
                va="top",
                xycoords="axes fraction",
                fontsize=fontsize - 1,
            )

    triu_indices = np.stack(np.triu_indices(dim, k=1), axis=1)
    for row, column in triu_indices:
        axes[row, column].hist(
            [values[row], values[column]],
            bins=np.linspace(*limits[i], nbins),
            linewidth=0.5,
            fill=False,
            histtype="step",
            density=True,
            color=[colors[row], colors[column]],
        )
    return fig, axes


# ---- KERNEL


class SignError(Exception):
    pass


def kernel(
    u,
    v,
    color,
    fontsize=10,
    cbar=True,
    edgecolor="k",
    fig=None,
    ax=None,
    figsize=(10, 10),
    midpoint=0,
    annotate=True,
    alpha=0.8,
    annotate_coords=False,
    coord_fs=8,
    cbar_height=0.3,
    cbar_x_offset=-1,
    **kwargs,
):
    """Wrapper of hex_scatter for plotting projective/receptive
       fields with good defaults.

    Args:
        u (array): array of hex coordinates in u direction.
        v (array): array of hex coordinates in v direction.
        color (array): array of pixel values per point (u_i, v_i).
        cbar (bool, optional): plots a colorbar. Defaults to True.
        edgecolor (str, optional): color of the edges. Defaults to "k".
        annotate (bool, optional): annotates a rounded value. Defaults to True.
        alpha (float, optional): alpha of the hexagon faces. Defaults to 0.8.
        annotate_coords (bool, optional): annotates (u_i, v_i).
            Defaults to True.
        coord_fs (int, optional): fontsize of the coord annotation.
            Defaults to 8.

    Returns:
        fig, ax, (label_text, scalarmapper)
    """
    sign = set(np.sign(color[np.nonzero(color)]))
    if len(sign) == 1:
        sign = sign.pop()
    elif len(sign) == 0:
        sign = 1
    else:
        raise SignError(f"inconsistent kernel with signs {sign}")
    cmap = cm.get_cmap("Blues_r") if sign < 0 else cm.get_cmap("Reds")
    # scalarmapper, norm = plt_utils.get_scalarmapper(vmin=kwargs['vmin'],
    # vmax=kwargs['vmax'],
    # midpoint=midpoint,
    # symlog=True)
    cmap = cm.get_cmap("seismic")
    _kwargs = locals()
    _kwargs.update(_kwargs["kwargs"])
    _kwargs.pop("kwargs")
    return hex_scatter(**_kwargs)


# ---- PARAMETER HISTOGRAM


def param_hist(
    params,
    fig=None,
    ax=None,
    figsize=(15, 5),
    bins=100,
    fontsize=18,
    title="",
    yscale="log",
    xlabel="",
    ylabel="",
):
    """Plots a histrogram.

    Args:
        params (array): contains values to bin.
        bins (int, optional): number of bins. Defaults to 100.
        yscale (str, optional): "linear", "log", "symlog", "logit".
            Defaults to "log".
        xlabel (str, optional): Defaults to "".
        ylabel (str, optional): Defaults to "".

    Returns:
        fig, ax
    """

    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax=ax, fig=fig)

    ax.hist(params, bins)

    if yscale:
        ax.set_yscale(yscale)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)

    ax.tick_params(labelsize=fontsize)

    return fig, ax


# ---- NETWORK GRAPH


def _network_graph_node_pos(layout, region_spacing=2):
    # one way to compute (x,y) coordinates for nodes
    x_coordinate = 0
    types_per_column = 8
    region_0 = "retina"
    pos = {}
    j = 0
    for typ in layout:
        if layout[typ] != region_0:
            x_coordinate += region_spacing
            j = 0
        elif (j % types_per_column) == 0 and j != 0:
            x_coordinate += 1
        y_coordinate = types_per_column - 1 - j % types_per_column
        pos[typ] = [x_coordinate, y_coordinate]
        region_0 = layout[typ]
        j += 1
    return pos


def _network_graph(nodes, edges):
    """Transform graph representation from df to list to create a networkx.Graph object."""
    nodes = nodes.groupby(by=["type"], sort=False, as_index=False).first().type
    edges = list(
        map(
            lambda x: x.split(","),
            (edges.source_type + "," + edges.target_type).unique(),
        )
    )
    return nodes, edges


def network_graph(
    nodes,
    edges,
    layout,
    node_color=None,
    edge_color="k",
    fig=None,
    ax=None,
    alpha=1,
    figsize=(11.69, 8.27),
    radius=1100,
    marker="H",
    vmin=None,
    vmax=None,
    cbar=True,
    region_spacing=2.0,
    node_cbar_title="",
    edge_cbar_title="",
    node_cbar_x_offset=0,
    node_cbar_y_offset=0,
    title="",
    fontsize=10,
    logcolor=None,
    node_cmap=cm.get_cmap("Reds"),
    edge_cmap=None,
):
    """Network graph.

    Args:
        nodes (array): a list of node types.
        edges (array): a list of edge tuples [(source, target), ...].
        node_color (array, optional): values for coloring the nodes.
            Defaults to None.
        radius (int, optional): radius of the nodes. Defaults to 1100.
        cbar_title (str, optional): colorbar label. Defaults to "".
        cmap (Colormap, optional): colormap. Defaults to plt.cm.GnBu.

    Returns:
        fig, ax
    """
    # to save time at library import
    import networkx as nx
    from matplotlib.patches import Circle

    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax=ax, fig=fig)
    # node_cmap = plt_utils.cmap_map(lambda x: x/2 + 0.5, node_cmap)

    if isinstance(nodes, pd.DataFrame) and isinstance(edges, pd.DataFrame):
        nodes, edges = _network_graph(nodes, edges)

    # Create graph object.
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    pos = _network_graph_node_pos(layout, region_spacing=region_spacing)

    # Plot the graph.
    node_vmin, node_vmax = None, None
    if isinstance(node_color, (list, np.ndarray)):
        node_color = np.array(node_color)
        node_vmin = (
            vmin
            if vmin is not None
            else (-np.max(node_color) if np.any(node_color < 0) else 0.0)
        )
        node_vmax = vmax if vmax is not None else np.max(node_color)
        node_sm, node_norm = plt_utils.get_scalarmapper(
            cmap=node_cmap, vmin=node_vmin, vmax=node_vmax, log=logcolor
        )
        if cbar:
            node_cbar = plt_utils.add_colorbar(
                fig,
                ax,
                pos="right",
                width=0.01,
                height=0.5,
                x_offset=node_cbar_x_offset,
                y_offset=node_cbar_y_offset,
                cmap=node_cmap,
                fontsize=10,
                tick_length=2,
                tick_width=1,
                rm_outline=True,
                norm=node_norm,
                label=node_cbar_title,
                plain=False,
            )

    # ------- DRAW NODES
    # node_collection = nx.draw_networkx_nodes(graph, pos=pos, node_size=radius,
    #                                  vmin=node_vmin,
    #                                  vmax=node_vmax,
    #                                  node_color=node_color,
    #                                  cmap=node_cmap,
    #                                  alpha=alpha,
    #                                  node_shape=marker,
    #                                  ax=ax)
    xy = np.asarray([pos[v] for v in list(graph)])
    node_collection = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=radius,
        c=node_color,
        marker=marker,
        alpha=alpha,
        cmap=node_cmap,
        norm=node_norm,
    )
    node_collection.set_zorder(2)
    # node_collection.set_edgecolor('black')

    # ------- LABEL NODES
    node_face_colors = node_sm.to_rgba(node_color)
    for i, n in enumerate(nodes):
        facecolor = node_face_colors[i]
        (x, y) = pos[n]
        _textcolor = (
            "black" if mpl.colors.rgb_to_hsv(facecolor[:-1])[-1] > 0.95 else "white"
        )
        t = ax.text(
            x,
            y,
            n,
            size=fontsize,
            color=_textcolor,
            family="sans-serif",
            weight="normal",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transData,
            clip_on=True,
        )
    # nx.draw_networkx_labels(graph, pos=pos, ax=ax, font_size=fontsize)

    edge_vmin, edge_vmax = None, None
    if isinstance(edge_color, np.ndarray):
        edge_vmin = -np.max(edge_color) if np.any(edge_color < 0) else 0
        edge_vmax = np.max(edge_color)
        edge_sm, edge_norm = plt_utils.get_scalarmapper(
            cmap=edge_cmap, vmin=edge_vmin, vmax=edge_vmax
        )
        if cbar:
            edge_cbar = plt_utils.add_colorbar(
                fig,
                ax,
                pos="left",
                width=0.01,
                height=0.5,
                x_offset=0,
                cmap=edge_cmap,
                fontsize=10,
                tick_length=2,
                tick_width=1,
                rm_outline=True,
                norm=edge_norm,
                label=edge_cbar_title,
                plain=False,
            )
    nx.draw_networkx_edges(
        graph,
        pos=pos,
        ax=ax,
        edge_color=edge_color or None,
        edge_cmap=edge_cmap or None,
        edge_vmin=edge_vmin or None,
        edge_vmax=edge_vmax or None,
        alpha=0.5,
    )

    # Hacky way to add reccurent connections.
    for a, b in graph.edges:
        if a == b:
            x, y = pos[a]
            x += 0.25
            y += 0.25
            ax.add_artist(
                Circle(
                    [x, y],
                    radius=0.3,
                    facecolor="None",
                    edgecolor="black",
                    alpha=0.5,
                )
            )

    ax = plt_utils.rm_spines(ax, rm_xticks=True, rm_yticks=True)
    ax.tick_params(labelsize=fontsize)
    return fig, ax


def network_layout_axes(
    layout,
    node_types=None,
    fig=None,
    figsize=[16, 10],
    region_spacing=2,
    wspace=0,
    hspace=0,
    as_dict=False,
):
    fig = fig or plt.figure(figsize=figsize)

    pos = _network_graph_node_pos(layout, region_spacing=region_spacing)
    pos = {
        key: value
        for key, value in pos.items()
        if (node_types is None or key in node_types)
    }
    xy = np.array(list(pos.values()))
    # why pad this?
    # hpad = 0.05
    # wpad = 0.05
    hpad = 0.0
    wpad = 0.0
    fig, axes, xy_scaled = plt_utils.regular_ax_scatter(
        xy[:, 0],
        xy[:, 1],
        fig=fig,
        wspace=wspace,
        hspace=hspace,
        hpad=hpad,
        wpad=wpad,
        alpha=0,
        labels=list(pos.keys()),
    )
    new_pos = {key: xy_scaled[i] for i, key in enumerate(pos.keys())}
    if as_dict:
        return (
            fig,
            {node_type: axes[i] for i, node_type in enumerate(new_pos)},
            new_pos,
        )
    return fig, axes, new_pos


def _network_graph_ax_scatter(
    node_types,
    nodes,
    edges,
    layout,
    figsize=[16, 10],
    fig=None,
    edge_color_key=None,
    arrows=False,
    edge_alpha=0.2,
    edge_width=1.0,
    constant_edge_width=False,
    constant_edge_color=None,
    edge_cmap=cm.get_cmap("seismic"),
    region_spacing=2,
    wspace=0,
    hspace=0,
    self_loop_radius=0.02,
    self_loop_width=None,
    self_loop_edge_alpha=None,
    **kwargs,
):
    """Get an ex scatter with the underlying network graph.

    Minimal working example of the logic to connect arbitrarily positioned axes
    with edges using networkx:
    >>> # create a graph of nodes and edges with positions x, y
    >>> nodes = ["A", "B"]
    >>> edges = [("A", "A"), ("A", "B"), ("B", "A"), ("B", "B")]
    >>> x, y = [0, 1], [5.3, 3.4]
    >>> graph = nx.Graph()
    >>> graph.add_nodes_from(nodes)
    >>> graph.add_edges_from(edges)

    # scatter one ax per node regularly across a figure
    >>> fig, axes, positions = dvs.plots.regular_ax_scatter(x, y, figsize=[7, 3])
    # create another ax in the background spanning the entire figure
    >>> (lefts, bottoms, rights, tops), (centers, widths, height) = dvs.plots.get_ax_positions(axes)
    >>> edge_ax = fig.add_axes([lefts.min(), bottoms.min(), rights.max() - lefts.min(), tops.max() - bottoms.min()])
    >>> edge_ax.set_ylim(0, 1)
    >>> edge_ax.set_xlim(0, 1)
    # compute the positions of the centers of the node axes in the ax coordinates
    # spanning the entire figure
    >>> fig_to_edge_ax = fig.transFigure + edge_ax.transData.inverted()
    >>> positions = {"A": fig_to_edge_ax.transform(positions[0]),
                     "B": fig_to_edge_ax.transform(positions[1])}

    # remove self loops to add them with a custom radius at the end
    >>> self_loops = list(nx.selfloop_edges(graph))
    >>> graph.remove_edges_from(self_loops)

    # draw the edges on the ax spanning the entire figure to connect the subaxes
    >>> nx.draw_networkx_edges(
            graph,
            pos=positions,
            ax=edge_ax)

    # add self loops with custom radius
    >>> for a, b in self_loops:
    >>>     self_loop_radius = 0.04
    >>>     x, y = positions[a]
    >>>     y += self_loop_radius
    >>>     edge_ax.add_artist(
                Circle(
                    [x, y],
                    radius=self_loop_radius,
                    facecolor="None",
                    edgecolor="b",
                    alpha=1.0
                    )
                )
    """
    # TODO: add node and edge color options and anatomy labels at the bottom.
    # to save time at library import
    import networkx as nx
    from matplotlib.patches import Circle

    fig, axes, positions = network_layout_axes(
        layout,
        node_types,
        fig,
        figsize,
        region_spacing,
        wspace=wspace,
        hspace=hspace,
    )
    (lefts, bottoms, rights, tops), (
        centers,
        widths,
        height,
    ) = plt_utils.get_ax_positions(axes)
    edge_ax = fig.add_axes(
        [
            lefts.min(),
            bottoms.min(),
            rights.max() - lefts.min(),
            tops.max() - bottoms.min(),
        ]
    )
    edge_ax.set_zorder(0)
    edge_ax = plt_utils.rm_spines(edge_ax, rm_xticks=True, rm_yticks=True)
    edge_ax.patch.set_alpha(0.0)
    edge_ax.set_ylim(0, 1)
    edge_ax.set_xlim(0, 1)

    fig_to_edge_ax = fig.transFigure + edge_ax.transData.inverted()
    positions = {
        key: fig_to_edge_ax.transform(value) for key, value in positions.items()
    }
    # fig.positions = positions

    assert set(layout) == set(node_types), "Subset of node types not supported."
    # nodes = utils.filter_df_by_list(node_types,
    #                                         nodes,
    #                                         column="type")
    # edges = utils.filter_df_by_list(node_types,
    #                                         edges,
    #                                         column="source_type")
    # edges = utils.filter_df_by_list(node_types,
    #                                         edges,
    #                                         column="target_type")

    nodes, edge_list = _network_graph(nodes, edges)

    # def _get_ax(fig, label):
    #     for ax in fig.axes:
    #         if ax.get_label() == label:
    #             return ax

    # positions = {
    #     node_type: plt_utils.get_ax_positions(_get_ax(fig, node_type))[1][
    #         0
    #     ].flatten()
    #     for node_type in nodes
    # }

    if edge_color_key is not None:
        grouped = edges.groupby(
            by=["source_type", "target_type"], sort=False, as_index=False
        ).mean()
        edge_color = {
            (row.source_type, row.target_type): row.sign
            for i, row in grouped.iterrows()
        }
        _edge_color = np.array(list(edge_color.values()))
        edge_vmin = -np.max(_edge_color) if np.any(_edge_color < 0) else 0
        edge_vmax = np.max(_edge_color)
        edge_sm, _ = plt_utils.get_scalarmapper(
            cmap=edge_cmap, vmin=edge_vmin, vmax=edge_vmax, midpoint=0.0
        )
    else:
        constant_edge_color = constant_edge_color or "k"
        edge_color = {tuple(edge): constant_edge_color for edge in edge_list}
        edge_vmin = None
        edge_vmax = None
        edge_sm = None

    grouped = edges.groupby(
        by=["source_type", "target_type"], sort=False, as_index=False
    ).mean()

    if constant_edge_width is None:
        edge_width = {
            (row.source_type, row.target_type): edge_width * (np.log(row.n_syn) + 1)
            for i, row in grouped.iterrows()
        }
    else:
        edge_width = {
            (row.source_type, row.target_type): constant_edge_width
            for i, row in grouped.iterrows()
        }

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edge_list)
    self_loops = list(nx.selfloop_edges(graph))
    graph.remove_edges_from(self_loops)
    nx.draw_networkx_edges(
        graph,
        pos=positions,
        ax=edge_ax,
        edge_color=np.array([edge_color[tuple(edge)] for edge in edge_list]),
        edge_cmap=edge_cmap,
        edge_vmin=edge_vmin,
        edge_vmax=edge_vmax,
        alpha=edge_alpha,
        arrows=arrows,
        width=np.array([edge_width[tuple(edge)] for edge in edge_list]),
    )
    # edge_sm, edge_norm = plt_utils.get_scalarmapper(cmap=edge_cmap, vmin=edge_vmin, vmax=edge_vmax)
    # _ = nx.draw_networkx_edges(graph, pos=positions, ax=edge_ax, alpha=0.1)

    # hacky way to add self-loops of custom size
    for a, b in self_loops:
        x, y = positions[a]
        y += self_loop_radius

        if edge_sm is not None:
            color = edge_sm.to_rgba(edge_color[(a, b)])
        else:
            color = edge_color[(a, b)]

        edge_ax.add_artist(
            Circle(
                [x, y],
                radius=self_loop_radius,
                facecolor="None",
                edgecolor=color,
                alpha=self_loop_edge_alpha or edge_alpha,
                linewidth=self_loop_width or edge_width[(a, b)],
            )
        )

    return fig, axes


# ---- HEX SCATTER


def hex_cs(extent=5, mode="tschopp", annotate_coords=True, edgecolor="black", **kwargs):
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
    color = utils.tensor_utils.to_numpy(color.squeeze())
    u, v = utils.hex_utils.get_hex_coords(utils.hex_utils.get_hextent(len(color)))
    return hex_scatter(u, v, color, cmap=cmap, **kwargs)


def hex_scatter(
    u,
    v,
    color,
    max_extent=None,
    fig=None,
    ax=None,
    figsize=(1, 1),
    fontcolor="black",
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
    mode="tschopp",
    orientation=np.radians(30),
    cmap=cm.get_cmap("seismic"),
    cbar=True,
    cbar_label="",
    cbar_height=None,
    cbar_width=None,
    cbar_x_offset=None,
    annotate=False,
    annotate_coords=False,
    annotate_indices=False,
    coord_fs=8,
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
        fontcolor (str, optional): color of annotation. Defaults to "black".
            Currently unused.
        label (str, optional): a label positioned per default in the upper left
            corner. Defaults to "".
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
        coord_fs (int, optional): fontsize of the coord annotation.
            Defaults to 8.

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

    # orientation = np.radians(0) if mode != 'flat' else np.radians(30)

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
        cbar = plt_utils.add_colorbar(
            fig,
            ax,
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
            #         f"NAN",
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


def hex_scatter_grid(hex_arrays, cmap="binary_r", cbar=False, **kwargs):
    """(n_frames, n_hexals)"""
    cmap = cm.get_cmap(cmap)
    N, n_hexals = hex_arrays.shape
    fig, axes, _ = plt_utils.get_axis_grid(list(range(N)))
    u, v = utils.get_hex_coords(utils.get_hextent(n_hexals))
    for n in range(N):
        hex_scatter(
            u,
            v,
            hex_arrays[n],
            fig=fig,
            ax=axes[n],
            cmap=cmap,
            cbar=cbar,
            **kwargs,
        )
    return fig, axes


def quick_hex_flow(flow, **kwargs):
    flow = utils.to_numpy(flow.squeeze())
    u, v = utils.get_hex_coords(utils.get_hextent(flow.shape[-1]))
    return hex_flow(u, v, flow, **kwargs)


def hex_flow_grid(hex_arrays, cmap=None, cwheel=False, **kwargs):
    """(n_frames, 2, n_hexals)"""
    if cmap is None:
        cmap = plt_utils.cm_uniform_2d
    N, _, n_hexals = hex_arrays.shape
    fig, axes, _ = plt_utils.get_axis_grid(list(range(N)))
    u, v = utils.get_hex_coords(utils.get_hextent(n_hexals))
    for n in range(N):
        hex_flow(
            u,
            v,
            hex_arrays[n],
            fig=fig,
            ax=axes[n],
            cmap=cmap,
            cwheel=cwheel,
            **kwargs,
        )
    return fig, axes


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
    mode="tschopp",
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
    """Plots a hexagonal lattice with coordinates u, v, and coloring color."""

    from matplotlib.patches import RegularPolygon

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
    x, y = utils.hex_to_pixel(u, v, mode=mode)
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

    extent = utils.get_extent(u, v)
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


def flow_to_rgba(flow):
    """(2, h, w)"""
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
    rgba = flow_to_rgba(flow)
    plt.imshow(rgba)
    plt.show()


def plot_flow_grid(flow, flow_est):
    def flow_to_rgba(flow):
        """(n, 2, h, w)"""
        if isinstance(flow, torch.Tensor):
            flow = flow.detach().cpu().numpy()

        n, _, h, w = flow.shape

        X, Y = flow[:, 0], flow[:, 1]
        R = np.sqrt(X * X + Y * Y)
        PHI = np.arctan2(Y, X)  # + np.pi
        scalarmapper, norm = plt_utils.get_scalarmapper(
            cmap=plt_utils.cm_uniform_2d, vmin=-np.pi, vmax=np.pi
        )
        rgba = np.concatenate(list(map(scalarmapper.to_rgba, PHI)), axis=0).reshape(
            n, h, w, 4
        )
        rgba[:, :, :, -1] = R / R.max()
        return rgba

    n_samples = flow.shape[0]

    flow_rgba = flow_to_rgba(flow)
    flow_est_rgba = flow_to_rgba(flow_est)

    fig, axes, _ = plt_utils.get_axis_grid(
        gridheight=n_samples, gridwidth=2, as_matrix=True
    )

    for sample in range(n_samples):
        flow_sample = flow_rgba[sample]
        flow_est_sample = flow_est_rgba[sample]
        axes[sample, 0].imshow(flow_sample)
        axes[sample, 1].imshow(flow_est_sample)

    return fig, axes


def init_3d(
    label=["", "", ""],
    elev=0,
    azim=0,
    figsize=[7, 7],
    title="",
    fontsize=10,
    fig=None,
    ax=None,
):
    fig, ax = plt_utils.init_plot(
        figsize=figsize,
        title=title,
        fontsize=fontsize,
        ax=ax,
        fig=fig,
        projection="3d",
        offset=None,
    )
    ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    plt.locator_params(nbins=2)

    ax.xaxis._axinfo["juggled"] = (0, 0, 0)
    ax.yaxis._axinfo["juggled"] = (1, 1, 1)
    ax.zaxis._axinfo["juggled"] = (2, 2, 2)

    ax.w_xaxis.set_pane_color(
        (0.9607843137254902, 0.9607843137254902, 0.9607843137254902, 1.0)
    )
    ax.w_yaxis.set_pane_color(
        (0.9607843137254902, 0.9607843137254902, 0.9607843137254902, 1.0)
    )
    ax.w_zaxis.set_pane_color(
        (0.9607843137254902, 0.9607843137254902, 0.9607843137254902, 1.0)
    )

    ax.w_xaxis.line.set_color((0.0, 0.0, 0.0, 0.5))
    ax.w_yaxis.line.set_color((0.0, 0.0, 0.0, 0.5))
    ax.w_zaxis.line.set_color((0.0, 0.0, 0.0, 0.5))

    ax.set_xlabel(label[0], fontsize=fontsize)
    ax.set_ylabel(label[1], fontsize=fontsize)
    ax.set_zlabel(label[2], fontsize=fontsize)
    ax.view_init(elev, azim)
    return fig, ax


def trajectory_3d(
    x,
    y,
    z,
    ax_label=["x", "y", "z"],
    elev=0,
    azim=0,
    figsize=[7, 7],
    title="",
    fontsize=10,
    fig=None,
    ax=None,
):
    fig, ax = init_3d(ax_label, elev, azim, figsize, title, fontsize, fig, ax)
    ax.plot(x, y, z)
    return fig, ax


# ---- (ACTIVITY) BARS


def activity_bars(
    nodes,
    activity,
    figsize=(15, 5),
    title="Activity",
    ylabel=False,
    fontsize=10,
    ax=None,
    fig=None,
    label="",
    labelxy=(0, 1.01),
    **kwargs,
):
    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax, fig)

    activity = np.ma.masked_invalid(activity)

    ax.bar(x=nodes, height=activity.data, color="#1f77b4", **kwargs)

    # ax.set_yticks([])
    xmin, xmax = -0.5, len(nodes)
    ax.set_xlim((xmin, xmax))
    ax.set_ylim(*plt_utils.get_lims(activity, 0.1))
    ax = plt_utils.rm_spines(ax, spines=["top", "right"])
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.set_xticks(range(len(nodes)))
    ax.set_xticklabels(nodes, rotation=90)
    if ylabel:
        ax.set_ylabel(title, fontsize=fontsize)
    else:
        ax.set_title(title, fontsize=fontsize)

    for i, inv in enumerate(activity.mask):
        text = "NaN" if inv else ""
        ax.text(
            (i + np.abs(xmin)) / (xmax + np.abs(xmin)),
            0.1,
            text,
            ha="center",
            va="center",
            rotation=90,
            fontsize=fontsize,
            transform=ax.transAxes,
        )

    label_text = ax.text(
        labelxy[0],
        labelxy[1],
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
    )

    return fig, ax, label_text


def bars(
    xticklabels,
    values,
    std=None,
    ecolor="#babab9",
    elinewidth=2,
    capsize=4,
    colors=[],
    figsize=(15, 5),
    title="",
    ylabel=None,
    ylim=None,
    rotation=90,
    ha=None,
    fontsize=10,
    ax=None,
    fig=None,
    legend=False,
    grid=True,
    label="",
    labelxy=(0, 1.01),
    legend_kwargs={},
    **kwargs,
):

    xticklabels = np.array(xticklabels).astype(str)

    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax, fig)
    if grid:
        ax.yaxis.grid()

    values = np.ma.masked_invalid(values)

    handles = ax.bar(
        x=xticklabels, height=values.data, color=colors or "#1f77b4", **kwargs
    )
    if std is not None:
        ax.errorbar(
            x=xticklabels,
            y=values.data,
            yerr=std,
            c=ecolor,
            fmt="none",
            capsize=capsize,
            elinewidth=elinewidth,
        )

    xmin, xmax = -0.5, len(xticklabels) + 0.5
    # ax.set_xlim((xmin, xmax))

    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(*plt_utils.get_lims(values, 0.1))

    ax = plt_utils.rm_spines(
        ax, spines=["top", "right"], rm_yticks=False, rm_xticks=False
    )
    ax.tick_params(axis="both", which="major", labelsize=fontsize)

    [t.set_rotation(rotation) for t in ax.get_xticklabels()]
    if rotation == 90:
        [t.set_ha("center") for t in ax.get_xticklabels()]

    ax.set_ylabel(ylabel or "", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    for i, inv in enumerate(values.mask):
        text = "NaN" if inv else ""
        ax.text(
            (i + np.abs(xmin)) / (xmax + np.abs(xmin)),
            0.1,
            text,
            ha="center",
            va="center",
            rotation=90,
            fontsize=fontsize,
            transform=ax.transAxes,
        )

    label_text = ax.text(
        labelxy[0],
        labelxy[1],
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
    )

    if legend:
        ax.legend(handles, xticklabels, **legend_kwargs)

    return fig, ax, label_text


def colored_bars(
    labels,
    values,
    sort=None,
    title="",
    fig=None,
    ax=None,
    midpoint=None,
    vmin=None,
    vmax=None,
    cbar=True,
    cbar_label="",
    figsize=[3, 3],
    orientation="vertical",
    fontsize=8,
    cmap=cm.get_cmap("seismic"),
    tick_label_rotation=0,
):
    """Vertical or horizontal colored bar plot.

    Args:
        labels (list): axis labels.
        values (list): values.
        sort (str, optional): 'asc' or 'desc'. Defaults to None.
        title (str, optional): Defaults to "".
        fig (object, optional): Defaults to None.
        ax (object, optional): Defaults to None.
        midpoint (float, optional): color midpoint. Defaults to 0.
        vmin (float, optional): color minimum. Defaults to 0.
        vmax (float, optional): color maximum. Defaults to 0.
        cbar (bool, optional): Defaults to True.
        figsize (list, optional): Defaults to [3, 3].
        orientation (str, optional): 'vertical or horizontal'. Defaults to 'vertical'.
        fontsize (int, optional): Defaults to 8.
        cmap (object, optional): Defaults to cm.get_cmap("seismic").
        tick_label_rotation (int, optional): Defaults to 0.

    Returns:
        tuple: (fig, ax)
    """
    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax, fig)
    values, labels = np.asarray(values).astype(float), np.asarray(labels)
    if sort == "asc":
        index = np.argsort(values)[::-1]
    elif sort == "desc":
        index = np.argsort(values)
    else:
        index = slice(None)
    values, labels = values[index], labels[index]

    scalarmapper, norm = plt_utils.get_scalarmapper(
        cmap=cmap,
        vmin=vmin or values.min(),
        vmax=vmax or values.max(),
        midpoint=midpoint,
    )
    color_rgba = scalarmapper.to_rgba(values)

    if orientation == "vertical":
        ax.bar(labels, values, color=color_rgba)
        ax.yaxis.grid(alpha=0.5)
        ax.spines["right"].set_visible(True)
        ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=True)
        ax.set_xticks(labels)
        ax.set_xticklabels(labels, rotation=tick_label_rotation)
        if cbar:
            cbar = plt_utils.add_colorbar(
                fig,
                ax,
                pos="top",
                width=0.2,
                height=0.01,
                x_offset=-0.5,
                cmap=cmap,
                norm=norm,
                label=cbar_label,
            )
    elif orientation == "horizontal":
        ax.barh(labels, values, color=color_rgba)
        ax.xaxis.grid(alpha=0.5)
        ax.spines["top"].set_visible(True)
        ax.tick_params(axis="x", top=True, bottom=True, labeltop=True, labelbottom=True)
        ax.set_yticks(labels)
        ax.set_yticklabels(labels, rotation=tick_label_rotation)
        if cbar:
            cbar = plt_utils.add_colorbar(
                fig,
                ax,
                pos="right",
                width=0.01,
                x_offset=2,
                cmap=cmap,
                norm=norm,
                label=cbar_label,
            )

    return fig, ax


def violins(
    data,
    rotation=90,
    scatter=False,
    cmap=plt.cm.Blues_r,
    colors=None,
    fontsize=5,
    figsize=[10, 1],
    width=0.7,
    scatter_edge_color="white",
    scatter_radius=5,
    scatter_edge_width=0.5,
    showmeans=False,
    showmedians=True,
    grid=False,
    violin_alpha=1,
    cstart=50,
    **kwargs,
):
    """
    Convenience function for violin_groups with good defaults.

    data (Array): (<#groups>, #samples, #random variables) first dimension
                        optional because mostly we have only one group.
                        Will be extended to (1, #samples, #random variables),
                        transposed, and passed to violin_groups.
    """
    if len(data.shape) == 3:
        pass
    elif len(data.shape) == 2:
        data = data[None]
    else:
        raise ValueError

    data = np.transpose(data, axes=(2, 0, 1))

    kwargs.update(vars())
    return violin_groups(data, **kwargs)


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


def display_significance_value(
    ax,
    pvalue,
    y,
    x0=None,
    x1=None,
    ticklabel=None,
    bar_width=0.7,
    pthresholds={0.01: "***", 0.05: "**", 0.1: "*"},
    fontsize=8,
    annotate_insignificant="",
    append_tick=False,
    show_bar=False,
    other_ax=None,
    bar_height_ylim_ratio=0.01,
    linewidth=0.5,
    annotate_pthresholds=True,
    loc_pthresh_annotation=(0.1, 0.1),
    location="above",
    asterisk_offset=None,
):
    """Display a significance value annotation along x at height y.

    Args:
        ax (Axis)
        pvalue (float)
        y (float)
        x0 (float, optional): left edge of bar if show_bar is True.
        x1 (float, optional): right edge of bar if show_bar is True.
        ticklabel (str, optional): ticklabel to
        bar_width (float, optional)
        pthresholds (Dict[float, str]): annotations
        xpvalues Dict[str, float]: x-ticklabel to pvalue mapping
        y: height to put text
        pthreshold: different thresholds for different annotations

    """

    if x0 is None and x1 is None and ticklabel is None and bar_width is None:
        raise ValueError("specify (x0, x1) or (ticklabel, bar_width)")

    if show_bar and ((x0 is None or x1 is None) and bar_width is None):
        raise ValueError("need to specify width of bar or specify x0 and x1")

    if location == "above":
        va = "bottom"
        asterisk_offset = asterisk_offset or -0.1
    elif location == "below":
        va = "top"
        asterisk_offset = asterisk_offset or -0.05
    else:
        raise ValueError(f"location {location}")

    if x0 is None and x1 is None and ticklabel is not None:
        ticklabels = ax.get_xticklabels()
        if not ticklabels:
            ticklabels = other_ax.get_xticklabels()
        if not ticklabels:
            raise AssertionError("no ticklables found")
        # get the tick to get the x position for the annotation
        tick = [tick for tick in ticklabels if tick.get_text() == ticklabel][0]
        x, _ = tick.get_position()
        x0 = x - bar_width / 2
        x1 = x + bar_width / 2

    text = ""
    any_thresh = False
    less = []
    for thresh, symbol in pthresholds.items():
        if pvalue < thresh:
            less.append(thresh)
            any_thresh = True

    if (any_thresh or annotate_insignificant) and show_bar:
        bar_height = (ax.get_ylim()[1] - ax.get_ylim()[0]) * bar_height_ylim_ratio
        bar_x = [x0, x0, x1, x1]
        if location == "above":
            bar_y = [y, y + bar_height, y + bar_height, y]
            y = y + bar_height
            mid = ((x0 + x1) / 2, y)
        elif location == "below":
            bar_y = [y, y - bar_height, y - bar_height, y]
            y = y - bar_height
            mid = ((x0 + x1) / 2, y)
        ax.plot(bar_x, bar_y, c="k", lw=linewidth)
        x = mid[0]

    if any_thresh:
        text = pthresholds[min(less)]
        if ticklabel is not None and append_tick:
            tick.set_text(f"{tick.get_text()}$^{{{text}}}$")
            ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels())
        else:
            ax.annotate(
                text,
                (x, y + asterisk_offset),
                fontsize=fontsize,
                ha="center",
                va=va,
            )

    elif annotate_insignificant:
        if ticklabel is not None and append_tick:
            tick.set_text(f"{tick.get_text()}$^{{{annotate_insignificant}}}$")
            ax.xaxis.set_ticklabels(ax.xaxis.get_ticklabels())
        else:
            ax.annotate(
                annotate_insignificant,
                (x, y),
                fontsize=fontsize,
                ha="center",
                va=va,
            )

    if annotate_pthresholds:
        pthreshold_annotation = ""
        for i, (thresh, symbol) in enumerate(pthresholds.items()):
            pthreshold_annotation += f"{symbol}p<{thresh:.2f}"
            if i != len(pthresholds) - 1:
                pthreshold_annotation += "\n"

        ax.annotate(
            pthreshold_annotation,
            loc_pthresh_annotation,
            xycoords="axes fraction",
            fontsize=fontsize,
            va="bottom",
            ha="left",
        )


def display_pvalues(
    ax,
    pvalues: Dict[str, float],
    ticklabels: List[str],
    data: np.ndarray,
    location="above",
    bar_width=0.7,
    show_bar=True,
    bar_height_ylim_ratio=0.01,
    fontsize=6,
    annotate_insignificant="ns",
    loc_pthresh_annotation=(0.01, 0.01),
    append_tick=False,
    data_relative_offset=0.05,
    asterisk_offset=0,
    pthresholds={0.01: "***", 0.05: "**", 0.1: "*"},
):
    """Annotate all pvalues from Dict[xticklabel, pvalue].

    data is Array[random variables, ...]
    """

    for key in pvalues:
        if key not in ticklabels:
            raise ValueError(f"pvalue key {key} is not a ticklabel")

    offset = data_relative_offset * np.abs(data.max() - data.min())

    ylim = ax.get_ylim()
    bars = []
    for ticklabel, pvalue in pvalues.items():
        index = [
            i for i, _ticklabel in enumerate(ticklabels) if _ticklabel == ticklabel
        ][0]
        _values = data[index]

        if location == "above":
            _max = _values.max()
            y = min(_max + offset, ylim[1])
        elif location == "below":
            _min = _values.min()
            y = max(_min - offset, ylim[0])

        # print(y)

        display_significance_value(
            ax,
            pvalue,
            y=y,
            ticklabel=str(ticklabel),
            bar_width=bar_width,
            show_bar=show_bar,
            bar_height_ylim_ratio=bar_height_ylim_ratio,
            fontsize=fontsize,
            annotate_insignificant=annotate_insignificant,
            loc_pthresh_annotation=loc_pthresh_annotation,
            append_tick=append_tick,
            location=location,
            asterisk_offset=asterisk_offset,
            pthresholds=pthresholds,
        )
        bars.append(y)

    ax.set_ylim(*plt_utils.get_lims([bars, ylim], 0.01))


def violin_groups(
    values,
    xticklabels=None,
    pvalues=None,
    display_pvalues_kwargs={},
    legend=False,
    legend_kwargs={},
    as_bars=False,
    colors=None,
    cmap=cm.get_cmap("tab10"),
    cstart=0,
    cdist=1,
    figsize=(10, 1),
    title="",
    ylabel=None,
    ylim=None,
    rotation=90,
    width=0.7,
    fontsize=6,
    ax=None,
    fig=None,
    showmeans=False,
    showmedians=True,
    grid=False,
    scatter=True,
    scatter_radius=3,
    scatter_edge_color=None,
    scatter_edge_width=0.5,
    violin_alpha=0.5,
    violin_marker_lw=0.5,
    violin_marker_color="k",
    color_by="groups",
    zorder_mean_median=5,
    zorder_min_max=5,
    mean_median_linewidth=0.5,
    mean_median_color="k",
    mean_median_bar_length=None,
    **kwargs,
):
    """
    Args:
        values: matrix of shape (#random variables, #groups, #samples).
                random variables are labeled with xticklabels.
                groups are labeled with legend.
        xticklabels: #independents labels
        legend: #groups labels
        legend_kwargs: cosmetics for the legend, see matplotlib docs.
        as_bars: switch from violins to bars.
        scatter: scatter plot of data points on top.
        cmap: colormap.
        cdist: color distance between groups, when indexing cmap.
        ...
    """
    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax, fig)
    if grid:
        ax.yaxis.grid(zorder=-100)

    def plot_bar(X, values, color):
        handle = ax.bar(x=X, width=width, height=np.mean(values), color=color, zorder=1)
        return handle

    def plot_violin(X, values, color):
        # breakpoint()

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
            # pc.set_edgecolor(color)
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
                parts["cmeans"].set_segments(
                    [
                        [
                            [x0_vert - mean_median_bar_length * width / 2, y0],
                            [x0_vert + mean_median_bar_length * width / 2, y1],
                        ]
                    ]
                )
        if "cmedians" in parts:
            parts["cmedians"].set_color(mean_median_color)
            parts["cmedians"].set_linewidth(mean_median_linewidth)
            parts["cmedians"].set_zorder(zorder_mean_median)
            # parts["cmedians"].set_alpha(0.8)
            # breakpoint()
            if mean_median_bar_length is not None:
                (_, y0), (_, y1) = parts["cmedians"].get_segments()[0]
                (x0_vert, _), _ = parts["cbars"].get_segments()[0]
                parts["cmedians"].set_segments(
                    [
                        [
                            [x0_vert - mean_median_bar_length * width / 2, y0],
                            [x0_vert + mean_median_bar_length * width / 2, y1],
                        ]
                    ]
                )
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
            C = np.asarray(
                [cmap(cstart + i * cdist) for i in range(n_random_variables)]
            ).reshape(n_random_variables, 4)
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

    try:
        ax.set_xlim(np.min(X - width), np.max(X + width))
    except ValueError:
        pass

    ax.set_ylabel(ylabel or "", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    if pvalues is not None:
        display_pvalues(ax, pvalues, xticklabels, values, **display_pvalues_kwargs)

    return fig, ax, ViolinData(values, X, colors)


def multi_row_violins(
    x,
    Y,
    min_features_per_ax=100,
    figwidth=10,
    ylabel="",
    as_bars=False,
    colors=None,
    color_by="experiments",
):

    samples, features = Y.shape
    n_axes = int(features / min_features_per_ax)
    features_per_ax = min_features_per_ax + int(
        np.ceil((features % min_features_per_ax) / n_axes)
    )

    fig, axes, _ = plt_utils.get_axis_grid(
        gridheight=n_axes, gridwidth=1, figsize=[10, n_axes * 1.2], hspace=1
    )

    for i in range(n_axes):
        fig, ax, C = violins(
            data=Y[:, i * features_per_ax : (i + 1) * features_per_ax],
            xticklabels=x[i * features_per_ax : (i + 1) * features_per_ax],
            fig=fig,
            ax=axes[i],
            as_bars=as_bars,
            colors=colors[i * features_per_ax : (i + 1) * features_per_ax]
            if colors is not None
            else None,
            color_by=color_by,
        )

        ax.grid(False)
        ax.tick_params(axis="both", width=0.5, length=3, pad=2, labelsize=5)
        [i.set_linewidth(0.5) for i in ax.spines.values()]
        ax.set_facecolor("none")

    if ylabel:
        lefts, bottoms, rights, tops = np.array(
            [ax.get_position().extents for ax in axes]
        ).T
        fig.text(
            lefts.min() - 0.2 * lefts.min(),
            (tops.max() - bottoms.min()) / 2,
            ylabel,
            rotation=90,
            fontsize=5,
            ha="right",
            va="center",
        )
    return fig, axes, features_per_ax


def violin_groups_v2(
    values,
    xticklabels=None,
    pvalues=None,
    display_pvalues_kwargs={},
    legend=False,
    legend_kwargs=dict(
        fontsize=5,
        markerscale=10,
        loc="lower left",
        bbox_to_anchor=(0.75, 0.9),
    ),
    as_bars=False,
    scatter=False,
    colors=None,
    cmap=cm.get_cmap("tab10"),
    cstart=0,
    cdist=1,
    figsize=(10, 1),
    title="",
    ylabel=None,
    ylim=None,
    rotation=90,
    width=0.7,
    fontsize=6,
    ax=None,
    fig=None,
    showmeans=False,
    showmedians=True,
    grid=False,
    scatter_radius=5,
    scatter_edge_color="white",
    scatter_edge_width=0.25,
    violin_alpha=0.5,
    violin_marker_lw=0.25,
    violin_marker_color="k",
    color_by="groups",
    **kwargs,
):

    """Agnostic to same number of samples across groups!

    Args:
        values: array-like of shape (#groups, #samples, #random variables).
                random variables are labeled with xticklabels.
                groups are labeled with legend.
        xticklabels: #independents labels
        legend: #groups labels
        legend_kwargs: cosmetics for the legend, see matplotlib docs.
        as_bars: switch from violins to bars.
        scatter: scatter plot of data points on top.
        cmap: colormap.
        cdist: color distance between groups, when indexing cmap.
        ...
    """
    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax, fig)
    if grid:
        ax.yaxis.grid(zorder=-200, linewidth=violin_marker_lw)

    def plot_bar(X, values, color):
        handle = ax.bar(x=X, width=width, height=np.mean(values), color=color, zorder=1)
        return handle

    def plot_violin(X, values, color):
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
            # pc.set_edgecolor(color)
            pc.set_alpha(violin_alpha)
        # Color the lines.
        parts["cbars"].set_color(violin_marker_color)
        parts["cbars"].set_linewidth(violin_marker_lw)
        parts["cmaxes"].set_color(violin_marker_color)
        parts["cmaxes"].set_linewidth(violin_marker_lw)
        parts["cmins"].set_color(violin_marker_color)
        parts["cmins"].set_linewidth(violin_marker_lw)
        if "cmeans" in parts:
            parts["cmeans"].set_color(violin_marker_color)
            parts["cmeans"].set_linewidth(violin_marker_lw)
        if "cmedians" in parts:
            parts["cmedians"].set_color(violin_marker_color)
            parts["cmedians"].set_linewidth(violin_marker_lw)
        return parts["bodies"][0]

    n_groups = len(values)
    n_random_variables = values[0].shape[1]

    for i in range(n_groups):
        if values[i].shape[1] != n_random_variables:
            raise ValueError("different amount of random variables among groups")

    # Create matrix for x position for each bar.
    X = np.zeros([n_random_variables, n_groups])
    x = (
        np.arange(0, n_groups * n_random_variables, n_groups)
        - width * (n_groups / 2)
        + width / 2
    )
    for j in range(n_groups):
        X[:, j] = x + j * width

    # breakpoint()

    if colors is None:
        # Create matrix of colors.
        if color_by == "groups":
            C = np.asarray([cmap(cstart + i * cdist) for i in range(n_groups)]).reshape(
                n_groups, 4
            )
        if color_by == "random_variables":
            C = np.asarray(
                [cmap(cstart + i * cdist) for i in range(n_random_variables)]
            ).reshape(n_random_variables, 4)
    elif isinstance(colors, Iterable):
        if color_by == "groups":
            if len(colors) == n_groups:
                C = colors
            else:
                raise ValueError
        if color_by == "random_variables":
            if len(colors) == n_random_variables:
                C = colors
            else:
                raise ValueError
    else:
        raise ValueError

    # Plot each violin or bar and optionally scatter.
    handles = []
    for j in range(n_groups):
        for i in range(n_random_variables):
            if color_by == "random_variables":
                _color = C[i]
            elif color_by == "groups":
                _color = C[j]
            else:
                raise ValueError

            if as_bars:
                h = plot_bar(X[i, j], values[j][:, i], _color)
            else:
                h = plot_violin(X[i, j], values[j][:, i], _color)

            if scatter:
                lims = plt_utils.get_lims(
                    (-width / (2 * n_groups), width / (2 * n_groups)), -0.05
                )
                xticks = np.ones_like(values[j][:, i]) * X[i, j]
                ax.scatter(
                    xticks + np.random.uniform(*lims, size=len(xticks)),
                    values[j][:, i],
                    s=scatter_radius,
                    zorder=-100,
                    facecolor="none",
                    edgecolor=_color,
                    linewidth=scatter_edge_width,
                    alpha=0.35,
                    **kwargs,
                )
        handles.append(h)

    if legend:
        ax.legend(handles, legend, **legend_kwargs)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.tick_params(axis="both", which="major", labelsize=fontsize)

    if xticklabels is not None:
        ax.set_xticks(x + (n_groups / 2 - 1 / 2) * width)
        ax.set_xticklabels(xticklabels, rotation=rotation)

    # try:
    #     # ax.set_xlim(np.min(x), np.max(x) + n_groups * width)
    # except ValueError:
    #     pass

    ax.set_ylabel(ylabel or "", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    if pvalues is not None:
        display_pvalues(ax, pvalues, xticklabels, values, **display_pvalues_kwargs)

    return fig, ax, C


def violin_groups_v3(
    values,
    xticklabels=None,
    pvalues=None,
    display_pvalues_kwargs={},
    legend=False,
    legend_kwargs=dict(
        fontsize=5,
        markerscale=10,
        loc="lower left",
        bbox_to_anchor=(0.75, 0.9),
    ),
    as_bars=False,
    scatter=False,
    colors=None,
    cmap=cm.get_cmap("tab10"),
    cstart=0,
    cdist=1,
    figsize=(10, 1),
    title="",
    ylabel=None,
    ylim=None,
    rotation=90,
    width=0.7,
    fontsize=6,
    ax=None,
    fig=None,
    showmeans=False,
    showmedians=True,
    grid=False,
    scatter_radius=5,
    scatter_edge_color="white",
    scatter_edge_width=0.25,
    violin_alpha=0.5,
    violin_marker_lw=0.25,
    violin_marker_color="k",
    color_by="groups",
    scatter_alpha=0.35,
    **kwargs,
):

    """Agnostic to same number of samples across random variables!

    Args:
        values: array-like of shape (#groups, #random variables, #samples).
                random variables are labeled with xticklabels.
                groups are labeled with legend.
        xticklabels: #independents labels
        legend: #groups labels
        legend_kwargs: cosmetics for the legend, see matplotlib docs.
        as_bars: switch from violins to bars.
        scatter: scatter plot of data points on top.
        cmap: colormap.
        cdist: color distance between groups, when indexing cmap.
        ...
    """
    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax, fig)
    if grid:
        ax.yaxis.grid(zorder=-200, linewidth=violin_marker_lw)

    def plot_bar(X, values, color):
        handle = ax.bar(x=X, width=width, height=np.mean(values), color=color, zorder=1)
        return handle

    def plot_violin(X, values, color):
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
            # pc.set_edgecolor(color)
            pc.set_alpha(violin_alpha)
        # Color the lines.
        parts["cbars"].set_color(violin_marker_color)
        parts["cbars"].set_linewidth(violin_marker_lw)
        parts["cmaxes"].set_color(violin_marker_color)
        parts["cmaxes"].set_linewidth(violin_marker_lw)
        parts["cmins"].set_color(violin_marker_color)
        parts["cmins"].set_linewidth(violin_marker_lw)
        if "cmeans" in parts:
            parts["cmeans"].set_color(violin_marker_color)
            parts["cmeans"].set_linewidth(violin_marker_lw)
        if "cmedians" in parts:
            parts["cmedians"].set_color(violin_marker_color)
            parts["cmedians"].set_linewidth(violin_marker_lw)
        return parts["bodies"][0]

    n_groups = len(values)
    n_random_variables = len(values[0])

    for i in range(n_groups):
        if len(values[i]) != n_random_variables:
            raise ValueError("different amount of random variables among groups")

    # Create matrix for x position for each bar.
    X = np.zeros([n_random_variables, n_groups])
    x = (
        np.arange(0, n_groups * n_random_variables, n_groups)
        - width * (n_groups / 2)
        + width / 2
    )
    for j in range(n_groups):
        X[:, j] = x + j * width

    # breakpoint()

    if colors is None:
        # Create matrix of colors.
        if color_by == "groups":
            C = np.asarray([cmap(cstart + i * cdist) for i in range(n_groups)]).reshape(
                n_groups, 4
            )
        if color_by == "random_variables":
            C = np.asarray(
                [cmap(cstart + i * cdist) for i in range(n_random_variables)]
            ).reshape(n_random_variables, 4)
    elif isinstance(colors, Iterable):
        if color_by == "groups":
            if len(colors) == n_groups:
                C = colors
            else:
                raise ValueError
        if color_by == "random_variables":
            if len(colors) == n_random_variables:
                C = colors
            else:
                raise ValueError
    else:
        raise ValueError

    # Plot each violin or bar and optionally scatter.
    handles = []
    for j in range(n_groups):
        for i in range(n_random_variables):
            if color_by == "random_variables":
                _color = C[i]
            elif color_by == "groups":
                _color = C[j]
            else:
                raise ValueError

            if as_bars:
                h = plot_bar(X[i, j], values[j][i], _color)
            else:
                h = plot_violin(X[i, j], values[j][i], _color)

            if scatter:
                lims = plt_utils.get_lims(
                    (-width / (2 * n_groups), width / (2 * n_groups)), -0.05
                )
                xticks = np.ones_like(values[j][i]) * X[i, j]
                ax.scatter(
                    xticks + np.random.uniform(*lims, size=len(xticks)),
                    values[j][i],
                    s=scatter_radius,
                    zorder=-100,
                    facecolor="none",
                    edgecolor=_color,
                    linewidth=scatter_edge_width,
                    alpha=scatter_alpha,
                    **kwargs,
                )
        handles.append(h)

    if legend:
        ax.legend(handles, legend, **legend_kwargs)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.tick_params(axis="both", which="major", labelsize=fontsize)

    if xticklabels is not None:
        ax.set_xticks(x + (n_groups / 2 - 1 / 2) * width)
        ax.set_xticklabels(xticklabels, rotation=rotation)

    # try:
    #     # ax.set_xlim(np.min(x), np.max(x) + n_groups * width)
    # except ValueError:
    #     pass

    ax.set_ylabel(ylabel or "", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    if pvalues is not None:
        display_pvalues(ax, pvalues, xticklabels, values, **display_pvalues_kwargs)

    return fig, ax, C


# ---- POTENTIAL


def potentials(
    ylabels,
    activity,
    t_stim=None,
    rectify=False,
    zero_center=False,
    max_norm=False,
    colors=None,
    offset=3,
    var_labels_right=None,
    node_label=True,
    tau_label=True,
    y_axis=False,
    labelpad=0,
    fig=None,
    ax=None,
    label_at_offset=False,
    vmin=0,
    vmax=0,
    midpoint=None,
    figsize=(10, 10),
    cmap=cm.get_cmap("Blues"),
    ylim=(None, None),
    fontsize=10,
    title="",
):
    """Plots the membrane potential.
    Args:
        ylabels (List): rows with node types.
        activity (array): (#frames, #nodes).
        time_const (bool): optional time constant annotation.
    """
    fig, ax = plt_utils.init_plot(
        figsize=figsize, title=title, fontsize=fontsize, ax=ax, fig=fig
    )

    if t_stim is not None:
        time = np.linspace(0, t_stim, activity.shape[0])
        dt = t_stim / len(time)
        ax.set_xlabel("time in s", fontsize=fontsize)
    else:
        time = np.arange(activity.shape[0])
        dt = 1
        t_stim = len(time)
        ax.set_xlabel("frame", fontsize=fontsize)

    variation = activity.std(axis=0) / (activity.mean(axis=0) + 1e-15)
    if rectify:
        activity = np.maximum(activity, 0)
    if zero_center:
        activity = activity - activity[0]
    if max_norm:
        activity = activity / (np.abs(activity).max(axis=0, keepdims=True) + 1e-15)

    _offset = 3 * np.ones_like(activity) * np.arange(0.5, activity.shape[1] + 0.5)
    _activity = activity + _offset

    if not colors:
        # get disctinct colors for individual traces
        scalarmapper, norm = plt_utils.get_scalarmapper(
            cmap=cmap, vmin=vmin, vmax=vmax, midpoint=midpoint
        )
        color_rgba = scalarmapper.to_rgba(_offset[0])
    else:
        color_rgba = colors

    # Plotting.
    for i, trace in enumerate(range(_activity.shape[1])):
        ax.plot(time, _activity[:, i], color=color_rgba[i])
    ax.tick_params(axis="x", which="major", labelsize=fontsize)
    # ax.spines['bottom'].set_bounds(time[0], time[-1])
    # Set y-axis limits.
    # ymin = -0.1
    # ymax = _offset[0,:].max() + 1.1
    ax.set_ylim(*plt_utils.get_lims(_offset, 0.11))
    ax = plt_utils.rm_spines(ax, ["left"], rm_xticks=False, rm_yticks=True)
    ax.hlines(_offset[0], xmin=time[0], xmax=time[-1], linewidth=0.5, color="0.5")

    # Annotate with nodes.
    for i, (label, _offset) in enumerate(zip(ylabels, _activity[0, :])):
        ax.annotate(
            label,
            fontsize=fontsize,
            xy=(time[-1] + labelpad * time[-1], _offset),
            ha="left",
            va="center",
        )  # , color=color_rgba[i])

    if var_labels_right:
        for i, (label, _offset) in enumerate(zip(variation, _activity[-1, :])):
            ax.annotate(
                "σ/μ: {:.2G}".format(label),
                fontsize=fontsize,
                xy=(time[-1] + 2 * dt, _offset),
                ha="left",
                va="center",
                color=color_rgba[i],
            )

    return fig, ax


def potential_over_frames(
    nodes,
    activity,
    time_const=None,
    offset=2,
    node_label=True,
    tau_label=True,
    y_axis=False,
    fig=None,
    ax=None,
    label_at_offset=False,
    figsize=(10, 10),
    cmap=cm.get_cmap("tab20b"),
    ylim=(None, None),
    fontsize=10,
    title="",
):
    """Plots the membrane potential.
    Args:
        nodes (dataframe): rows with node types, time constants.
        activity (array): (#frames, #nodes).
        time_const (bool): optional time constant annotation.
    """
    fig, ax = plt_utils.init_plot(
        figsize=figsize, title=title, fontsize=fontsize, ax=ax, fig=fig
    )

    # reverse order
    activity = activity[:, ::-1]
    nodes = nodes[::-1]

    # add some offset to the traces so that they fit in a single plot
    _low = np.abs(np.mean(activity[activity <= 0])) if (activity <= 0).any() else 0
    _high = np.abs(np.mean(activity[activity > 0])) if (activity > 0).any() else 0
    _dist = offset * (_low + _high)
    _offset_matrix = (
        _dist * np.ones_like(activity) * np.arange(1, activity.shape[1] + 1)
    )
    _activity = activity + _offset_matrix

    # get disctinct colors for individual traces
    np.random.seed(0)
    colors = [cmap(i) for i in np.linspace(0, 1, activity.shape[1])]
    np.random.shuffle(colors)
    ax.set_prop_cycle("color", colors)

    # Plotting.
    frames = np.arange(1, activity.shape[0] + 1)
    ax.plot(frames, _activity)
    ax.tick_params(axis="x", which="major", labelsize=fontsize)
    # ax.spines['bottom'].set_bounds(frames[0], frames[-1])
    ax.set_xlabel("Frame", fontsize=fontsize)

    # Set axis limits.
    y_offset = _offset_matrix.mean(axis=0)
    ymin, ymax = plt_utils.get_lims(_activity, 0.1)
    ymin = ylim[0] or min(ymin, y_offset.min() - 0.1 * y_offset.min())
    ymax = ylim[1] or max(ymax, y_offset.max() + 0.1 * y_offset.max())
    ax.set_ylim(ymin, ymax)
    if y_axis is False:
        ax = plt_utils.rm_spines(ax, ["left"], rm_xticks=False, rm_yticks=True)

    # Annotate with node types and time constants.
    nodes.insert(
        0,
        "y_offset_initial",
        _activity[0, :] if not label_at_offset else y_offset,
    )
    nodes.insert(
        0, "y_offset_end", _activity[-1, :] if not label_at_offset else y_offset
    )
    nodes.reset_index(inplace=True)
    for (i, row) in nodes.iterrows():
        if node_label is True:
            ax.annotate(
                row.type,
                fontsize=fontsize,
                xy=(frames[0] - 0.2, row.y_offset_initial),
                ha="right",
                va="center",
                color=colors[i],
            )
        if time_const and tau_label:
            tau = f" $\\tau$={row.time_const_trained * 1000:.1f}ms"
            ax.annotate(
                tau,
                fontsize=fontsize,
                xy=(frames[-1] + 0.2, row.y_offset_end),
                ha="left",
                va="center",
                color=colors[i],
            )

    return fig, ax


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

    if not closed:
        r = np.append(r, np.expand_dims(r[0], 0), axis=0)

    line_effects = None
    if stroke_kwargs:
        line_effects = [
            path_effects.Stroke(**stroke_kwargs),
            path_effects.Normal(),
        ]

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
    ax.set_xticks(
        [
            0,
            np.pi / 4,
            np.pi / 2,
            3 / 4 * np.pi,
            np.pi,
            5 / 4 * np.pi,
            3 / 2 * np.pi,
            7 / 4 * np.pi,
        ]
    )
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
    ax.set_xticks(
        [
            0,
            np.pi / 4,
            np.pi / 2,
            3 / 4 * np.pi,
            np.pi,
            5 / 4 * np.pi,
            3 / 2 * np.pi,
            7 / 4 * np.pi,
        ]
    )
    ax.set_xticklabels(["0°", "45°", "90°", "", "", "", "", ""])

    ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=xlabelpad, fontweight=fontweight)
    if all((val is not None for val in (ymin, ymax))):
        ax.set_ylim((ymin, ymax))
    plt.setp(ax.spines.values(), color="grey", linewidth=1)

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
    ax.set_xticks(
        [
            0,
            np.pi / 4,
            np.pi / 2,
            3 / 4 * np.pi,
            np.pi,
            5 / 4 * np.pi,
            3 / 2 * np.pi,
            7 / 4 * np.pi,
        ]
    )
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
    """Plots traces with optional smoothing.

    Args:
        trace (array): 2D array (#traces, #points).
        legend (list, optional): legend for each trace. Defaults to [].
        smooth (float, optional): size of smoothing window in percent of #points.
            Default is 0.05.

    Returns:
        fig, ax, trace (smoothed)
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

    # if (
    #     isinstance(color, str)
    #     or isinstance(color, tuple)
    #     or isinstance(color, np.ndarray)
    # ):
    #     colors = (color,) * shape[0]
    # elif isinstance(color, list):
    #     colors = color
    # else:
    #     colors = (None,) * shape[0]
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
            cmap=cm.get_cmap("bone"),
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

    # if ylim is None:
    #     ylim = plt_utils.get_lims(np.ma.masked_invalid(trace), 0.01)
    # ax.set_ylim(ylim)

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


# ---- SCATTER


def scatter(
    x,
    y,
    ax=None,
    fig=None,
    color=None,
    markersize=20,
    title="",
    figsize=(7, 4),
    fontsize=10,
    grid=True,
    xlabel=None,
    ylabel=None,
    **kwargs,
):
    """Scatter plot of vectors x and y.

    Args:
        x (array): x values.
        y (array): y values.
        .. style, matplotlib specific arguments.

    Returns:
        fig, ax
    """
    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax=ax, fig=fig)

    # remove redundant pairs
    x, y = np.unique(np.stack((x, y)), axis=1)

    ax.scatter(x, y, c=color, s=markersize)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    return fig, ax


def labeled_scatter(
    x,
    y,
    labels,
    offset_in_points=(0, -10),
    color=None,
    size=20,
    figsize=[4, 4],
    xlabel="",
    ylabel="",
    fontsize=10,
):
    fig, ax = plt_utils.init_plot(figsize=figsize)
    ax.scatter(x, y, s=size, color=None)
    if isinstance(offset_in_points, tuple):
        offset_in_points = (offset_in_points,) * len(labels)
    elif isinstance(offset_in_points, list):
        pass
    else:
        raise AssertionError
    for i, (_x, _y, label) in enumerate(zip(x, y, labels)):
        ax.annotate(
            label,
            fontsize=fontsize,
            xy=(_x, _y),
            xytext=offset_in_points[i],
            textcoords="offset points",
            ha="center",
            va="center",
        )
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    return fig, ax


# ---- REPRESENTATIONAL SIMILARITY


def rs_matrix(
    matrix,
    cmap=cm.get_cmap("viridis"),
    cbar=True,
    figsize=(5, 5),
    vmin=None,
    vmax=None,
    ax=None,
    fig=None,
    title=None,
    ticklabels=(),
    fontsize=10,
    midpoint=0.0,
):
    """Plots a representational similarity matrix using imshow.

    Args:
        matrix (array): (N, N) matrix.
        matplotlib specific arguments

    Returns:
        fig, ax, (cbar, cmap, norm)
    """
    fig, ax = plt_utils.init_plot(fig=fig, ax=ax, figsize=figsize, fontsize=fontsize)
    norm = plt_utils.get_norm(vmin=vmin, vmax=vmax, midpoint=midpoint)

    ax.imshow(matrix, cmap=cmap, norm=norm, interpolation=None)

    if cbar:
        cbar = plt_utils.add_colorbar(fig, ax, cmap=cmap, norm=norm)

    ax.set_xticks(range(len(ticklabels)))
    ax.set_yticks(range(len(ticklabels)))
    ax.set_xticklabels(ticklabels, rotation=45)
    ax.set_yticklabels(ticklabels)
    ax.tick_params(labelsize=fontsize, pad=0)

    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(True)
    ax.xaxis.tick_top()

    ax.text(
        0.5,
        -0.05,
        title,
        ha="center",
        va="center",
        fontweight="bold",
        fontsize=fontsize,
        transform=ax.transAxes,
    )

    return fig, ax, (cbar, cmap, norm)


# ---- GRATINGS TUNING


def gratings_traces(
    input,
    response,
    dt,
    figsize=[2, 2],
    vmin=0,
    vmax=1,
    color=None,
    fig=None,
    ax=None,
    fontsize=10,
    xlabel="",
):
    fig, ax = plt_utils.init_plot(figsize=figsize, fig=fig, ax=ax)

    _x = np.arange(len(input)) * dt
    _y = np.linspace(-2000, 2000, 100)
    Z = np.tile(input, (len(_y), 1))
    ax.contourf(
        _x,
        _y,
        Z,
        cmap=cm.get_cmap("binary_r"),
        levels=2,
        alpha=0.3,
        vmin=vmin,
        vmax=vmax,
    )

    ax.plot(_x, input, color="k", linestyle="--")
    ax.plot(_x, response, color=color)

    plt.locator_params(nbins=2)
    xlabel = xlabel or "Time in s"
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.set_ylim(
        *plt_utils.get_lims(
            np.array(
                [
                    min(input.min(), response.min()),
                    max(input.max(), response.max()),
                ]
            ),
            0.1,
        )
    )
    return fig, ax


def speed_width_on_off_tuning(
    activities,
    node_type="T4d",
    cmap=cm.get_cmap("coolwarm_r"),
    ax=None,
    fig=None,
    fontsize=10,
    **kwargs,
):
    """
    Args:
        activities (dict): {"on": {width_0: {speed_0: datastruct.Activity, ... , speed_n: datastruct.Activity},
                                ...
                            width_m: {speed_0: datastruct.Activity, ... , speed_n: datastruct.Activity}}
                            "off": {width_0: {speed_0: datastruct.Activity, ... , speed_n: datastruct.Activity},
                                ...
                            width_m: {speed_0: datastruct.Activity, ... , speed_n: datastruct.Activity}}}

    Note: see function plots.datastructs.activity_by_on_off_width_speed
    """
    # Get axis limits.
    ymin = 1e15
    ymax = -1e15
    for on_or_off, activity_i in activities.items():
        for w, activity_ij in activity_i.items():
            for s, activity_ijk in activity_ij.items():
                ymin = min(activity_ijk[node_type].min(), ymin)
                ymax = max(activity_ijk[node_type].max(), ymax)

    rows = len(activities)
    columns = len(activities[next(iter(activities.keys()))])
    fig, axes, (gw, gh) = plt_utils.get_axis_grid(
        list(range(columns * rows)),
        gridwidth=columns,
        fig=fig,
        ax=ax,
        gridheight=rows,
        projection="polar",
    )
    fig.suptitle(node_type)
    axes = np.array(axes).reshape(gh, gw)
    for i, (on_or_off, activity) in enumerate(activities.items()):
        for j, (w, activity_given_width_and_speeds) in enumerate(activity.items()):
            ylabel = on_or_off.upper() if j == 0 else ""
            xlabel = f"Bar width: {w} col." if j == 0 else f"{w} col."
            cbar = True if (i == 0 and j == columns - 1) else False
            speed_tuning(
                activity_given_width_and_speeds,
                ylabel=ylabel,
                xlabel=xlabel,
                node_type=node_type,
                fig=fig,
                ax=axes[i, j],
                ymin=ymin,
                ymax=ymax,
                cbar=cbar,
                fontsize=fontsize,
                **kwargs,
            )
    return fig, axes


def speed_polar(
    theta,
    activities,
    speeds,
    cmap=cm.get_cmap("coolwarm"),
    ax=None,
    fig=None,
    xlabel="",
    cbar=True,
    ylabel="",
    ymin=None,
    ymax=None,
    fontsize=10,
    cbar_offset=(1.1, 0),
    **kwargs,
):

    cmap = plt_utils.get_discrete_color_map(
        speeds, cmap, vmin=speeds.min(), vmax=speeds.max()
    )

    for i, activity in enumerate(activities):
        speed = speeds[i]
        fig, ax = polar(
            theta,
            activity,
            color=cmap(i),
            linestyle="-",
            marker="",
            xlabel=xlabel,
            fig=fig,
            ax=ax,
            fontsize=fontsize,
            fontweight="normal",
            **kwargs,
        )
        ax.set_ylabel(ylabel, fontsize=fontsize)
        if ymin and ymax:
            ax.set_ylim((ymin, ymax))

    if cbar:
        cbar = plt_utils.add_colorbar(
            fig,
            ax,
            pos="right",
            cmap=cmap,
            x_offset=cbar_offset[0],
            y_offset=cbar_offset[1],
            width=0.05,
            height=0.7,
            label="speed in °/s",
            fontsize=fontsize,
            tick_width=0.5,
            tick_length=1,
        )
        cbar.set_ticks(np.arange(0, 1, 1 / len(speeds)) + 0.5 * 1 / len(speeds))
        cbar.set_ticklabels(np.round(speeds * 5.8, 1))

    return fig, ax, cbar


def speed_tuning(
    activities,
    node_type="T4d",
    cmap=cm.get_cmap("coolwarm_r"),
    ax=None,
    fig=None,
    xlabel="",
    cbar=True,
    ylabel="",
    ymin=None,
    ymax=None,
    fontsize=10,
    cbar_offset=(1, 0),
    **kwargs,
):
    """
    Args:
        activities (dict): {speed_0: datastruct.Activity, ..., speed_n: datastruct.Activity}.

    Note: see function plots.datastructs.activity_by_on_off_width_speed
    """
    speeds = np.array([speed for speed in activities])

    cmap = plt_utils.get_discrete_color_map(
        speeds,
        cmap,
        vmin=speeds.min(),
        vmax=speeds.max() - np.median(speeds),
        midpoint=np.median(speeds),
    )

    for i, (speed, activity) in enumerate(activities.items()):
        # print(speed, cmap(i))
        fig, ax = polar(
            activity.batch_dim,
            activity[node_type],
            color=cmap(i),
            linestyle="-",
            marker="",
            xlabel=xlabel,
            fig=fig,
            ax=ax,
            fontsize=fontsize,
            fontweight="normal",
            **kwargs,
        )
        ax.set_ylabel(ylabel)
        if ymin and ymax:
            ax.set_ylim((ymin, ymax))

    if cbar:
        cbar = plt_utils.add_colorbar(
            fig,
            ax,
            pos="right",
            cmap=cmap,
            x_offset=cbar_offset[0],
            y_offset=cbar_offset[1],
            width=0.1,
            height=0.9,
            label="Speed in col./s",
            fontsize=fontsize,
        )
        cbar.set_ticks(np.arange(0, 1, 1 / len(speeds)) + 0.5 * 1 / len(speeds))
        cbar.set_ticklabels(speeds)

    return fig, ax


# ---- FLASH RESPONSES


def stimulus_response(
    activity,
    conditions,
    dt=1 / 200,
    title="",
    fig=None,
    ax=None,
    cmap=cm.get_cmap("binary"),
    ylabel="Activity",
    xlabel="Time in s",
    fontsize=10,
    rm_yaxis=False,
    figsize=[10, 3],
    **kwargs,
):
    """
    Plots a response to a binary stimulus (specified over time in conditions array).
    """
    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax, fig, **kwargs)

    time = np.arange(0, len(activity)) * dt
    traces(activity, x=time, smooth=False, fig=fig, ax=ax)

    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.set_title(title, fontsize=fontsize)

    binary_shading(ax, time, conditions, cmap)

    if rm_yaxis:
        plt_utils.rm_spines(ax, spines=("left",), visible=False, rm_yticks=True)

    return fig, ax


def binary_shading(ax, time, conditions, cmap=cm.get_cmap("binary")):
    sm, norm = plt_utils.get_scalarmapper(
        vmin=conditions.min(), vmax=conditions.max(), cmap=cm.get_cmap("binary")
    )
    _edges = np.where(np.diff(conditions) != 0)[0]
    startend = []
    start = 0
    for end in _edges:
        startend.append((start, end))
        start = end
    startend.append((start, time.size - 2))
    for i, (start, end) in enumerate(startend):
        color = sm.to_rgba(conditions[end + 1])
        if i == len(startend) - 1:
            color = sm.to_rgba(conditions[start])
        ax.axvspan(time[start], time[end], facecolor=color, alpha=0.2, zorder=-100)


def flash_response(
    activities,
    node_type="T4a",
    fig=None,
    ax=None,
    cmap=cm.get_cmap("binary_r"),
    ymin=None,
    ymax=None,
    wspace=0.2,
    ylabel="Activity",
    xlabel="Time in s",
    fontsize=10,
    cbar=True,
    cbar_height=0.05,
    cbar_offset=(0, -1.5),
    rm_yaxis=False,
    figsize=[10, 3],
):
    """
    Args:
        activities (dict): {"idx_0": datastruct.ActivityByNode,
                                ...
                            "idx_n": datastruct.ActivityByNode}.

    Note: see function plots.datastructs.activity_by_flash
    """
    fig, ax, _ = plt_utils.get_axis_grid(
        list(range(len(activities))), fig=fig, figsize=figsize, ax=ax
    )

    plt.subplots_adjust(wspace=wspace) if wspace is not False else None

    ymin, ymax = 1e5, 1e-5
    for i, (key, activity) in enumerate(activities.items()):

        a = activity[node_type].squeeze()
        ymin, ymax = min(ymin, a.min()), max(ymax, a.max())
        traces(
            a,
            x=np.arange(0, len(a)) * activity.spec.dt,
            smooth=False,
            fig=fig,
            ax=ax[i],
        )

        ax[i].set_xlabel(xlabel, fontsize=fontsize)
        ax[i].set_ylabel(ylabel, fontsize=fontsize) if i == 0 else None
        ax[i].tick_params(axis="both", which="major", labelsize=fontsize)

        sm, norm = plt_utils.get_scalarmapper(
            vmin=min(activity.spec.pre_stim, activity.spec.stim),
            vmax=max(activity.spec.pre_stim, activity.spec.stim),
            cmap=cm.get_cmap("binary_r"),
        )

        pre_stim, stim = sm.to_rgba(activity.spec.pre_stim), sm.to_rgba(
            activity.spec.stim
        )
        colors = np.array([pre_stim, stim])

        width = {0: activity.spec.t_pre, 1: activity.spec.t_stim}

        start = 0
        for state in activity.spec.alternations:
            ax[i].axvspan(
                start,
                start + width[state],
                facecolor=colors[state],
                alpha=0.2,
                zorder=-100,
            )
            start += width[state]

        if rm_yaxis:
            plt_utils.rm_spines(ax, spines=("left",), visible=False, rm_yticks=True)

        # cm = plt_utils.get_discrete_color_map(activity.time_dim, cmap, vmin=activity.time_dim.min(),
        #                                vmax=activity.time_dim.max(), midpoint=None)

        # cbar = plt_utils.add_colorbar(fig, ax[i],
        #                        pos='top',
        #                        x_offset=cbar_offset[0],
        #                        y_offset=cbar_offset[1],
        #                        rm_outline=False,
        #                        width=1, height=0.05,
        #                        cmap=cm, norm=None, plain=True)
    for _ax in ax:
        _ax.set_ylim(*plt_utils.get_lims(np.array([ymin, ymax]), 0.15))

    return fig, ax


# ---- PARAMETER STATISTICS (VIOLIN)


def param_stats(
    x,
    Y,
    mode="violin",
    title="",
    figsize=(20, 5),
    color=None,
    legend=True,
    ylabel=False,
    yscale="linear",
    markersize=1,
    fontsize=10,
    ax=None,
    fig=None,
    label="",
    labelxy=(0, 1.01),
    initial_value=None,
    initial_std=0,
    **kwargs,
):
    """Plots the statistics of parameters in a horizontal errorbar-, box- or violinplot.

    Args:
        x (array): x-Axis values.
        Y (array): data. Array of shape (n_samples, values).
        mode (str): 'bar', 'box' or 'violin'
    """
    fig, ax = plt_utils.init_plot(figsize, title, fontsize, ax, fig)
    mean = np.ma.masked_invalid(np.mean(Y, axis=0))
    std = np.ma.masked_invalid(np.std(Y, axis=0))

    if mode == "bar":
        ax.errorbar(
            x=x,
            y=mean,
            yerr=std,
            fmt="o",
            capsize=2,
            label="Mean",
            elinewidth=20,
        )
    elif mode == "box":
        ax.boxplot(Y, positions=range(len(x)))
    elif mode == "violin":
        ax.violinplot(Y, positions=range(len(x)), widths=0.9)

    if initial_value is not None:
        ax.axhline(y=initial_value, color="orange", zorder=-100)
    if initial_value is not None and initial_std != 0:
        ax.axhspan(
            initial_value - initial_std,
            initial_value + initial_std,
            facecolor="orange",
            alpha=0.2,
            zorder=-100,
        )

    legend = "Trained NN {}" if legend is True else ""
    for i in range(len(Y)):
        ax.plot(Y[i], "o", ms=markersize, color=color, label=legend.format(i + 1))
    ax.legend(edgecolor="white", fontsize=fontsize) if legend else ""

    kwargs = (
        dict(linthreshy=np.abs(Y).min() or 1e-5, subsy=[2, 3, 4, 5, 6, 7, 8, 9])
        if yscale == "symlog"
        else dict()
    )
    ax.set_yscale(yscale, **kwargs)

    xmin, xmax = -0.5, len(x) + 1
    ax.set_xlim((xmin, xmax))
    _data = np.concatenate((mean - std, mean + std, Y.min(axis=0), Y.max(axis=0)))
    ymin, ymax = _data.min(), _data.max()
    init_min, init_max = initial_value, initial_value
    if initial_value is not None and initial_std != 0:
        init_min -= initial_std
        init_max += initial_std
        lims = np.array([ymin, ymax, init_min, init_max])
    lims = np.array([ymin, ymax])
    ax.set_ylim(*plt_utils.get_lims(lims, 0.1))

    ax = plt_utils.rm_spines(ax, spines=["top", "right"])
    ax.tick_params(axis="both", which="major", labelsize=fontsize)
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, rotation=90)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)
    else:
        ax.set_title(title, fontsize=fontsize)

    for i, inv in enumerate(mean.mask):
        text = "NaN" if inv else ""
        ax.text(
            (i + np.abs(xmin)) / (xmax + np.abs(xmin)),
            0.1,
            text,
            ha="center",
            va="center",
            rotation=90,
            fontsize=fontsize,
            transform=ax.transAxes,
        )

    label_text = ax.text(
        labelxy[0],
        labelxy[1],
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
    )

    return fig, ax


def egoarrow(
    cam_disp,
    center=(0, 0),
    normalize=False,
    figsize=[7, 7],
    elev=110,
    azim=-90,
    fig=None,
    maxnorm=None,
    ax=None,
    title="",
    fontsize=10,
):
    """Plot ego displacement arrow.

    Args:
        cam_disp (array): 3d camera displacement (translational or rotational).
    """
    fig, ax = init_3d(
        ["dx", "dy", "\ndz"],
        elev=elev,
        azim=azim,
        fig=fig,
        ax=ax,
        title=title,
        figsize=figsize,
        fontsize=fontsize,
    )
    center = center + (0,)
    if normalize:
        cam_disp = cam_disp / (cam_disp.max() + 1e-15)

    maxnorm = maxnorm or np.abs(cam_disp).max()
    xlim, ylim, zlim = (
        (center[0] - maxnorm, center[0] + maxnorm),
        (center[1] - maxnorm, center[1] + maxnorm),
        (-abs(cam_disp[-1]), +abs(cam_disp[-1])),
    )
    ax.set_xlim(xlim), ax.set_ylim(ylim), ax.set_zlim(zlim)
    ax.invert_zaxis()
    zticklabels = list(ax.get_zticklabels())
    zticklabels[1].set_text("")
    ax.margins(0, 0, 0)

    # shadow in x-z
    shadowcenter = (center[0], ax.get_ylim()[0], center[-1])
    shadowpoint = (cam_disp[0], ax.get_ylim()[0], cam_disp[-1])
    arrow_s = plt_utils.Arrow3D(
        shadowcenter,
        shadowpoint,
        mutation_scale=10,
        lw=3,
        arrowstyle="fancy",
        color="k",
        alpha=0.3,
    )
    arrow_s.set_arrowstyle("fancy", head_width=1)
    ax.add_artist(arrow_s)

    # actual arrow
    arrow = plt_utils.Arrow3D(
        center, cam_disp, mutation_scale=10, lw=3, arrowstyle="fancy", color="r"
    )
    arrow.set_arrowstyle("fancy", head_width=1)
    ax.add_artist(arrow)

    return fig, ax
