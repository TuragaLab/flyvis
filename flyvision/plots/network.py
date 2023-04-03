"""Utils for plotting whole network graph."""
import numpy as np
import networkx as nx
from typing import Dict
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from toolz import valfilter

from flyvision.plots import plt_utils


class WholeNetworkFigure:
    def __init__(self, connectome):
        self.nodes = connectome.nodes.to_df()
        self.edges = connectome.edges.to_df()
        self.layout = dict(connectome.layout[:].astype(str))
        self.cell_types = connectome.unique_cell_types[:].astype(str)

    def init_figure(self, figsize=[15, 6], fontsize=6):
        self.fig, self.axes, self.axes_centers = network_layout_axes(
            self.layout, figsize=figsize
        )
        self.ax_dict = {ax.get_label(): ax for ax in self.axes}  # type: Dict[str, Axes]
        self.add_graph()
        self.add_decoded_box()
        self.add_ax_labels(fontsize=fontsize)

    def add_graph(
        self,
        edge_color_key=None,
        arrows=True,
        edge_alpha=1.0,
        edge_width=1.0,
        constant_edge_width=0.25,
        constant_edge_color="#c5c5c5",
        edge_cmap=None,
        self_loop_radius=0.02,
        self_loop_width=None,
        self_loop_edge_alpha=None,
    ):
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

        axes = {
            cell_type: [ax for ax in self.axes if ax.get_label() == cell_type][0]
            for cell_type in self.cell_types
        }

        (lefts, bottoms, rights, tops), (
            centers,
            widths,
            height,
        ) = plt_utils.get_ax_positions(list(axes.values()))
        edge_ax = self.fig.add_axes(
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

        fig_to_edge_ax = self.fig.transFigure + edge_ax.transData.inverted()
        positions = {
            key: fig_to_edge_ax.transform(value)
            for key, value in self.axes_centers.items()
        }

        nodes, edge_list = _network_graph(self.nodes, self.edges)

        if edge_color_key is not None and not constant_edge_color:
            grouped = self.edges.groupby(
                by=["source_type", "target_type"], sort=False, as_index=False
            ).mean(numeric_only=True)
            edge_color = {
                (row.source_type, row.target_type): row.sign
                for i, row in grouped.iterrows()
            }
            _edge_color = np.array(list(edge_color.values()))
            edge_vmin = -np.max(_edge_color) if np.any(_edge_color < 0) else 0
            edge_vmax = np.max(_edge_color)
        else:
            edge_color = {tuple(edge): constant_edge_color for edge in edge_list}
            edge_vmin = None
            edge_vmax = None

        grouped = self.edges.groupby(
            by=["source_type", "target_type"], sort=False, as_index=False
        ).mean(numeric_only=True)

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
            arrowstyle="-|>, head_length=0.4, head_width=0.075, widthA=1.0, widthB=1.0, lengthA=0.2, lengthB=0.2",
            width=np.array([edge_width[tuple(edge)] for edge in edge_list]),
        )

    def add_decoded_box(self, pad=0.2):
        output_cell_types = valfilter(lambda v: v == "output", self.layout)
        axes = {
            cell_type: [ax for ax in self.axes if ax.get_label() == cell_type][0]
            for cell_type in output_cell_types
        }
        (lefts, bottoms, rights, tops), _ = plt_utils.get_ax_positions(
            list(axes.values())
        )
        bottom, top = plt_utils.get_lims((bottoms, tops), 0.02)
        left, right = plt_utils.get_lims((lefts, rights), 0.01)
        decoded_box_ax = self.fig.add_axes(
            [
                left,
                bottom,
                right - left,
                top - bottom,
            ],
            label="decoded_box",
        )
        decoded_box_ax.patch.set_alpha(0)
        decoded_box_ax.spines["top"].set_visible(True)
        decoded_box_ax.spines["right"].set_visible(True)
        decoded_box_ax.spines["left"].set_visible(True)
        decoded_box_ax.spines["bottom"].set_visible(True)
        decoded_box_ax.set_xticks([])
        decoded_box_ax.set_yticks([])
        self.ax_dict["decoded box"] = decoded_box_ax

    def add_ax_labels(self, fontsize=5):
        for label, ax in self.ax_dict.items():
            if label in self.cell_types:
                ax.annotate(
                    label,
                    (0, 0.9),
                    xycoords="axes fraction",
                    va="bottom",
                    ha="right",
                    fontsize=fontsize,
                )

        retina_cell_types = valfilter(lambda v: v == "retina", self.layout)
        axes = {
            cell_type: [ax for ax in self.axes if ax.get_label() == cell_type][0]
            for cell_type in retina_cell_types
        }
        (lefts, bottoms, rights, tops), _ = plt_utils.get_ax_positions(
            list(axes.values())
        )
        self.fig.text(
            lefts.min() + (rights.max() - lefts.min()) / 2,
            0,
            "retina",
            fontsize=fontsize,
            va="top",
            ha="center",
        )

        intermediate_cell_types = valfilter(lambda v: v == "intermediate", self.layout)
        axes = {
            cell_type: [ax for ax in self.axes if ax.get_label() == cell_type][0]
            for cell_type in intermediate_cell_types
        }
        (lefts, bottoms, rights, tops), (
            centers,
            widths,
            height,
        ) = plt_utils.get_ax_positions(list(axes.values()))
        self.fig.text(
            lefts.min() + (rights.max() - lefts.min()) / 2,
            0,
            "lamina, medulla intrinsic cells, CT1",
            fontsize=fontsize,
            va="top",
            ha="center",
        )

        output_cell_types = valfilter(lambda v: v == "output", self.layout)
        axes = {
            cell_type: [ax for ax in self.axes if ax.get_label() == cell_type][0]
            for cell_type in output_cell_types
        }
        (lefts, bottoms, rights, tops), (
            centers,
            widths,
            height,
        ) = plt_utils.get_ax_positions(list(axes.values()))
        self.fig.text(
            lefts.min() + (rights.max() - lefts.min()) / 2,
            0,
            "T-shaped, transmedullary cells",
            fontsize=fontsize,
            va="top",
            ha="center",
        )


def network_layout_axes(
    layout,
    cell_types=None,
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
        if (cell_types is None or key in cell_types)
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
            {cell_type: axes[i] for i, cell_type in enumerate(new_pos)},
            new_pos,
        )
    return fig, axes, new_pos


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
