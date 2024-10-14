"""Utils for plotting whole network graph."""

import collections
import itertools
from numbers import Number
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from toolz import valfilter

from . import plt_utils


class WholeNetworkFigure:
    """
    Class for creating a whole network figure.

    Attributes:
        nodes (pd.DataFrame): DataFrame containing node information.
        edges (pd.DataFrame): DataFrame containing edge information.
        layout (Dict[str, str]): Dictionary mapping node types to layout positions.
        cell_types (List[str]): List of unique cell types.
        video (bool): Whether to include video node.
        rendering (bool): Whether to include rendering node.
        motion_decoder (bool): Whether to include motion decoder node.
        decoded_motion (bool): Whether to include decoded motion node.
        pixel_accurate_motion (bool): Whether to include pixel-accurate motion node.
    """

    def __init__(
        self,
        connectome,
        video: bool = False,
        rendering: bool = False,
        motion_decoder: bool = False,
        decoded_motion: bool = False,
        pixel_accurate_motion: bool = False,
    ):
        self.nodes = connectome.nodes.to_df()
        self.edges = connectome.edges.to_df()
        self.layout = dict(connectome.layout[:].astype(str))
        self.cell_types = connectome.unique_cell_types[:].astype(str)

        layout = {}
        if video:
            layout.update({"video": "cartesian"})
        if rendering:
            layout.update({"rendering": "hexagonal"})
        layout.update(dict(connectome.layout[:].astype(str)))
        if motion_decoder:
            layout.update({"motion decoder": "decoder"})
        if decoded_motion:
            layout.update({"decoded motion": "motion"})
        if pixel_accurate_motion:
            layout.update({"pixel-accurate motion": "motion"})
        self.layout = layout
        self.video = video
        self.rendering = rendering
        self.motion_decoder = motion_decoder
        self.decoded_motion = decoded_motion
        self.pixel_accurate_motion = pixel_accurate_motion

    def init_figure(
        self,
        figsize: List[int] = [15, 6],
        fontsize: int = 6,
        decoder_box: bool = True,
        cell_type_labels: bool = True,
        neuropil_labels: bool = True,
        network_layout_axes_kwargs: Dict = {},
        add_graph_kwargs: Dict = {},
    ) -> None:
        """
        Initialize the figure with various components.

        Args:
            figsize: Size of the figure.
            fontsize: Font size for labels.
            decoder_box: Whether to add a decoder box.
            cell_type_labels: Whether to add cell type labels.
            neuropil_labels: Whether to add neuropil labels.
            network_layout_axes_kwargs: Additional kwargs for network_layout_axes.
            add_graph_kwargs: Additional kwargs for add_graph.
        """
        self.fig, self.axes, self.axes_centers = network_layout_axes(
            self.layout, figsize=figsize, **network_layout_axes_kwargs
        )
        self.ax_dict = {ax.get_label(): ax for ax in self.axes}
        self.add_graph(**add_graph_kwargs)

        self.add_retina_box()

        if decoder_box:
            self.add_decoded_box()

        if cell_type_labels:
            self.add_cell_type_labels(fontsize=fontsize)

        if neuropil_labels:
            self.add_neuropil_labels(fontsize=fontsize)

        if self.motion_decoder:
            self.add_decoder_sketch()

        self.add_arrows()

    def add_graph(
        self,
        edge_color_key: Optional[str] = None,
        arrows: bool = True,
        edge_alpha: float = 1.0,
        edge_width: float = 1.0,
        constant_edge_width: Optional[float] = 0.25,
        constant_edge_color: str = "#c5c5c5",
        edge_cmap: Optional[str] = None,
        nx_kwargs: Dict = {},
    ) -> None:
        """
        Add the graph to the figure.

        Args:
            edge_color_key: Key for edge color.
            arrows: Whether to add arrows to edges.
            edge_alpha: Alpha value for edges.
            edge_width: Width of edges.
            constant_edge_width: Constant width for all edges.
            constant_edge_color: Constant color for all edges.
            edge_cmap: Colormap for edges.
            nx_kwargs: Additional kwargs for networkx drawing.
        """

        def _network_graph(nodes, edges):
            """Transform graph from df to list to create networkx.Graph object."""
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

        (
            (lefts, bottoms, rights, tops),
            (
                centers,
                widths,
                height,
            ),
        ) = plt_utils.get_ax_positions(list(axes.values()))
        edge_ax = self.fig.add_axes([
            lefts.min(),
            bottoms.min(),
            rights.max() - lefts.min(),
            tops.max() - bottoms.min(),
        ])
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

        draw_networkx_edges(
            graph,
            pos=positions,
            ax=edge_ax,
            edge_color=np.array([edge_color[tuple(edge)] for edge in edge_list]),
            edge_cmap=edge_cmap,
            edge_vmin=edge_vmin,
            edge_vmax=edge_vmax,
            alpha=edge_alpha,
            arrows=arrows,
            arrowstyle=(
                "-|>, head_length=0.4, head_width=0.075, widthA=1.0, "
                "widthB=1.0, lengthA=0.2, lengthB=0.2"
            ),
            width=np.array([edge_width[tuple(edge)] for edge in edge_list]),
            **nx_kwargs,
        )
        self.edge_ax = edge_ax

    def add_retina_box(self):
        retina_node_types = valfilter(lambda v: v == "retina", self.layout)
        axes = {
            node_type: [ax for ax in self.axes if ax.get_label() == node_type][0]
            for node_type in retina_node_types
        }
        (
            (lefts, bottoms, rights, tops),
            (
                centers,
                widths,
                height,
            ),
        ) = plt_utils.get_ax_positions(list(axes.values()))
        retina_box_ax = self.fig.add_axes(
            [
                lefts.min(),
                bottoms.min(),
                rights.max() - lefts.min(),
                tops.max() - bottoms.min(),
            ],
            label="retina_box",
        )
        retina_box_ax.patch.set_alpha(0)
        plt_utils.rm_spines(retina_box_ax)
        self.ax_dict["retina box"] = retina_box_ax

    def add_decoded_box(self):
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

    def add_decoder_sketch(self):
        ax = self.ax_dict["motion decoder"]
        nodes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        edges = [
            (1, 11),
            (2, 10),
            (2, 12),
            (3, 10),
            (3, 11),
            (3, 12),
            (4, 11),
            (4, 12),
            (4, 14),
            (5, 10),
            (5, 12),
            (5, 13),
            (6, 11),
            (6, 13),
            (6, 14),
            (7, 12),
            (7, 14),
            (8, 13),
            (9, 14),
            (10, 15),
            (11, 16),
            (12, 15),
            (13, 15),
            (13, 16),
            (14, 16),
        ]
        graph = nx.Graph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        x = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]
        y = [9, 8, 7, 6, 5, 4, 3, 2, 1, 7.3, 6.3, 5.3, 4.3, 3.3, 5.7, 4.7]
        x, y, width, height = plt_utils.scale(x, y)
        nx.draw_networkx(
            graph,
            pos=dict(zip(nodes, zip(x, y))),
            node_shape="H",
            node_size=50,
            node_color="#5a5b5b",
            edge_color="#C5C4C4",
            width=0.25,
            with_labels=False,
            ax=ax,
            arrows=False,
        )
        plt_utils.rm_spines(ax)

    def add_arrows(self):
        def arrow_between_axes(axA, axB):
            # Create the arrow
            # 1. Get transformation operators for axis and figure
            ax0tr = axA.transAxes  # Axis 0 -> Display
            ax1tr = axB.transAxes  # Axis 1 -> Display
            figtr = self.fig.transFigure.inverted()  # Display -> Figure
            # 2. Transform arrow start point from axis 0 to figure coordinates
            ptA = figtr.transform(ax0tr.transform((1, 0.5)))
            # 3. Transform arrow end point from axis 1 to figure coordinates
            ptB = figtr.transform(ax1tr.transform((0, 0.5)))
            # 4. Create the patch
            arrow = matplotlib.patches.FancyArrowPatch(
                ptA,
                ptB,
                transform=self.fig.transFigure,  # Place arrow in figure coord system
                # fc=self.fontcolor,
                # ec=self.fontcolor,
                #     connectionstyle="arc3",
                arrowstyle="simple, head_width=3, head_length=6, tail_width=0.15",
                alpha=1,
                mutation_scale=1.0,
            )
            arrow.set_lw(0.25)
            # 5. Add patch to list of objects to draw onto the figure
            self.fig.patches.append(arrow)

        if self.video and self.rendering:
            arrow_between_axes(self.ax_dict["video"], self.ax_dict["rendering"])
            arrow_between_axes(self.ax_dict["rendering"], self.ax_dict["retina box"])
        elif self.video:
            arrow_between_axes(self.ax_dict["video"], self.ax_dict["retina box"])
        elif self.rendering:
            arrow_between_axes(self.ax_dict["rendering"], self.ax_dict["retina box"])

        if self.motion_decoder and self.decoded_motion:
            arrow_between_axes(
                self.ax_dict["decoded box"], self.ax_dict["motion decoder"]
            )
            arrow_between_axes(
                self.ax_dict["motion decoder"], self.ax_dict["decoded motion"]
            )
        elif self.motion_decoder:
            arrow_between_axes(
                self.ax_dict["decoded box"], self.ax_dict["motion decoder"]
            )
        elif self.decoded_motion:
            arrow_between_axes(
                self.ax_dict["decoded box"], self.ax_dict["decoded motion"]
            )

    def add_cell_type_labels(self, fontsize=5):
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

    def add_neuropil_labels(self, fontsize=5):
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
        (
            (lefts, bottoms, rights, tops),
            (
                centers,
                widths,
                height,
            ),
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
        (
            (lefts, bottoms, rights, tops),
            (
                centers,
                widths,
                height,
            ),
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
    layout: Dict[str, str],
    cell_types: Optional[List[str]] = None,
    fig: Optional[plt.Figure] = None,
    figsize: List[int] = [16, 10],
    types_per_column: int = 8,
    region_spacing: int = 2,
    wspace: float = 0,
    hspace: float = 0,
    as_dict: bool = False,
    pos: Optional[Dict[str, List[float]]] = None,
) -> Tuple[
    plt.Figure, Union[List[plt.Axes], Dict[str, plt.Axes]], Dict[str, List[float]]
]:
    """
    Create axes for network layout.

    Args:
        layout: Dictionary mapping node types to layout positions.
        cell_types: List of cell types to include.
        fig: Existing figure to use.
        figsize: Size of the figure.
        types_per_column: Number of types per column.
        region_spacing: Spacing between regions.
        wspace: Width space between subplots.
        hspace: Height space between subplots.
        as_dict: Whether to return axes as a dictionary.
        pos: Pre-computed positions for nodes.

    Returns:
        Tuple containing the figure, axes, and node positions.
    """
    fig = fig or plt.figure(figsize=figsize)

    pos = pos or _network_graph_node_pos(
        layout, region_spacing=region_spacing, types_per_column=types_per_column
    )
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
    fig, axes, xy_scaled = plt_utils.ax_scatter(
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


def _network_graph_node_pos(
    layout: Dict[str, str], region_spacing: float = 2, types_per_column: int = 8
) -> Dict[str, List[float]]:
    """
    Compute (x, y) coordinates for nodes in a network graph.

    Args:
        layout: Dictionary mapping node types to layout positions.
        region_spacing: Spacing between regions.
        types_per_column: Number of types per column.

    Returns:
        Dictionary mapping node types to their (x, y) coordinates.

    Note:
        Special nodes like 'video', 'rendering', etc. are positioned at the middle
        y-coordinate of their respective columns.
    """
    x_coordinate = 0
    region_0 = "retina"
    pos = {}
    j = 0
    special_nodes = [
        "video",
        "rendering",
        "motion decoder",
        "decoded motion",
        "pixel-accurate motion",
    ]

    for typ in layout:
        if typ in special_nodes:
            region_spacing = 1.25
        if layout[typ] != region_0:
            x_coordinate += region_spacing
            j = 0
        elif (j % types_per_column) == 0 and j != 0:
            x_coordinate += 1
        y_coordinate = types_per_column - 1 - j % types_per_column
        pos[typ] = [x_coordinate, y_coordinate]
        region_0 = layout[typ]
        j += 1

    y_mid = (types_per_column - 1) / 2

    for node in special_nodes:
        if node in layout:
            pos[node][1] = y_mid

    if "pixel-accurate motion" in layout:
        pos["pixel-accurate motion"][1] = y_mid - 1.5

    return pos


### Overriding some networkx functions to make selfloop height and width controllable
class FancyArrowFactory:
    """Draw arrows with `matplotlib.patches.FancyarrowPatch`"""

    class ConnectionStyleFactory:
        def __init__(self, connectionstyles, selfloop_height, ax=None):
            import matplotlib as mpl
            import matplotlib.path  # call as mpl.path
            import numpy as np

            self.ax = ax
            self.mpl = mpl
            self.np = np
            self.base_connection_styles = [
                mpl.patches.ConnectionStyle(cs) for cs in connectionstyles
            ]
            self.n = len(self.base_connection_styles)
            self.selfloop_height = selfloop_height

        def curved(self, edge_index):
            return self.base_connection_styles[edge_index % self.n]

        def self_loop(
            self,
            edge_index,
            horizontal_shift=None,
            vertical_shift=None,
            origin_x=0,
            origin_y=0,
        ):
            def self_loop_connection(posA, posB, *args, **kwargs):
                if not self.np.all(posA == posB):
                    raise nx.NetworkXError(
                        "`self_loop` connection style method"
                        "is only to be used for self-loops"
                    )
                # this is called with _screen space_ values
                # so convert back to data space
                data_loc = self.ax.transData.inverted().transform(posA)
                v_shift = vertical_shift or 0.1 * self.selfloop_height
                h_shift = horizontal_shift or v_shift * 0.5
                # put the top of the loop first so arrow is not hidden by node
                path = self.np.asarray([
                    # 1
                    [origin_x, v_shift],
                    # 4 4 4
                    [h_shift, v_shift],
                    [h_shift, origin_y],
                    [origin_x, origin_y],
                    # 4 4 4
                    [-h_shift, origin_y],
                    [-h_shift, v_shift],
                    [origin_x, v_shift],
                ])
                # Rotate self loop 90 deg. if more than 1
                # This will allow for maximum of 4 visible self loops
                if edge_index % 4:
                    x, y = path.T
                    for _ in range(edge_index % 4):
                        x, y = y, -x
                    path = self.np.array([x, y]).T
                return self.mpl.path.Path(
                    self.ax.transData.transform(data_loc + path), [1, 4, 4, 4, 4, 4, 4]
                )

            return self_loop_connection

    def __init__(
        self,
        edge_pos,
        edgelist,
        nodelist,
        edge_indices,
        node_size,
        selfloop_height,
        connectionstyle="arc3",
        node_shape="o",
        arrowstyle="-",
        arrowsize=10,
        edge_color="k",
        alpha=None,
        linewidth=1.0,
        style="solid",
        min_source_margin=0,
        min_target_margin=0,
        ax=None,
    ):
        import matplotlib as mpl
        import matplotlib.patches  # call as mpl.patches
        import numpy as np

        if isinstance(connectionstyle, str):
            connectionstyle = [connectionstyle]
        elif np.iterable(connectionstyle):
            connectionstyle = list(connectionstyle)
        else:
            msg = "ConnectionStyleFactory arg `connectionstyle` must be str or iterable"
            raise nx.NetworkXError(msg)
        self.ax = ax
        self.mpl = mpl
        self.np = np
        self.edge_pos = edge_pos
        self.edgelist = edgelist
        self.nodelist = nodelist
        self.node_shape = node_shape
        self.min_source_margin = min_source_margin
        self.min_target_margin = min_target_margin
        self.edge_indices = edge_indices
        self.node_size = node_size
        self.connectionstyle_factory = self.ConnectionStyleFactory(
            connectionstyle, selfloop_height, ax
        )
        self.arrowstyle = arrowstyle
        self.arrowsize = arrowsize
        self.arrow_colors = mpl.colors.colorConverter.to_rgba_array(edge_color, alpha)
        self.linewidth = linewidth
        self.style = style
        if isinstance(arrowsize, list) and len(arrowsize) != len(edge_pos):
            raise ValueError("arrowsize should have the same length as edgelist")

    def __call__(self, i, h_shift=None, v_shift=None, x0=0, y0=0):
        (x1, y1), (x2, y2) = self.edge_pos[i]
        shrink_source = 0  # space from source to tail
        shrink_target = 0  # space from  head to target
        if self.np.iterable(self.node_size):  # many node sizes
            source, target = self.edgelist[i][:2]
            source_node_size = self.node_size[self.nodelist.index(source)]
            target_node_size = self.node_size[self.nodelist.index(target)]
            shrink_source = self.to_marker_edge(source_node_size, self.node_shape)
            shrink_target = self.to_marker_edge(target_node_size, self.node_shape)
        else:
            shrink_source = self.to_marker_edge(self.node_size, self.node_shape)
            shrink_target = shrink_source
        shrink_source = max(shrink_source, self.min_source_margin)
        shrink_target = max(shrink_target, self.min_target_margin)

        # scale factor of arrow head
        if isinstance(self.arrowsize, list):
            mutation_scale = self.arrowsize[i]
        else:
            mutation_scale = self.arrowsize

        if len(self.arrow_colors) > i:
            arrow_color = self.arrow_colors[i]
        elif len(self.arrow_colors) == 1:
            arrow_color = self.arrow_colors[0]
        else:  # Cycle through colors
            arrow_color = self.arrow_colors[i % len(self.arrow_colors)]

        if self.np.iterable(self.linewidth):
            if len(self.linewidth) > i:
                linewidth = self.linewidth[i]
            else:
                linewidth = self.linewidth[i % len(self.linewidth)]
        else:
            linewidth = self.linewidth

        if (
            self.np.iterable(self.style)
            and not isinstance(self.style, str)
            and not isinstance(self.style, tuple)
        ):
            if len(self.style) > i:
                linestyle = self.style[i]
            else:  # Cycle through styles
                linestyle = self.style[i % len(self.style)]
        else:
            linestyle = self.style

        if x1 == x2 and y1 == y2:
            connectionstyle = self.connectionstyle_factory.self_loop(
                self.edge_indices[i],
                horizontal_shift=h_shift,
                vertical_shift=v_shift,
                origin_x=x0,
                origin_y=y0,
            )
        else:
            connectionstyle = self.connectionstyle_factory.curved(self.edge_indices[i])
        return self.mpl.patches.FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=self.arrowstyle,
            shrinkA=shrink_source,
            shrinkB=shrink_target,
            mutation_scale=mutation_scale,
            color=arrow_color,
            linewidth=linewidth,
            connectionstyle=connectionstyle,
            linestyle=linestyle,
            zorder=1,  # arrows go behind nodes
        )

    def to_marker_edge(self, marker_size, marker):
        if marker in "s^>v<d":  # `large` markers need extra space
            return self.np.sqrt(2 * marker_size) / 2
        else:
            return self.np.sqrt(marker_size) / 2


def draw_networkx_edges(
    G,
    pos,
    edgelist=None,
    width=1.0,
    edge_color="k",
    style="solid",
    alpha=None,
    arrowstyle=None,
    arrowsize=10,
    edge_cmap=None,
    edge_vmin=None,
    edge_vmax=None,
    ax=None,
    arrows=None,
    label=None,
    node_size=300,
    nodelist=None,
    node_shape="o",
    connectionstyle="arc3",
    min_source_margin=0,
    min_target_margin=0,
    hide_ticks=True,
    selfloop_x0=0,
    selfloop_y0=0,
    selfloop_h_shift=None,
    selfloop_v_shift=None,
):
    r"""Draw the edges of the graph G.

    This draws only the edges of the graph G.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edgelist : collection of edge tuples (default=G.edges())
        Draw only specified edges

    width : float or array of floats (default=1.0)
        Line width of edges

    edge_color : color or array of colors (default='k')
        Edge color. Can be a single color or a sequence of colors with the same
        length as edgelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    style : string or array of strings (default='solid')
        Edge line style e.g.: '-', '--', '-.', ':'
        or words like 'solid' or 'dashed'.
        Can be a single style or a sequence of styles with the same
        length as the edge list.
        If less styles than edges are given the styles will cycle.
        If more styles than edges are given the styles will be used sequentially
        and not be exhausted.
        Also, `(offset, onoffseq)` tuples can be used as style instead of a strings.
        (See `matplotlib.patches.FancyArrowPatch`: `linestyle`)

    alpha : float or array of floats (default=None)
        The edge transparency.  This can be a single alpha value,
        in which case it will be applied to all specified edges. Otherwise,
        if it is an array, the elements of alpha will be applied to the colors
        in order (cycling through alpha multiple times if necessary).

    edge_cmap : Matplotlib colormap, optional
        Colormap for mapping intensities of edges

    edge_vmin,edge_vmax : floats, optional
        Minimum and maximum for edge colormap scaling

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    arrows : bool or None, optional (default=None)
        If `None`, directed graphs draw arrowheads with
        `~matplotlib.patches.FancyArrowPatch`, while undirected graphs draw edges
        via `~matplotlib.collections.LineCollection` for speed.
        If `True`, draw arrowheads with FancyArrowPatches (bendable and stylish).
        If `False`, draw edges using LineCollection (linear and fast).

        Note: Arrowheads will be the same color as edges.

    arrowstyle : str (default='-\|>' for directed graphs)
        For directed graphs and `arrows==True` defaults to '-\|>',
        For undirected graphs default to '-'.

        See `matplotlib.patches.ArrowStyle` for more options.

    arrowsize : int (default=10)
        For directed graphs, choose the size of the arrow head's length and
        width. See `matplotlib.patches.FancyArrowPatch` for attribute
        `mutation_scale` for more info.

    connectionstyle : string or iterable of strings (default="arc3")
        Pass the connectionstyle parameter to create curved arc of rounding
        radius rad. For example, connectionstyle='arc3,rad=0.2'.
        See `matplotlib.patches.ConnectionStyle` and
        `matplotlib.patches.FancyArrowPatch` for more info.
        If Iterable, index indicates i'th edge key of MultiGraph

    node_size : scalar or array (default=300)
        Size of nodes. Though the nodes are not drawn with this function, the
        node size is used in determining edge positioning.

    nodelist : list, optional (default=G.nodes())
       This provides the node order for the `node_size` array (if it is an array).

    node_shape :  string (default='o')
        The marker used for nodes, used in determining edge positioning.
        Specification is as a `matplotlib.markers` marker, e.g. one of 'so^>v<dph8'.

    label : None or string
        Label for legend

    min_source_margin : int (default=0)
        The minimum margin (gap) at the beginning of the edge at the source.

    min_target_margin : int (default=0)
        The minimum margin (gap) at the end of the edge at the target.

    hide_ticks : bool, optional
        Hide ticks of axes. When `True` (the default), ticks and ticklabels
        are removed from the axes. To set ticks and tick labels to the pyplot default,
        use ``hide_ticks=False``.

    Returns
    -------
     matplotlib.collections.LineCollection or a list of matplotlib.patches.FancyArrowPatch
        If ``arrows=True``, a list of FancyArrowPatches is returned.
        If ``arrows=False``, a LineCollection is returned.
        If ``arrows=None`` (the default), then a LineCollection is returned if
        `G` is undirected, otherwise returns a list of FancyArrowPatches.

    Notes
    -----
    For directed graphs, arrows are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False or by passing an arrowstyle without
    an arrow on the end.

    Be sure to include `node_size` as a keyword argument; arrows are
    drawn considering the size of nodes.

    Self-loops are always drawn with `~matplotlib.patches.FancyArrowPatch`
    regardless of the value of `arrows` or whether `G` is directed.
    When ``arrows=False`` or ``arrows=None`` and `G` is undirected, the
    FancyArrowPatches corresponding to the self-loops are not explicitly
    returned. They should instead be accessed via the ``Axes.patches``
    attribute (see examples).

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edges = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))

    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 3)])
    >>> arcs = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    >>> alphas = [0.3, 0.4, 0.5]
    >>> for i, arc in enumerate(arcs):  # change alpha values of arcs
    ...     arc.set_alpha(alphas[i])

    The FancyArrowPatches corresponding to self-loops are not always
    returned, but can always be accessed via the ``patches`` attribute of the
    `matplotlib.Axes` object.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> G = nx.Graph([(0, 1), (0, 0)])  # Self-loop at node 0
    >>> edge_collection = nx.draw_networkx_edges(G, pos=nx.circular_layout(G), ax=ax)
    >>> self_loop_fap = ax.patches[0]

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_labels
    draw_networkx_edge_labels

    """
    import warnings

    import matplotlib as mpl
    import matplotlib.collections  # call as mpl.collections
    import matplotlib.colors  # call as mpl.colors
    import matplotlib.pyplot as plt
    import numpy as np

    # The default behavior is to use LineCollection to draw edges for
    # undirected graphs (for performance reasons) and use FancyArrowPatches
    # for directed graphs.
    # The `arrows` keyword can be used to override the default behavior
    if arrows is None:
        use_linecollection = not (G.is_directed() or G.is_multigraph())
    else:
        if not isinstance(arrows, bool):
            raise TypeError("Argument `arrows` must be of type bool or None")
        use_linecollection = not arrows

    if isinstance(connectionstyle, str):
        connectionstyle = [connectionstyle]
    elif np.iterable(connectionstyle):
        connectionstyle = list(connectionstyle)
    else:
        msg = "draw_networkx_edges arg `connectionstyle` must be str or iterable"
        raise nx.NetworkXError(msg)

    # Some kwargs only apply to FancyArrowPatches. Warn users when they use
    # non-default values for these kwargs when LineCollection is being used
    # instead of silently ignoring the specified option
    if use_linecollection:
        msg = (
            "\n\nThe {0} keyword argument is not applicable when drawing edges\n"
            "with LineCollection.\n\n"
            "To make this warning go away, either specify `arrows=True` to\n"
            "force FancyArrowPatches or use the default values.\n"
            "Note that using FancyArrowPatches may be slow for large graphs.\n"
        )
        if arrowstyle is not None:
            warnings.warn(msg.format("arrowstyle"), category=UserWarning, stacklevel=2)
        if arrowsize != 10:
            warnings.warn(msg.format("arrowsize"), category=UserWarning, stacklevel=2)
        if min_source_margin != 0:
            warnings.warn(
                msg.format("min_source_margin"), category=UserWarning, stacklevel=2
            )
        if min_target_margin != 0:
            warnings.warn(
                msg.format("min_target_margin"), category=UserWarning, stacklevel=2
            )
        if any(cs != "arc3" for cs in connectionstyle):
            warnings.warn(
                msg.format("connectionstyle"), category=UserWarning, stacklevel=2
            )

    # NOTE: Arrowstyle modification must occur after the warnings section
    if arrowstyle is None:
        arrowstyle = "-|>" if G.is_directed() else "-"

    if ax is None:
        ax = plt.gca()

    if edgelist is None:
        edgelist = list(G.edges)  # (u, v, k) for multigraph (u, v) otherwise

    if len(edgelist):
        if G.is_multigraph():
            key_count = collections.defaultdict(lambda: itertools.count(0))
            edge_indices = [next(key_count[tuple(e[:2])]) for e in edgelist]
        else:
            edge_indices = [0] * len(edgelist)
    else:  # no edges!
        return []

    if nodelist is None:
        nodelist = list(G.nodes())

    # FancyArrowPatch handles color=None different from LineCollection
    if edge_color is None:
        edge_color = "k"

    # set edge positions
    edge_pos = np.asarray([(pos[e[0]], pos[e[1]]) for e in edgelist])

    # Check if edge_color is an array of floats and map to edge_cmap.
    # This is the only case handled differently from matplotlib
    if (
        np.iterable(edge_color)
        and (len(edge_color) == len(edge_pos))
        and np.all([isinstance(c, Number) for c in edge_color])
    ):
        if edge_cmap is not None:
            assert isinstance(edge_cmap, mpl.colors.Colormap)
        else:
            edge_cmap = plt.get_cmap()
        if edge_vmin is None:
            edge_vmin = min(edge_color)
        if edge_vmax is None:
            edge_vmax = max(edge_color)
        color_normal = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
        edge_color = [edge_cmap(color_normal(e)) for e in edge_color]

    # compute initial view
    minx = np.amin(np.ravel(edge_pos[:, :, 0]))
    maxx = np.amax(np.ravel(edge_pos[:, :, 0]))
    miny = np.amin(np.ravel(edge_pos[:, :, 1]))
    maxy = np.amax(np.ravel(edge_pos[:, :, 1]))
    w = maxx - minx
    h = maxy - miny

    # Self-loops are scaled by view extent, except in cases the extent
    # is 0, e.g. for a single node. In this case, fall back to scaling
    # by the maximum node size
    selfloop_height = h if h != 0 else 0.005 * np.array(node_size).max()
    fancy_arrow_factory = FancyArrowFactory(
        edge_pos,
        edgelist,
        nodelist,
        edge_indices,
        node_size,
        selfloop_height,
        connectionstyle,
        node_shape,
        arrowstyle,
        arrowsize,
        edge_color,
        alpha,
        width,
        style,
        min_source_margin,
        min_target_margin,
        ax=ax,
    )

    # Draw the edges
    if use_linecollection:
        edge_collection = mpl.collections.LineCollection(
            edge_pos,
            colors=edge_color,
            linewidths=width,
            antialiaseds=(1,),
            linestyle=style,
            alpha=alpha,
        )
        edge_collection.set_cmap(edge_cmap)
        edge_collection.set_clim(edge_vmin, edge_vmax)
        edge_collection.set_zorder(1)  # edges go behind nodes
        edge_collection.set_label(label)
        ax.add_collection(edge_collection)
        edge_viz_obj = edge_collection

        # Make sure selfloop edges are also drawn
        # ---------------------------------------
        selfloops_to_draw = [loop for loop in nx.selfloop_edges(G) if loop in edgelist]
        if selfloops_to_draw:
            edgelist_tuple = list(map(tuple, edgelist))
            arrow_collection = []
            for loop in selfloops_to_draw:
                i = edgelist_tuple.index(loop)
                arrow = fancy_arrow_factory(
                    i,
                    h_shift=selfloop_h_shift,
                    v_shift=selfloop_v_shift,
                    x0=selfloop_x0,
                    y0=selfloop_y0,
                )
                arrow_collection.append(arrow)
                ax.add_patch(arrow)
    else:
        edge_viz_obj = []
        for i in range(len(edgelist)):
            arrow = fancy_arrow_factory(
                i,
                h_shift=selfloop_h_shift,
                v_shift=selfloop_v_shift,
                x0=selfloop_x0,
                y0=selfloop_y0,
            )
            ax.add_patch(arrow)
            edge_viz_obj.append(arrow)

    # update view after drawing
    padx, pady = 0.05 * w, 0.05 * h
    corners = (minx - padx, miny - pady), (maxx + padx, maxy + pady)
    ax.update_datalim(corners)
    ax.autoscale_view()

    if hide_ticks:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )

    return edge_viz_obj
