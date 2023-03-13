""" Defines the `Connectome` Directory type """

import json
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from toolz import groupby, valmap
import matplotlib.path as mp
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.figure import Figure
from matplotlib.axis import Axis
from matplotlib.colors import Colormap
from matplotlib.colorbar import Colorbar
import numpy as np
from pandas import DataFrame
import networkx as nx

from datamate import Directory, Namespace, ArrayFile, root


import flyvision
from flyvision.utils import nodes_edges_utils, df_utils, hex_utils
from flyvision.plots import plots, plt_utils


__all__ = ["Connectome", "ConnectomeView"]

# -- `Connectome` --------------------------------------------------------------
@root(flyvision.root_dir / "connectome")
class Connectome(Directory):
    """A graph with nodes (cells) and edges (synapse sets connecting them)

    Example:
            config = Namespace(file='fib25-fib19.json',
                         extent=15,
                         n_syn_fill=1)
            cc = Connectome(config)
    """

    class Config:
        file: str
        "The name of a JSON connectome file"
        extent: int
        "The array radius, in columns"
        n_syn_fill: int
        "The number of synapses to assume in data gaps"
        allow_autapses: bool
        "Whether autapses are allowed, eg when filling data gaps."

    # -- Contents ------------------------------------------
    nodes: Directory
    "A table with a row for each node"
    central_nodes_index: ArrayFile
    "Index of the central node of a unique node type in the nodes table"
    edges: Directory
    "A table with a row for each edge"
    input_node_types: ArrayFile
    "The node types to use as inputs"
    intermediate_node_types: ArrayFile
    "The hidden node types"
    output_node_types: ArrayFile
    "The node types to use for task readout"
    layout: ArrayFile
    "Input, hidden, output layout for later circuit visualization"
    unique_node_types: ArrayFile
    "A list of all node types"

    # ------------------------------------------------------

    """
    Attributes:

        unique_node_types (array ofs tr)
        input_node_types (array of str)
        intermediate_node_types (array of str)
        output_node_types (array of str)
        central_nodes_index (array of int)

        nodes (Directory)

            Attributes:

                layer_index (Directory): Indices of all nodes of each node type in the table of nodes

                    Attributes:
                        [*unique_node_types] (array of int): indices for node fields defining a layer.

            Fields:
                - "type": `string` // cell type name
                - "u": `int` // location coordinate #1 (oblique coordinates)
                - "v": `int` // location coordinate #2 (oblique coordinates)
                - "role": 'string'

        edges (Directory)

            Attributes:

            Fields:
                - "source_index": `int` // presynaptic cell
                - "target_index": `int` // postsynaptic cell
                - "sign": `int` // +1 (excitatory) or -1 (inhibitory)
                - "n_syn": `number` // synapse count
                - `<dereferenced_source_and_target_fields>...` // for convenience

    Note:
        A connectome can be constructed from a JSON model file following this
        schema:.connectome import *

            {
                "nodes": [{
                    "name": string,
                    "pattern": (
                        ["stride", [<u_stride:int>, <v_stride:int>]]
                        | ["tile", <stride:int>]
                        | ["single", null]
                    )
                }*],
                "edges": [{
                    "src": string,
                    "tar": string,
                    "alpha": int,
                    "offsets": [[
                        [<du:int>, <dv:int>],
                        <n_synapses:number>
                    ]*]
                }*]
            }

        See "data/_fib25.json" for an example.
    """

    def __init__(self, config: Config) -> None:
        # Load the connectome spec.
        spec = json.loads(Path(self.path.parent / config.file).read_text())

        # Store unique node types.
        self.unique_node_types = np.string_([n["name"] for n in spec["nodes"]])
        self.input_node_types = np.string_(spec["input_units"])
        self.output_node_types = np.string_(spec["output_units"])
        intermediate_node_types, _ = nodes_edges_utils.order_nodes_list(
            np.array(
                list(
                    set(self.unique_node_types)
                    - set(self.input_node_types)
                    - set(self.output_node_types)
                )
            ).astype(str)
        )
        self.intermediate_node_types = np.array(intermediate_node_types).astype("S")

        layout = []
        layout.extend(
            list(
                zip(
                    self.input_node_types,
                    [b"retina" for _ in range(len(self.input_node_types))],
                )
            )
        )
        layout.extend(
            list(
                zip(
                    self.intermediate_node_types,
                    [b"intermediate" for _ in range(len(self.intermediate_node_types))],
                )
            )
        )
        layout.extend(
            list(
                zip(
                    self.output_node_types,
                    [b"output" for _ in range(len(self.output_node_types))],
                )
            )
        )
        self.layout = np.string_(layout)

        # Construct nodes and edges.
        nodes: List[Node] = []
        edges: List[Edge] = []
        add_nodes(nodes, spec["nodes"], config.extent)
        add_edges(
            edges,
            nodes,
            spec["edges"],
            config.n_syn_fill,
            getattr(config, "allow_autapses", True),
        )

        # Define node roles (input, intermediate, output).
        _role = {node: "intermediate" for node in set([n.type for n in nodes])}
        _role.update({node: "input" for node in _role if node in spec["input_units"]})
        _role.update({node: "output" for node in _role if node in spec["output_units"]})

        # Store the graph.
        self.nodes = dict(  # type: ignore
            type=np.string_([n.type for n in nodes]),
            u=np.int32([n.u for n in nodes]),
            v=np.int32([n.v for n in nodes]),
            role=np.string_([_role[n.type] for n in nodes]),
        )

        self.edges = dict(  # type: ignore
            # [Essential fields]
            source_index=np.int64([e.source.id for e in edges]),
            target_index=np.int64([e.target.id for e in edges]),
            sign=np.float32([e.sign for e in edges]),
            n_syn=np.float32([e.n_syn for e in edges]),
            # [Convenience fields]
            source_type=np.string_([e.source.type for e in edges]),
            target_type=np.string_([e.target.type for e in edges]),
            source_u=np.int32([e.source.u for e in edges]),
            target_u=np.int32([e.target.u for e in edges]),
            source_v=np.int32([e.source.v for e in edges]),
            target_v=np.int32([e.target.v for e in edges]),
            du=np.int32([e.target.u - e.source.u for e in edges]),
            dv=np.int32([e.target.v - e.source.v for e in edges]),
            edge_type=np.string_([e.type for e in edges]),
            n_syn_certainty=np.float32([e.n_syn_certainty for e in edges]),
        )

        # Store central indices.
        self.central_nodes_index = np.int64(
            np.nonzero((self.nodes.u[:] == 0) & (self.nodes.v[:] == 0))[0]
        )

        # Store layer indices.
        layer_index = {}
        for node_type in self.unique_node_types[:]:
            node_indices = np.nonzero(self.nodes["type"][:] == node_type)[0]
            layer_index[node_type.decode()] = np.int64(node_indices)
        self.nodes.layer_index = layer_index


# -- Node construction ---------------------------------------------------------


@dataclass
class Node:
    id: int
    "index (0..n_nodes)"
    type: str
    "cell type name"
    u: int
    "location coordinate #1 (oblique coordinates)"
    v: int
    "location coordinate #2 (oblique coordinates)"
    u_stride: int
    "stride in u"
    v_stride: int
    "stride in v"


def add_nodes(seq: List[Node], node_spec: dict, extent: int) -> None:
    """Add nodes to `seq`, based on `node_spec`."""
    for n in node_spec:
        typ, (pattern, args) = n["name"], n["pattern"]
        # if typ == "Lawf2":
        #     breakpoint()
        if pattern == "stride":
            add_strided_nodes(seq, typ, extent, args)
        elif pattern == "tile":
            add_tiled_nodes(seq, typ, extent, args)
        elif pattern == "single":
            add_single_node(seq, typ, extent)


def add_strided_nodes(
    seq: List[Node], typ: str, extent: int, strides: Tuple[int, int]
) -> None:
    """
    Add to `seq` a population of neurons arranged in a hexagonal grid.
    """
    n = extent
    u_stride, v_stride = strides
    for u in range(-n, n + 1):
        for v in range(max(-n, -n - u), min(n, n - u) + 1):
            if u % u_stride == 0 and v % v_stride == 0:
                seq.append(Node(len(seq), typ, u, v, u_stride, v_stride))


def add_tiled_nodes(seq: List[Node], typ: str, extent: int, n: int) -> None:
    """Add to `seq` a population of neurons with strides `(n, n)`."""
    add_strided_nodes(seq, typ, extent, (n, n))


def add_single_node(seq: List[Node], typ: str, extent: int) -> None:
    """Add to `seq` a single-neuron population."""
    add_strided_nodes(seq, typ, 0, (1, 1))


# -- Edge construction ---------------------------------------------------------


@dataclass
class Edge:
    id: int
    "index (0..n_edges)"
    source: Node
    "presynaptic cell"
    target: Node
    "postsynaptic cell"
    sign: int
    "+1 (excitatory) or -1 (inhibitory)"
    n_syn: float
    "synapse count"
    type: str
    "synapse type"
    n_syn_certainty: float


def add_edges(
    seq: List[Edge],
    nodes: List[Node],
    edge_spec: dict,
    n_syn_fill: float,
    allow_autapses: bool,
) -> None:
    """
    Add edges to `seq` based on `edge_spec`.
    """
    node_index = {
        **groupby(lambda n: n.type, nodes),
        **groupby(lambda n: (n.type, n.u, n.v), nodes),
    }
    for e in edge_spec:
        # if e['src'] == "Lawf2" and e["tar"] == "Lawf2":
        # breakpoint()
        offsets = (
            fill_hull(e["offsets"], n_syn_fill)
            if n_syn_fill > 0 and len(e["offsets"]) >= 3
            else e["offsets"]
        )
        add_conv_edges(
            seq,
            node_index,
            e["src"],
            e["tar"],
            e["alpha"],
            offsets,
            e["edge_type"],
            e["lambda_mult"],
            allow_autapses,
        )


def fill_hull(offsets: List[list], n_syn_fill: float) -> List[list]:
    """
    Fill in the convex hull of the reported edges.
    """
    # to save time at library import
    import scipy.spatial as ss

    # overide provided n_syn_fill value with smallest existing value
    # min_n_syn = np.min([offset[1] for offset in offsets])
    # if min_n_syn < n_syn_fill:
    #     n_syn_fill = min_n_syn

    # Collect the points (column offsets, (du, dv)) reported as edges.
    known_pts = np.array([offset[0] for offset in offsets])
    known_pts_as_set = set(map(tuple, known_pts))

    # Compute the convex hull of the reported edges.
    hull = ss.ConvexHull((1 + 1e-6) * known_pts, False, "QJ")
    hull_vertices = known_pts[hull.vertices]

    # Find the points within the convex hull.
    grid = np.concatenate(
        np.dstack(
            np.mgrid[
                known_pts[:, 0].min() : known_pts[:, 0].max() + 1,
                known_pts[:, 1].min() : known_pts[:, 1].max() + 1,
            ]
        )
    )
    contained_pts = grid[mp.Path(hull_vertices).contains_points(grid)]

    # Add unknown points within the convex hull as offsets.
    return offsets + [
        [[u, v], n_syn_fill] for u, v in contained_pts if (u, v) not in known_pts_as_set
    ]


def add_conv_edges(
    seq: List[Edge],
    node_index: dict,
    source_typ: str,
    target_typ: str,
    sign: int,
    offsets: List[list],
    type: str,
    n_syn_certainty: float,
    allow_autapses: bool,
) -> None:
    """
    Construct a connection set with convolutional weight symmetry.
    """
    allow_central = (source_typ != target_typ) or allow_autapses

    for (du, dv), n_syn in offsets:
        for src in node_index.get(source_typ, []):
            u_tgt = src.u + du
            v_tgt = src.v + dv
            with suppress(KeyError):
                tgt = node_index[target_typ, u_tgt, v_tgt][0]
                # if src.type == "Lawf2" and tgt.type == "Lawf2":
                #     breakpoint()
                if not allow_central and du == 0 and dv == 0:
                    continue
                seq.append(Edge(len(seq), src, tgt, sign, n_syn, type, n_syn_certainty))


# ---- ConnectomeView


class ConnectomeView:
    """Wrapper for plots of the connectome.

    Args:
        ctome (Connectome): Directory of the connectome.
        groups (List[str]): regular expressions to sort the nodes by.

    Parameters:
        ctome (Connectome): ctome of connectome.
        nodes (DataFrame): node table.
        edges (DataFrame): edge table.

    Note: differs between Connectome and NetworkWrap. NetworkWraps can
        contain a 'prior_param_api' and a 'param_api' in which case the
        DataFrames are populated with prior and trained parameter columns.
    """

    def __init__(
        self,
        ctome: Connectome,
        groups=[
            r"R\d",
            r"L\d",
            r"Lawf\d",
            r"A",
            r"C\d",
            r"CT\d.*",
            r"Mi\d{1,2}",
            r"T\d{1,2}.*",
            r"Tm.*\d{1,2}.*",
        ],
    ):
        self.ctome = ctome

        assert "nodes" in self.ctome and "edges" in self.ctome

        self.edges = self.ctome.edges

        self.nodes = self.ctome.nodes

        self.node_types_unsorted = self.ctome.unique_node_types[:].astype(str)

        (
            self.node_types_sorted,
            self.node_types_sort_index,
        ) = nodes_edges_utils.order_nodes_list(
            self.ctome.unique_node_types[:].astype(str), groups
        )

        self.layout = dict(self.ctome.layout[:].astype(str))
        self.node_indexer = nodes_edges_utils.NodeIndexer(self.ctome)

    # ---- CONNECTIVITY MATRIX

    def connectivity_matrix(
        self,
        plot_type: str = "n_syn",
        only_sign: Optional[str] = None,
        node_types: Optional[List[str]] = None,
        no_symlog: Optional[bool] = False,
        min_number: Optional[float] = None,
        cmap: Optional[Colormap] = None,
        size_scale: Optional[float] = None,
        title: Optional[str] = None,
        cbar_label: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Figure, Axis, Colorbar, np.ndarray]:
        """Plots the connectivity matrix as counts or weights.

        Args:
            plot_type: 'n_syn' referring to number of input synapses,
                    'count' referring to number of input neurons.
            only_sign: '+' for displaying only excitatory projections,
                    '-' for displaying only inhibitory projections.
            node_types: provide a subset of nodes to only display those.
            no_symlog: disable symetric log scale.
            size_scale: determines the size of the scattered squares.

        Note, kwargs are passed to the heatmap plot function.
        """

        _kwargs = dict(
            n_syn=dict(
                symlog=1e-5,
                grid=True,
                cmap=cmap or cm.get_cmap("seismic"),
                title=title or "Connectivity between identified cell types",
                cbar_label=cbar_label or r"$\pm\sum_{pre} N_\mathrm{syn.}^{pre, post}$",
                size_scale=size_scale or 0.05,
            ),
            count=dict(
                grid=True,
                cmap=cmap or cm.get_cmap("seismic"),
                midpoint=0,
                title=title or "Number of Input Neurons",
                cbar_label=cbar_label or r"$\sum_{pre} 1$",
                size_scale=size_scale or 0.05,
            ),
        )

        kwargs.update(_kwargs[plot_type])
        if no_symlog:
            kwargs.update(symlog=None)
            kwargs.update(midpoint=0)

        edges = self.edges.to_df()

        # to take projections onto central nodes (home columns) into account
        edges = edges[(edges.target_u == 0) & (edges.target_v == 0)]

        # filter edges to allow providing a subset of node types
        node_types = node_types or self.node_types_sorted
        edges = df_utils.filter_df_by_list(
            node_types,
            df_utils.filter_df_by_list(node_types, edges, column="source_type"),
            column="target_type",
        )
        weights = self._weights()[edges.index]

        # lookup table for key -> (i, j)
        type_index = {node_typ: i for i, node_typ in enumerate(node_types)}
        matrix = np.zeros([len(type_index), len(type_index)])

        for srctyp, tgttyp, weight in zip(
            edges.source_type.values, edges.target_type.values, weights
        ):
            if plot_type == "count":
                # to simply count the number of projections
                matrix[type_index[srctyp], type_index[tgttyp]] += 1
            elif plot_type in ["weight", "n_syn"]:
                # to sum the synapse counts
                matrix[type_index[srctyp], type_index[tgttyp]] += weight
            else:
                raise ValueError

        # to filter out all connections weaker than min_number
        if min_number is not None:
            matrix[np.abs(matrix) <= min_number] = np.nan

        # to display either only excitatory or inhibitory connections
        if only_sign == "+":
            matrix[matrix < 0] = 0
            kwargs.update(symlog=None, midpoint=0)
        elif only_sign == "-":
            matrix[matrix > 0] = 0
            kwargs.update(symlog=None, midpoint=0)
        elif only_sign is None:
            pass
        else:
            raise ValueError

        fig, axes, cbar, matrix = plots.heatmap(matrix, node_types, **kwargs)
        return fig

    def _weights(self):
        return self.edges.sign[:] * self.edges.n_syn[:]

    # ---- NETWORK GRAPHS

    def network_layout(
        self,
        node_types=None,
        edge_color="#c5c5c5",
        edge_width=0.25,
        max_extent=5,
        fig=None,
        **kwargs,
    ):
        """Show hexagonal lattice columnar organization of the network."""

        fig = fig or plt.figure(figsize=[20, 10])

        if node_types:
            nodes = df_utils.filter_df_by_list(
                node_types, self.nodes.to_df(), column="type"
            )
            edges = df_utils.filter_df_by_list(
                node_types, self.edges.to_df(), column="source_type"
            )
            edges = df_utils.filter_df_by_list(
                node_types, edges.to_df(), column="target_type"
            )
            node_types = node_types
        else:
            nodes = self.nodes.to_df()
            edges = self.edges.to_df()
            node_types = self.node_types_sorted

        hpad = 0.05
        wpad = 0.05
        pos = plots._network_graph_node_pos(self.layout)
        pos = {key: value for key, value in pos.items() if key in node_types}
        xy = np.array([(x, y) for (x, y) in pos.values()])

        fig, axes, xy_scaled = plt_utils.regular_ax_scatter(
            xy[:, 0], xy[:, 1], fig=fig, hpad=hpad, wpad=wpad, alpha=0
        )
        _ = self.hex_layout_all(
            node_types=node_types,
            anatomic_order=True,
            fig=fig,
            axes=axes,
            max_extent=max_extent,
            labelxy=(-0.05, 1.05),
            fontsize=14,
            **kwargs,
        )

        new_pos = {key: xy_scaled[i] for i, key in enumerate(pos.keys())}

        nodes, vertices = plots._network_graph(nodes, edges)
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(vertices)
        edge_ax = fig.add_axes([0, 0, 1, 1], label="edge axis")
        edge_ax.set_zorder(0)
        edge_ax = plt_utils.rm_spines(edge_ax, rm_xticks=True, rm_yticks=True)
        edge_ax.patch.set_alpha(0.0)

        nx.draw_networkx_edges(
            graph,
            pos=new_pos,
            ax=edge_ax,
            width=edge_width,
            edge_color=edge_color,
            arrows=True,
        )

        return fig

    def hex_layout(
        self,
        node_type,
        edgecolor="black",
        edgewidth=0.5,
        alpha=1,
        fill=True,
        max_extent=5,
        cmap=cm.get_cmap("binary"),
        fig=None,
        ax=None,
        **kwargs,
    ):
        """
        Plot the layout of a node type on a regular hex grid.
        """
        nodes = self.nodes.to_df()
        node_condition = nodes.type == node_type
        u, v = nodes.u[node_condition], nodes.v[node_condition]
        max_extent = hex_utils.get_extent(u, v) if max_extent is None else max_extent
        extent_condition = (
            (-max_extent <= u)
            & (u <= max_extent)
            & (-max_extent <= v)
            & (v <= max_extent)
            & (-max_extent <= u + v)
            & (u + v <= max_extent)
        )
        u, v = u[extent_condition].values, v[extent_condition].values
        fig, ax, _ = plots.hex_scatter(
            u,
            v,
            color=1,
            label=node_type,
            fig=fig,
            ax=ax,
            edgecolor=edgecolor,
            edgewidth=edgewidth,
            alpha=alpha,
            fill=fill,
            cmap=cmap,
            cbar=False,
            **kwargs,
        )
        return fig

    def hex_layout_all(
        self,
        node_types=None,
        anatomic_order=False,
        edgecolor="black",
        alpha=1,
        fill=True,
        max_extent=5,
        cmap=cm.get_cmap("binary"),
        fig=None,
        axes=None,
        **kwargs,
    ):
        """Plot the layout of a node type on a regular hex grid."""
        node_types = self.node_types_sorted if node_types is None else node_types
        if not (fig and axes):
            fig, axes, (gw, gh) = plt_utils.get_axis_grid(self.node_types_sorted)
        if anatomic_order:
            node_types = [key for key in self.layout.keys() if key in node_types]
        for i, node_type in enumerate(node_types):
            self.hex_layout(
                node_type,
                edgecolor=edgecolor,
                edgewidth=0.1,
                alpha=alpha,
                fill=fill,
                max_extent=max_extent,
                cmap=cmap,
                fig=fig,
                ax=axes[i],
                **kwargs,
            )
        return fig

    def get_uv(self, node_type):
        """hex-coordinates of a particular node type"""
        nodes = self.nodes.to_df()
        nodes = nodes[nodes.type == node_type]
        u, v = nodes[["u", "v"]].values.T
        return u, v

    # ---- RECEPTIVE FIELDS

    def sources_list(self, node_type):
        edges = self.edges.to_df()
        return np.unique(edges[edges.target_type == node_type].source_type.values)

    def targets_list(self, node_type):
        edges = self.edges.to_df()
        return np.unique(edges[edges.source_type == node_type].target_type.values)

    def receptive_field(
        self,
        source="Mi9",
        target="T4a",
        rfs=None,
        max_extent=None,
        vmin=None,
        vmax=None,
        title="{source} :→ {target}",
        **kwargs,
    ):
        """
        Plots the receptive field from 'taregt' from 'source'.
        """
        if rfs is None:
            rfs = ReceptiveFields(target, self.edges.to_df())
            max_extent = max_extent or rfs.max_extent
        # weights
        weights = self._weights()

        # to derive color range values taking all inputs into account
        vmin = min(
            0,
            min(weights[rfs[source].index].min() for source in rfs.source_types),
        )

        vmax = max(
            0,
            max(weights[rfs[source].index].max() for source in rfs.source_types),
        )

        weights = weights[rfs[source].index]
        label = ""

        du_inv, dv_inv = -rfs[source].du.values, -rfs[source].dv.values
        fig, ax, (label_text, scalarmapper) = plots.kernel(
            du_inv,
            dv_inv,
            weights,
            label=label,
            max_extent=max_extent,
            fill=True,
            vmin=vmin,
            vmax=vmax,
            title=title.format(**locals()),
            **kwargs,
        )
        return fig

    def receptive_fields_grid(
        self,
        target,
        sources=None,
        sort_alphabetically=True,
        aspect_ratio=1,
        ax_titles="{source} :→ {target}",
        figsize=[20, 20],
        max_h_axes=None,
        max_v_axes=None,
        hspace=0,
        wspace=0.0,
        min_axes=-1,
        keep_nan_axes=True,
        max_extent=None,
        fig=None,
        axes=None,
        ignore_sign_error=False,
        **kwargs,
    ):
        """
        Plots all receptive fields of 'target' inside a regular grid of axes.
        """

        rfs = ReceptiveFields(target, self.edges.to_df())
        max_extent = max_extent or rfs.max_extent
        weights = self._weights()

        # to sort in descending order by sum of inputs
        sorted_sum_of_inputs = dict(
            sorted(
                valmap(lambda v: weights[v.index].sum(), rfs).items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        # to sort alphabetically in case sources is specified
        if sort_alphabetically:
            sources, _ = nodes_edges_utils.order_nodes_list(sources)
        sources = sources or list(sorted_sum_of_inputs.keys())

        # to derive color range values taking all inputs into account
        vmin = min(0, min(weights[rfs[source].index].min() for source in sources))
        vmax = max(0, max(weights[rfs[source].index].max() for source in sources))

        num_axes = max(min_axes, len(sources))

        if fig is None or axes is None:
            width, height = plt_utils.width_n_height(
                num_axes,
                aspect_ratio,
                max_width=max_h_axes,
                max_height=max_v_axes,
            )
            panels = np.arange(width * height).astype(float)
            panels[panels > len(sources) - 1] = np.nan
            panels = panels.reshape(height, width)
            fig, axes = plt_utils.divide_figure_to_grid(
                panels,
                figsize=figsize,
                wspace=wspace,
                hspace=hspace,
                keep_nan_axes=keep_nan_axes,
            )
        cbar = kwargs.get("cbar", False)
        for i, src in enumerate(sources):
            if i == 0 and cbar:
                cbar = True
                kwargs.update(cbar=cbar)
            else:
                cbar = False
                kwargs.update(cbar=cbar)
            try:
                self.receptive_field(
                    target=target,
                    source=src,
                    fig=fig,
                    ax=axes[i],
                    title=ax_titles,
                    vmin=vmin,
                    vmax=vmax,
                    rfs=rfs,
                    max_extent=max_extent,
                    annotate_coords=False,
                    **kwargs,
                )
            except plots.SignError as e:
                if ignore_sign_error:
                    pass
                else:
                    raise e
        return fig

    # ---- PROJECTIVE FIELDS

    def projective_field(
        self,
        source="Mi9",
        target="T4a",
        title="{source} →: {target}",
        prfs=None,
        max_extent=None,
        vmin=None,
        vmax=None,
        **kwargs,
    ):
        """
        Plots the projective field from 'source' to 'target'.
        """
        if prfs is None:
            prfs = ProjectiveFields(source, self.edges.to_df())
            max_extent = max_extent or prfs.max_extent
        if max_extent is None:
            return None
        weights = self._weights()

        # to derive color range values taking all inputs into account
        vmin = min(
            0,
            min(weights[prfs[target].index].min() for target in prfs.target_types),
        )

        vmax = max(
            0,
            max(weights[prfs[target].index].max() for target in prfs.target_types),
        )

        weights = weights[prfs[target].index]
        label = ""
        du, dv = prfs[target].du.values, prfs[target].dv.values
        fig, ax, (label_text, scalarmapper) = plots.kernel(
            du,
            dv,
            weights,
            label=label,
            fill=True,
            max_extent=max_extent,
            vmin=vmin,
            vmax=vmax,
            title=title.format(**locals()),
            **kwargs,
        )
        return fig

    def projective_fields_grid(
        self,
        source,
        targets=None,
        fig=None,
        axes=None,
        aspect_ratio=1,
        figsize=[20, 20],
        ax_titles="{source} →: {target}",
        max_h_axes=None,
        max_v_axes=None,
        hspace=0,
        wspace=0.0,
        min_axes=-1,
        keep_nan_axes=True,
        max_extent=None,
        sort_alphabetically=False,
        ignore_sign_error=False,
        **kwargs,
    ):
        """
        Plots all projective field of 'source' inside a regular grid of axes.
        """
        prfs = ProjectiveFields(source, self.edges.to_df())
        max_extent = max_extent or prfs.max_extent
        weights = self._weights()
        sorted_sum_of_outputs = dict(
            sorted(
                valmap(lambda v: weights[v.index].sum(), prfs).items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        # to sort alphabetically in case sources is specified
        if sort_alphabetically:
            targets, _ = nodes_edges_utils.order_nodes_list(targets)

        targets = targets or list(sorted_sum_of_outputs.keys())

        vmin = min(0, min(weights[prfs[target].index].min() for target in targets))
        vmax = max(0, max(weights[prfs[target].index].max() for target in targets))
        num_axes = max(min_axes, len(targets))

        if fig is None or axes is None:
            width, height = plt_utils.width_n_height(
                num_axes,
                aspect_ratio,
                max_width=max_h_axes,
                max_height=max_v_axes,
            )
            panels = np.arange(width * height).astype(float)
            panels[panels > len(targets) - 1] = np.nan
            panels = panels.reshape(height, width)
            fig, axes = plt_utils.divide_figure_to_grid(
                panels,
                figsize=figsize,
                wspace=wspace,
                hspace=hspace,
                keep_nan_axes=keep_nan_axes,
            )

        cbar = kwargs.get("cbar", False)
        for i, target in enumerate(targets):
            if i == 0 and cbar:
                cbar = True
                kwargs.update(cbar=cbar)
            else:
                cbar = False
                kwargs.update(cbar=cbar)
            try:
                self.projective_field(
                    source=source,
                    target=target,
                    fig=fig,
                    ax=axes[i],
                    title=ax_titles,
                    prfs=prfs,
                    max_extent=max_extent,
                    vmin=vmin,
                    vmax=vmax,
                    annotate_coords=False,
                    **kwargs,
                )
            except plots.SignError as e:
                if ignore_sign_error:
                    pass
                else:
                    raise e
        return fig

    def receptive_fields_df(self, target_type):
        return ReceptiveFields(target_type, self.edges.to_df())

    def projective_fields_df(self, source_type):
        return ProjectiveFields(source_type, self.edges.to_df())

    def receptive_fields_sum(self, target_type):
        return ReceptiveFields(target_type, self.edges.to_df()).sum()

    def projective_fields_sum(self, source_type):
        return ProjectiveFields(source_type, self.edges.to_df()).sum()


class ReceptiveFields(Namespace):
    """Dictionary of receptive field dataframes for a specific cell type.

    Args:
        target_type: target cell type.
        edges: all edges of a Connectome.

    Attributes:
        target_type str
        source_types List[str]
    """

    def __init__(self, target_type, edges, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_extents", [])
        _receptive_fields_edge_dfs(self, target_type, edges)

    @property
    def extents(self):
        return dict(zip(self.source_types, self._extents))

    @property
    def max_extent(self):
        return max(self._extents) if self._extents else None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.target_type})"

    def sum(self):
        return {key: self[key].n_syn.sum() for key in self}


class ProjectiveFields(Namespace):
    """Dictionary of projective field dataframes for a specific cell type.

    Args:
        source_type: target cell type.
        edges: all edges of a Connectome.

    Attributes:
        source_type str
        target_types List[str]
    """

    def __init__(self, source_type: str, edges: DataFrame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_extents", [])
        _projective_fields_edge_dfs(self, source_type, edges)

    @property
    def extents(self):
        return dict(zip(self.target_types, self._extents))

    @property
    def max_extent(self):
        return max(self._extents) if self._extents else None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.source_type})"

    def sum(self):
        return {key: self[key].n_syn.sum() for key in self}


def _receptive_fields_edge_dfs(
    cls: ReceptiveFields, target_type: str, edges: DataFrame
) -> ReceptiveFields:
    """Populate ReceptiveFields."""

    edges = edges[edges.target_type == target_type]
    source_types = edges.source_type.unique()

    object.__setattr__(cls, "target_type", target_type)
    object.__setattr__(cls, "source_types", source_types)

    for source_type in source_types:
        _edges = edges[edges.source_type == source_type]

        most_central_edge = _edges.iloc[
            np.argmin(np.abs(_edges.target_u) + np.abs(_edges.target_v))
        ]
        target_u_min = most_central_edge.target_u
        target_v_min = most_central_edge.target_v

        cls[source_type] = _edges[
            (_edges.target_u == target_u_min) & (_edges.target_v == target_v_min)
        ]
        cls._extents.append(
            hex_utils.get_extent(cls[source_type].du, cls[source_type].dv)
        )
    return cls


def _projective_fields_edge_dfs(
    cls: ProjectiveFields, source_type: str, edges: DataFrame
) -> ProjectiveFields:
    """Populate ProjectiveFields."""

    edges = edges[edges.source_type == source_type]
    target_types = edges.target_type.unique()

    object.__setattr__(cls, "source_type", source_type)
    object.__setattr__(cls, "target_types", target_types)

    for target_type in target_types:
        _edges = edges[edges.target_type == target_type]
        most_central_edge = _edges.iloc[
            np.argmin(np.abs(_edges.source_u) + np.abs(_edges.source_v))
        ]
        source_u_min = most_central_edge.source_u
        source_v_min = most_central_edge.source_v
        cls[target_type] = _edges[
            (_edges.source_u == source_u_min) & (_edges.source_v == source_v_min)
        ]
        cls._extents.append(
            hex_utils.get_extent(cls[target_type].du, cls[target_type].dv)
        )

    return cls  # , max_extent
