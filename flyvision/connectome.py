"""Connectome compiler and visualizer."""

import json
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from toolz import groupby, valmap
import matplotlib.path as mp
from matplotlib import colormaps as cm
from matplotlib.figure import Figure
from matplotlib.colors import Colormap
import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame

from datamate import Directory, Namespace, ArrayFile, root

import flyvision
from flyvision.plots.figsize_utils import figsize_from_n_items
from flyvision.plots.network import WholeNetworkFigure
from flyvision.utils import nodes_edges_utils, df_utils, hex_utils
from flyvision.plots import plots, plt_utils


__all__ = ["ConnectomeDir", "ConnectomeView"]


# -- `Connectome` --------------------------------------------------------------
@root(flyvision.root_dir / "connectome")
class ConnectomeDir(Directory):
    """Compiles a connectome graph from average convolutional filters.

    The graph is cells (nodes) and synapse sets (edges).

    Attributes:

        Files:
            unique_cell_types (str): identified cell types
            input_cell_types (str): input cell types
            intermediate_cell_types (str): hidden cell types
            output_cell_types (str): decoded cell types
            central_cells_index (int): index of central cell in nodes table
                for each cell type in unique_cell_types.
            layout (str): input, hidden, output definitions for visualization.

        SubDirs:

            nodes (NodeDir): table with a row for each individual node/cell and
                columns/files describing their attributed.

                Files:
                    type (str): cell type name
                    u (int): hex-coordinates #1 (oblique coordinates)
                    v (int): hex-coordinates #2 (oblique coordinates)
                    role (str): input, hidden, or output

                SubDirs:

                    layer_index (Directory): all indices of a cell type in nodes.

                        Files:
                            <cell_type> (int): all cell indices of cell_type in
                                nodes.

            edges (EdgeDir): A table with a row for each edge.

                Files:
                    source_index (str): presynaptic cell
                    target_index (int): postsynaptic cell
                    sign (int): +1 (excitatory) or -1 (inhibitory)
                    n_syn (float): synapse count
                    other files for convenience

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
                    ]*],
                    "edge_type": "chem" | "elec"
                }*]
            }

        See "data/connectome/fib25-fib19_v2.2.json" for an example.

    Example:
        >>> config = Namespace(file='fib25-fib19_v2.2.json',
                               extent=15,
                               n_syn_fill=1)
        >>> connectome = Connectome(config)
    """

    class Config:
        file: str
        "The name of a JSON connectome file"
        extent: int
        "The array radius, in columns"
        n_syn_fill: int
        "The number of synapses to assume in data gaps"

    # -- Contents ------------------------------------------
    unique_cell_types: ArrayFile
    "A list of all cell types"
    input_cell_types: ArrayFile
    "The cell types to use as inputs"
    intermediate_cell_types: ArrayFile
    "The hidden cell types"
    output_cell_types: ArrayFile
    "The cell types to use for task readout"
    central_cells_index: ArrayFile
    "Index of the central node of a unique cell type in the nodes table"
    layout: ArrayFile
    "Input, hidden, output layout for later circuit visualization"
    nodes: Directory
    "A table with a row for each node"
    edges: Directory
    "A table with a row for each edge"
    # ------------------------------------------------------

    def __init__(self, config: Config) -> None:
        # Load the connectome spec.
        spec = json.loads(Path(self.path.parent / config.file).read_text())

        # Store unique cell types and layout variables.
        self.unique_cell_types = np.string_([n["name"] for n in spec["nodes"]])
        self.input_cell_types = np.string_(spec["input_units"])
        self.output_cell_types = np.string_(spec["output_units"])
        intermediate_cell_types, _ = nodes_edges_utils.oder_node_type_list(
            np.array(
                list(
                    set(self.unique_cell_types)
                    - set(self.input_cell_types)
                    - set(self.output_cell_types)
                )
            ).astype(str)
        )
        self.intermediate_cell_types = np.array(intermediate_cell_types).astype("S")

        layout = []
        layout.extend(
            list(
                zip(
                    self.input_cell_types,
                    [b"retina" for _ in range(len(self.input_cell_types))],
                )
            )
        )
        layout.extend(
            list(
                zip(
                    self.intermediate_cell_types,
                    [b"intermediate" for _ in range(len(self.intermediate_cell_types))],
                )
            )
        )
        layout.extend(
            list(
                zip(
                    self.output_cell_types,
                    [b"output" for _ in range(len(self.output_cell_types))],
                )
            )
        )
        self.layout = np.string_(layout)

        # Construct nodes and edges.
        nodes: List[Node] = []
        edges: List[Edge] = []
        add_nodes(nodes, spec["nodes"], config.extent)
        add_edges(edges, nodes, spec["edges"], config.n_syn_fill)

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
        self.central_cells_index = np.int64(
            np.nonzero((self.nodes.u[:] == 0) & (self.nodes.v[:] == 0))[0]
        )

        # Store layer indices.
        layer_index = {}
        for cell_type in self.unique_cell_types[:]:
            node_indices = np.nonzero(self.nodes["type"][:] == cell_type)[0]
            layer_index[cell_type.decode()] = np.int64(node_indices)
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


class NodeDir(Directory):
    """Stored data of the compiled Connectome describing the nodes."""

    pass


def add_nodes(seq: List[Node], node_spec: dict, extent: int) -> None:
    """Add nodes to `seq`, based on `node_spec`."""
    for n in node_spec:
        typ, (pattern, args) = n["name"], n["pattern"]
        if pattern == "stride":
            add_strided_nodes(seq, typ, extent, args)
        elif pattern == "tile":
            add_tiled_nodes(seq, typ, extent, args)
        elif pattern == "single":
            add_single_node(seq, typ, extent)


def add_strided_nodes(
    seq: List[Node], typ: str, extent: int, strides: Tuple[int, int]
) -> None:
    """Add to `seq` a population of neurons arranged in a hexagonal grid."""
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


class EdgeDir(Directory):
    """Stored data of the compiled Connectome describing the edges."""

    pass


def add_edges(
    seq: List[Edge], nodes: List[Node], edge_spec: dict, n_syn_fill: float
) -> None:
    """Add edges to `seq` based on `edge_spec`."""
    node_index = {
        **groupby(lambda n: n.type, nodes),
        **groupby(lambda n: (n.type, n.u, n.v), nodes),
    }
    for e in edge_spec:
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
        )


def fill_hull(offsets: List[list], n_syn_fill: float) -> List[list]:
    """Fill in the convex hull of the reported edges."""
    # to save time at library import
    import scipy.spatial as ss

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
) -> None:
    """Construct a connection set with convolutional weight symmetry."""
    for (du, dv), n_syn in offsets:
        for src in node_index.get(source_typ, []):
            u_tgt = src.u + du
            v_tgt = src.v + dv
            with suppress(KeyError):
                tgt = node_index[target_typ, u_tgt, v_tgt][0]
                seq.append(Edge(len(seq), src, tgt, sign, n_syn, type, n_syn_certainty))


# -- ConnectomeView ------------------------------------------------------------


class ConnectomeView:
    """Visualization of the connectome data.

    Args:
        connectome (ConnectomeDir): Directory of the connectome.
        groups (List[str]): regular expressions to sort the nodes by.

    Attributes:
        connectome (ConnectomeDir): connectome of connectome.
        nodes (Directory): node table.
        edges (Directory): edge table.
    """

    def __init__(
        self,
        connectome: ConnectomeDir,
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
        self.connectome = connectome

        assert "nodes" in self.connectome and "edges" in self.connectome

        self.edges = self.connectome.edges

        self.nodes = self.connectome.nodes

        self.cell_types_unsorted = self.connectome.unique_cell_types[:].astype(str)

        (
            self.cell_types_sorted,
            self.cell_types_sort_index,
        ) = nodes_edges_utils.oder_node_type_list(
            self.connectome.unique_cell_types[:].astype(str), groups
        )

        self.layout = dict(self.connectome.layout[:].astype(str))
        self.node_indexer = nodes_edges_utils.NodeIndexer(self.connectome)

    # -- connectivity matrix -------------------------------------------------------

    def connectivity_matrix(
        self,
        mode: str = "n_syn",
        only_sign: Optional[str] = None,
        cell_types: Optional[List[str]] = None,
        no_symlog: Optional[bool] = False,
        min_number: Optional[float] = None,
        cmap: Optional[Colormap] = None,
        size_scale: Optional[float] = None,
        title: Optional[str] = None,
        cbar_label: Optional[str] = None,
        **kwargs,
    ) -> Figure:
        """Plots the connectivity matrix as counts or weights.

        Args:
            mode: 'n_syn' referring to number of input synapses,
                    'count' referring to number of input neurons.
            only_sign: '+' for displaying only excitatory projections,
                    '-' for displaying only inhibitory projections.
            cell_types: provide a subset of nodes to only display those.
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

        kwargs.update(_kwargs[mode])
        if no_symlog:
            kwargs.update(symlog=None)
            kwargs.update(midpoint=0)

        edges = self.edges.to_df()

        # to take projections onto central nodes (home columns) into account
        edges = edges[(edges.target_u == 0) & (edges.target_v == 0)]

        # filter edges to allow providing a subset of cell types
        cell_types = cell_types or self.cell_types_sorted
        edges = df_utils.filter_by_column_values(
            df_utils.filter_by_column_values(
                edges, column="source_type", values=cell_types
            ),
            column="target_type",
            values=cell_types,
        )
        weights = self._weights()[edges.index]

        # lookup table for key -> (i, j)
        type_index = {node_typ: i for i, node_typ in enumerate(cell_types)}
        matrix = np.zeros([len(type_index), len(type_index)])

        for srctyp, tgttyp, weight in zip(
            edges.source_type.values, edges.target_type.values, weights
        ):
            if mode == "count":
                # to simply count the number of projections
                matrix[type_index[srctyp], type_index[tgttyp]] += 1
            elif mode in ["weight", "n_syn"]:
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

        fig, axes, cbar, matrix = plots.heatmap(matrix, cell_types, **kwargs)
        return fig

    def _weights(self) -> NDArray:
        return self.edges.sign[:] * self.edges.n_syn[:]

    # -- network graphs ------------------------------------------------------------

    def network_layout(
        self,
        max_extent: int = 5,
        **kwargs,
    ) -> Figure:
        """Retinotopic hexagonal lattice columnar organization of the network.

        Args:
            max_extent: integer column radius to visualize.
        """
        backbone = WholeNetworkFigure(self.connectome)
        backbone.init_figure(figsize=[7, 3])
        return self.hex_layout_all(
            max_extent=max_extent, fig=backbone.fig, axes=backbone.axes, **kwargs
        )

    def hex_layout(
        self,
        cell_type: str,
        max_extent: int = 5,
        edgecolor="none",
        edgewidth=0.5,
        alpha=1,
        fill=False,
        cmap=None,
        fig=None,
        ax=None,
        **kwargs,
    ):
        """Retinotopic hexagonal lattice columnar organization of the cell type."""
        nodes = self.nodes.to_df()
        node_condition = nodes.type == cell_type
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

        label = cell_type
        if ax is not None:
            # prevent labeling twice
            label = (
                cell_type if cell_type not in [t.get_text() for t in ax.texts] else ""
            )

        fig, ax, _ = plots.hex_scatter(
            u,
            v,
            color=1,
            label=label,
            fig=fig,
            ax=ax,
            edgecolor=edgecolor,
            edgewidth=edgewidth,
            alpha=alpha,
            fill=fill,
            cmap=cmap or plt_utils.get_alpha_colormap("#2f3541", 1),
            cbar=False,
            **kwargs,
        )
        return fig

    def hex_layout_all(
        self,
        cell_types: str = None,
        max_extent: int = 5,
        edgecolor="none",
        alpha=1,
        fill=False,
        cmap=None,
        fig=None,
        axes=None,
        **kwargs,
    ):
        """Retinotopic hexagonal lattice columnar organization of all cell types."""
        cell_types = self.cell_types_sorted if cell_types is None else cell_types
        if fig is None or axes is None:
            fig, axes, (gw, gh) = plt_utils.get_axis_grid(self.cell_types_sorted)

        for i, cell_type in enumerate(cell_types):
            self.hex_layout(
                cell_type,
                edgecolor=edgecolor,
                edgewidth=0.1,
                alpha=alpha,
                fill=fill,
                max_extent=max_extent,
                cmap=cmap or plt_utils.get_alpha_colormap("#2f3541", 1),
                fig=fig,
                ax=axes[i],
                **kwargs,
            )
        return fig

    def get_uv(self, cell_type) -> Tuple[NDArray]:
        """Hex-coordinates of a particular cell type to pass to hex_scatter plot."""
        nodes = self.nodes.to_df()
        nodes = nodes[nodes.type == cell_type]
        u, v = nodes[["u", "v"]].values.T
        return u, v

    # -- receptive fields ----------------------------------------------------------

    def sources_list(self, cell_type: str) -> NDArray:
        """Presynaptic cell types."""
        edges = self.edges.to_df()
        return np.unique(edges[edges.target_type == cell_type].source_type.values)

    def targets_list(self, cell_type: str) -> NDArray:
        """Postsynaptic cell types."""
        edges = self.edges.to_df()
        return np.unique(edges[edges.source_type == cell_type].target_type.values)

    def receptive_field(
        self,
        source: str = "Mi9",
        target: str = "T4a",
        rfs: Optional["ReceptiveFields"] = None,
        max_extent: int = None,
        vmin=None,
        vmax=None,
        title="{source} :→ {target}",
        **kwargs,
    ):
        """Receptive field of target from source."""
        if rfs is None:
            rfs = ReceptiveFields(target, self.edges.to_df())
            max_extent = max_extent or rfs.max_extent

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

        # requires to look from the target cell, ie mirror the coordinates
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
        target: str,
        sources: Iterable[str] = None,
        sort_alphabetically=True,
        ax_titles="{source} :→ {target}",
        figsize=[20, 20],
        max_extent=None,
        fig=None,
        axes=None,
        ignore_sign_error=False,
        max_figure_height_cm=22,
        panel_height_cm=3,
        max_figure_width_cm=18,
        panel_width_cm=3.6,
        **kwargs,
    ):
        """Receptive fields of target inside a regular grid of axes."""

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
            sources, _ = nodes_edges_utils.oder_node_type_list(sources)
        sources = sources or list(sorted_sum_of_inputs.keys())

        # to derive color range values taking all inputs into account
        vmin = min(0, min(weights[rfs[source].index].min() for source in sources))
        vmax = max(0, max(weights[rfs[source].index].max() for source in sources))

        if fig is None or axes is None:
            figsize = figsize_from_n_items(
                len(rfs.source_types),
                max_figure_height_cm=max_figure_height_cm,
                panel_height_cm=panel_height_cm,
                max_figure_width_cm=max_figure_width_cm,
                panel_width_cm=panel_width_cm,
            )
            fig, axes = figsize.axis_grid(
                unmask_n=len(rfs.source_types), hspace=0.0, wspace=0
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
                    annotate=False,
                    annotate_coords=False,
                    title_y=0.9,
                    **kwargs,
                )
            except plots.SignError as e:
                if ignore_sign_error:
                    pass
                else:
                    raise e
        return fig

    # -- projective fields ---------------------------------------------------------

    def projective_field(
        self,
        source: str = "Mi9",
        target: str = "T4a",
        title="{source} →: {target}",
        prfs: Optional["ProjectiveFields"] = None,
        max_extent: int = None,
        vmin=None,
        vmax=None,
        **kwargs,
    ):
        """Projective field from source to target."""
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
        source: str,
        targets: Iterable[str] = None,
        fig=None,
        axes=None,
        figsize=[20, 20],
        ax_titles="{source} →: {target}",
        min_axes=-1,
        max_figure_height_cm=22,
        panel_height_cm=3,
        max_figure_width_cm=18,
        panel_width_cm=3.6,
        max_extent=None,
        sort_alphabetically=False,
        ignore_sign_error=False,
        **kwargs,
    ):
        """Projective fields of source inside a regular grid of axes."""
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
            targets, _ = nodes_edges_utils.oder_node_type_list(targets)

        targets = targets or list(sorted_sum_of_outputs.keys())

        vmin = min(0, min(weights[prfs[target].index].min() for target in targets))
        vmax = max(0, max(weights[prfs[target].index].max() for target in targets))

        if fig is None or axes is None:
            figsize = figsize_from_n_items(
                len(prfs.target_types),
                max_figure_height_cm=max_figure_height_cm,
                panel_height_cm=panel_height_cm,
                max_figure_width_cm=max_figure_width_cm,
                panel_width_cm=panel_width_cm,
            )
            fig, axes = figsize.axis_grid(
                unmask_n=len(prfs.target_types), hspace=0.0, wspace=0
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
                    annotate=False,
                    title_y=0.9,
                    **kwargs,
                )
            except plots.SignError as e:
                if ignore_sign_error:
                    pass
                else:
                    raise e
        return fig

    def receptive_fields_df(self, target_type) -> "ReceptiveFields":
        return ReceptiveFields(target_type, self.edges.to_df())

    def projective_fields_df(self, source_type) -> "ProjectiveFields":
        return ProjectiveFields(source_type, self.edges.to_df())

    def receptive_fields_sum(self, target_type) -> Dict[str, int]:
        return ReceptiveFields(target_type, self.edges.to_df()).sum()

    def projective_fields_sum(self, source_type) -> Dict[str, int]:
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

    return cls
