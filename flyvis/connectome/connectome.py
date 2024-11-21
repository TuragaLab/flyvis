"""Connectome compiler and visualizer."""

import json
from contextlib import suppress
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    Union,
)

import matplotlib.path as mp
import numpy as np
from datamate import ArrayFile, Directory, Namespace, root
from matplotlib import colormaps as cm
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pandas import DataFrame
from toolz import groupby, valmap

import flyvis
from flyvis.analysis.visualization import plots, plt_utils
from flyvis.analysis.visualization.figsize_utils import figsize_from_n_items
from flyvis.analysis.visualization.network_fig import WholeNetworkFigure
from flyvis.utils import df_utils, hex_utils, nodes_edges_utils

__all__ = [
    "ConnectomeFromAvgFilters",
    "ConnectomeView",
    "ReceptiveFields",
    "ProjectiveFields",
    "init_connectome",
    "get_avgfilt_connectome",
]


class Connectome(Protocol):
    """Protocol for connectome classes compatible with flyvis.network.Network.

    Note:
        Nodes and edges have additional attributes that require compatibility
        with `Parameter` class implementations. For instance, when a parameter
        for edges is derived from synapse counts, the edges have an `n_syn`
        attribute (ArrayFile or np.ndarray).
    """

    class nodes:
        index: Union[np.ndarray, ArrayFile]
        ...

    class edges:
        source_index: Union[np.ndarray, ArrayFile]
        target_index: Union[np.ndarray, ArrayFile]
        ...


AVAILABLE_CONNECTOMES: Dict[str, Type[Connectome]] = {}


def register_connectome(
    cls: Optional[Type[Connectome]] = None,
) -> Union[Callable[[Type[Connectome]], Type[Connectome]], Type[Connectome]]:
    """
    Register a new connectome class.

    Args:
        cls: The connectome class to register (optional when used as a decorator).

    Returns:
        Registered class or decorator function.

    Example:
        As a standalone function: register_connectome(name, cls)
        ```python
        class CustomConnectome(Connectome):
            ...
        register_connectome("CustomConnectome", CustomConnectome)
        ```

        As a decorator:
        ```python
        @register_connectome("CustomConnectome")
        class CustomConnectome(Connectome):
            ...
        ```
    """

    def decorator(cls: Type[Connectome]) -> Type[Connectome]:
        AVAILABLE_CONNECTOMES[cls.__name__] = cls
        return cls

    return decorator if cls is None else decorator(cls)


# -- `Connectome` --------------------------------------------------------------
@register_connectome
@root(flyvis.root_dir / "connectome")
class ConnectomeFromAvgFilters(Directory):
    """Compiles a connectome graph from average convolutional filters.

    The graph consists of cells (nodes) and synapse sets (edges).

    Args:
        file: The name of a JSON connectome file.
        extent: The array radius, in columns.
        n_syn_fill: The number of synapses to assume in data gaps.

    Attributes:
        unique_cell_types (ArrayFile): Identified cell types.
        input_cell_types (ArrayFile): Input cell types.
        intermediate_cell_types (ArrayFile): Hidden cell types.
        output_cell_types (ArrayFile): Decoded cell types.
        central_cells_index (ArrayFile): Index of central cell in nodes table
            for each cell type in unique_cell_types.
        layout (ArrayFile): Input, hidden, output definitions for visualization.
        nodes (NodeDir): Table with a row for each individual node/cell.
        edges (EdgeDir): Table with a row for each edge.

    Note:
        A connectome can be constructed from a JSON model file following this schema:

        ```json
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
                }*]
            }
        }
        ```

        See "data/connectome/fib25-fib19_v2.2.json" for an example.

    Example:
        ```python
        config = Namespace(file='fib25-fib19_v2.2.json', extent=15, n_syn_fill=1)
        connectome = Connectome(config)
        ```
    """

    def __init__(self, file=flyvis.connectome_file.name, extent=15, n_syn_fill=1) -> None:
        # case 0: file is an absolute path
        if Path(file).exists():
            file = Path(file)
        # case 1: file is specified within the package resources
        elif (resources.files("flyvis.connectome") / file).is_file():
            file = resources.files("flyvis.connectome").joinpath(file)
        # case 2: file is specified relative to the root directory
        elif (flyvis.root_dir / "connectome" / file).exists():
            file = flyvis.root_dir / "connectome" / file
        else:
            raise FileNotFoundError(f"Connectome file {file} not found.")

        # Load the connectome spec.
        spec = json.loads(Path(file).read_text())

        # Store unique cell types and layout variables.
        self.unique_cell_types = np.bytes_([n["name"] for n in spec["nodes"]])
        self.input_cell_types = np.bytes_(spec["input_units"])
        self.output_cell_types = np.bytes_(spec["output_units"])
        intermediate_cell_types, _ = nodes_edges_utils.order_node_type_list(
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
        self.layout = np.bytes_(layout)

        # Construct nodes and edges.
        nodes: List[Node] = []
        edges: List[Edge] = []
        add_nodes(nodes, spec["nodes"], extent)
        add_edges(edges, nodes, spec["edges"], n_syn_fill)

        # Define node roles (input, intermediate, output).
        _role = {node: "intermediate" for node in set([n.type for n in nodes])}
        _role.update({node: "input" for node in _role if node in spec["input_units"]})
        _role.update({node: "output" for node in _role if node in spec["output_units"]})

        # Store the graph.
        self.nodes = dict(  # type: ignore
            index=np.int64([n.id for n in nodes]),
            type=np.bytes_([n.type for n in nodes]),
            u=np.int32([n.u for n in nodes]),
            v=np.int32([n.v for n in nodes]),
            role=np.bytes_([_role[n.type] for n in nodes]),
        )

        self.edges = dict(  # type: ignore
            # [Essential fields]
            source_index=np.int64([e.source.id for e in edges]),
            target_index=np.int64([e.target.id for e in edges]),
            sign=np.float32([e.sign for e in edges]),
            n_syn=np.float32([e.n_syn for e in edges]),
            # [Convenience fields]
            source_type=np.bytes_([e.source.type for e in edges]),
            target_type=np.bytes_([e.target.type for e in edges]),
            source_u=np.int32([e.source.u for e in edges]),
            target_u=np.int32([e.target.u for e in edges]),
            source_v=np.int32([e.source.v for e in edges]),
            target_v=np.int32([e.target.v for e in edges]),
            du=np.int32([e.target.u - e.source.u for e in edges]),
            dv=np.int32([e.target.v - e.source.v for e in edges]),
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
    """Represents a node in the connectome graph.

    Attributes:
        id: Index (0..n_nodes).
        type: Cell type name.
        u: Location coordinate #1 (oblique coordinates).
        v: Location coordinate #2 (oblique coordinates).
        u_stride: Stride in u.
        v_stride: Stride in v.
    """

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


def add_nodes(seq: List[Node], node_spec: dict, extent: int) -> None:
    """Add nodes to `seq`, based on `node_spec`.

    Args:
        seq: List to append nodes to.
        node_spec: Dictionary specifying node properties.
        extent: The array radius, in columns.
    """
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
    """Add to `seq` a population of neurons arranged in a hexagonal grid.

    Args:
        seq: List to append nodes to.
        typ: Cell type name.
        extent: The array radius, in columns.
        strides: Tuple of (u_stride, v_stride).
    """
    n = extent
    u_stride, v_stride = strides
    for u in range(-n, n + 1):
        for v in range(max(-n, -n - u), min(n, n - u) + 1):
            if u % u_stride == 0 and v % v_stride == 0:
                seq.append(Node(len(seq), typ, u, v, u_stride, v_stride))


def add_tiled_nodes(seq: List[Node], typ: str, extent: int, n: int) -> None:
    """Add to `seq` a population of neurons with strides `(n, n)`.

    Args:
        seq: List to append nodes to.
        typ: Cell type name.
        extent: The array radius, in columns.
        n: Stride value for both u and v.
    """
    add_strided_nodes(seq, typ, extent, (n, n))


def add_single_node(seq: List[Node], typ: str, extent: int) -> None:
    """Add to `seq` a single-neuron population.

    Args:
        seq: List to append nodes to.
        typ: Cell type name.
        extent: The array radius, in columns.
    """
    add_strided_nodes(seq, typ, 0, (1, 1))


# -- Edge construction ---------------------------------------------------------


@dataclass
class Edge:
    """Represents an edge in the connectome graph.

    Attributes:
        id: Index (0..n_edges).
        source: Presynaptic cell.
        target: Postsynaptic cell.
        sign: +1 (excitatory) or -1 (inhibitory).
        n_syn: Synapse count.
        type: Synapse type.
        n_syn_certainty: Certainty of synapse count.
    """

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
    n_syn_certainty: float
    "certainty of synapse count"


class EdgeDir(Directory):
    """Stored data of the compiled Connectome describing the edges."""


def add_edges(
    seq: List[Edge], nodes: List[Node], edge_spec: dict, n_syn_fill: float
) -> None:
    """Add edges to `seq` based on `edge_spec`.

    Args:
        seq: List to append edges to.
        nodes: List of all nodes in the connectome.
        edge_spec: Dictionary specifying edge properties.
        n_syn_fill: Number of synapses to assume in data gaps.
    """
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
            e["lambda_mult"],
        )


def fill_hull(offsets: List[list], n_syn_fill: float) -> List[list]:
    """Fill in the convex hull of the reported edges.

    Args:
        offsets: List of edge offsets.
        n_syn_fill: Number of synapses to assume in data gaps.

    Returns:
        List of filled offsets.
    """
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
    n_syn_certainty: float,
) -> None:
    """Construct a connection set with convolutional weight symmetry.

    Args:
        seq: List to append edges to.
        node_index: Dictionary mapping node types to nodes.
        source_typ: Source cell type.
        target_typ: Target cell type.
        sign: +1 (excitatory) or -1 (inhibitory).
        offsets: List of edge offsets.
        n_syn_certainty: Certainty of synapse count.
    """
    for (du, dv), n_syn in offsets:
        for src in node_index.get(source_typ, []):
            u_tgt = src.u + du
            v_tgt = src.v + dv
            with suppress(KeyError):
                tgt = node_index[target_typ, u_tgt, v_tgt][0]
                seq.append(Edge(len(seq), src, tgt, sign, n_syn, n_syn_certainty))


# -- ConnectomeView ------------------------------------------------------------


class ConnectomeView:
    """Visualization of the connectome data.

    Args:
        connectome: Directory of the connectome.
        groups: Regular expressions to sort the nodes by.

    Attributes:
        dir (ConnectomeFromAvgFilters): Connectome directory.
        edges (Directory): Edge table.
        nodes (Directory): Node table.
        cell_types_unsorted (List[str]): Unsorted list of cell types.
        cell_types_sorted (List[str]): Sorted list of cell types.
        cell_types_sort_index (List[int]): Indices for sorting cell types.
        layout (Dict[str, str]): Layout information for cell types.
        node_indexer (NodeIndexer): Indexer for nodes.
    """

    def __init__(
        self,
        connectome: ConnectomeFromAvgFilters,
        groups: List[str] = [
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
    ) -> None:
        self.dir = connectome

        assert "nodes" in self.dir and "edges" in self.dir

        self.edges = self.dir.edges
        self.nodes = self.dir.nodes

        self.cell_types_unsorted = self.dir.unique_cell_types[:].astype(str)

        (
            self.cell_types_sorted,
            self.cell_types_sort_index,
        ) = nodes_edges_utils.order_node_type_list(
            self.dir.unique_cell_types[:].astype(str), groups
        )

        self.layout = dict(self.dir.layout[:].astype(str))
        self.node_indexer = nodes_edges_utils.NodeIndexer(self.dir)

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
        """Plot the connectivity matrix as counts or weights.

        Args:
            mode: 'n_syn' for number of input synapses, 'count' for number of neurons.
            only_sign: '+' for excitatory projections, '-' for inhibitory projections.
            cell_types: Subset of nodes to display.
            no_symlog: Disable symmetric log scale.
            min_number: Minimum value to display.
            cmap: Custom colormap.
            size_scale: Size of the scattered squares.
            title: Custom title for the plot.
            cbar_label: Custom colorbar label.
            **kwargs: Additional arguments passed to the heatmap plot function.

        Returns:
            Figure: Matplotlib figure object.
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

        return plots.heatmap(matrix, cell_types, **kwargs)

    def _weights(self) -> NDArray:
        """Calculate weights for edges.

        Returns:
            NDArray: Array of edge weights.
        """
        return self.edges.sign[:] * self.edges.n_syn[:]

    def network_layout(
        self,
        max_extent: int = 5,
        **kwargs,
    ) -> Figure:
        """Plot retinotopic hexagonal lattice columnar organization of the network.

        Args:
            max_extent: Integer column radius to visualize.
            **kwargs: Additional arguments passed to hex_layout_all.

        Returns:
            Figure: Matplotlib figure object.
        """
        backbone = WholeNetworkFigure(self.dir)
        backbone.init_figure(figsize=[7, 3])
        return self.hex_layout_all(
            max_extent=max_extent, fig=backbone.fig, axes=backbone.axes, **kwargs
        )

    def hex_layout(
        self,
        cell_type: str,
        max_extent: int = 5,
        edgecolor: str = "none",
        edgewidth: float = 0.5,
        alpha: float = 1,
        fill: bool = False,
        cmap: Optional[Colormap] = None,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ) -> Figure:
        """Plot retinotopic hexagonal lattice organization of a cell type.

        Args:
            cell_type: Type of cell to plot.
            max_extent: Maximum extent of the layout.
            edgecolor: Color of the hexagon edges.
            edgewidth: Width of the hexagon edges.
            alpha: Transparency of the hexagons.
            fill: Whether to fill the hexagons.
            cmap: Custom colormap.
            fig: Existing figure to plot on.
            ax: Existing axis to plot on.
            **kwargs: Additional arguments passed to hex_scatter.

        Returns:
            Figure: Matplotlib figure object.
        """
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
            label = cell_type if cell_type not in [t.get_text() for t in ax.texts] else ""

        fig, ax, _ = plots.hex_scatter(
            u,
            v,
            values=1,
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
        cell_types: Optional[List[str]] = None,
        max_extent: int = 5,
        edgecolor: str = "none",
        alpha: float = 1,
        fill: bool = False,
        cmap: Optional[Colormap] = None,
        fig: Optional[Figure] = None,
        axes: Optional[List[Axes]] = None,
        **kwargs,
    ) -> Figure:
        """Plot retinotopic hexagonal lattice organization of all cell types.

        Args:
            cell_types: List of cell types to plot.
            max_extent: Maximum extent of the layout.
            edgecolor: Color of the hexagon edges.
            alpha: Transparency of the hexagons.
            fill: Whether to fill the hexagons.
            cmap: Custom colormap.
            fig: Existing figure to plot on.
            axes: List of existing axes to plot on.
            **kwargs: Additional arguments passed to hex_layout.

        Returns:
            Figure: Matplotlib figure object.
        """
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

    def get_uv(self, cell_type: str) -> Tuple[NDArray, NDArray]:
        """Get hex-coordinates of a particular cell type.

        Args:
            cell_type: Type of cell to get coordinates for.

        Returns:
            Tuple[NDArray, NDArray]: Arrays of u and v coordinates.
        """
        nodes = self.nodes.to_df()
        nodes = nodes[nodes.type == cell_type]
        u, v = nodes[["u", "v"]].values.T
        return u, v

    def sources_list(self, cell_type: str) -> NDArray:
        """Get presynaptic cell types.

        Args:
            cell_type: Type of cell to get sources for.

        Returns:
            NDArray: Array of presynaptic cell types.
        """
        edges = self.edges.to_df()
        return np.unique(edges[edges.target_type == cell_type].source_type.values)

    def targets_list(self, cell_type: str) -> NDArray:
        """Get postsynaptic cell types.

        Args:
            cell_type: Type of cell to get targets for.

        Returns:
            NDArray: Array of postsynaptic cell types.
        """
        edges = self.edges.to_df()
        return np.unique(edges[edges.source_type == cell_type].target_type.values)

    def receptive_field(
        self,
        source: str = "Mi9",
        target: str = "T4a",
        rfs: Optional["ReceptiveFields"] = None,
        max_extent: Optional[int] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        title: str = "{source} :→ {target}",
        **kwargs,
    ) -> Figure:
        """Plot the receptive field of a target cell type from a source cell type.

        Args:
            source: Source cell type.
            target: Target cell type.
            rfs: ReceptiveFields object. If None, it will be created.
            max_extent: Maximum extent of the receptive field.
            vmin: Minimum value for colormap.
            vmax: Maximum value for colormap.
            title: Title format string for the plot.
            **kwargs: Additional arguments passed to plots.kernel.

        Returns:
            Matplotlib Figure object.
        """
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
        sources: Optional[Iterable[str]] = None,
        sort_alphabetically: bool = True,
        ax_titles: str = "{source} :→ {target}",
        figsize: List[int] = [20, 20],
        max_extent: Optional[int] = None,
        fig: Optional[Figure] = None,
        axes: Optional[List[Axes]] = None,
        ignore_sign_error: bool = False,
        max_figure_height_cm: float = 22,
        panel_height_cm: float = 3,
        max_figure_width_cm: float = 18,
        panel_width_cm: float = 3.6,
        **kwargs,
    ) -> Figure:
        """Plot receptive fields of a target cell type in a grid layout.

        Args:
            target: Target cell type.
            sources: Iterable of source cell types. If None, all sources are used.
            sort_alphabetically: Whether to sort source types alphabetically.
            ax_titles: Title format string for each subplot.
            figsize: Figure size in inches.
            max_extent: Maximum extent of the receptive fields.
            fig: Existing figure to plot on.
            axes: List of existing axes to plot on.
            ignore_sign_error: Whether to ignore sign errors in plotting.
            max_figure_height_cm: Maximum figure height in cm.
            panel_height_cm: Height of each panel in cm.
            max_figure_width_cm: Maximum figure width in cm.
            panel_width_cm: Width of each panel in cm.
            **kwargs: Additional arguments passed to receptive_field.

        Returns:
            Matplotlib Figure object.
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
            sources, _ = nodes_edges_utils.order_node_type_list(sources)
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

    def projective_field(
        self,
        source: str = "Mi9",
        target: str = "T4a",
        title: str = "{source} →: {target}",
        prfs: Optional["ProjectiveFields"] = None,
        max_extent: Optional[int] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        **kwargs,
    ) -> Optional[Figure]:
        """Plot the projective field from a source cell type to a target cell type.

        Args:
            source: Source cell type.
            target: Target cell type.
            title: Title format string for the plot.
            prfs: ProjectiveFields object. If None, it will be created.
            max_extent: Maximum extent of the projective field.
            vmin: Minimum value for colormap.
            vmax: Maximum value for colormap.
            **kwargs: Additional arguments passed to plots.kernel.

        Returns:
            Matplotlib Figure object or None if max_extent is None.
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
        source: str,
        targets: Optional[Iterable[str]] = None,
        fig: Optional[Figure] = None,
        axes: Optional[List[Axes]] = None,
        figsize: List[int] = [20, 20],
        ax_titles: str = "{source} →: {target}",
        max_figure_height_cm: float = 22,
        panel_height_cm: float = 3,
        max_figure_width_cm: float = 18,
        panel_width_cm: float = 3.6,
        max_extent: Optional[int] = None,
        sort_alphabetically: bool = False,
        ignore_sign_error: bool = False,
        **kwargs,
    ) -> Figure:
        """Plot projective fields of a source cell type in a grid layout.

        Args:
            source: Source cell type.
            targets: Iterable of target cell types. If None, all targets are used.
            fig: Existing figure to plot on.
            axes: List of existing axes to plot on.
            figsize: Figure size in inches.
            ax_titles: Title format string for each subplot.
            max_figure_height_cm: Maximum figure height in cm.
            panel_height_cm: Height of each panel in cm.
            max_figure_width_cm: Maximum figure width in cm.
            panel_width_cm: Width of each panel in cm.
            max_extent: Maximum extent of the projective fields.
            sort_alphabetically: Whether to sort target types alphabetically.
            ignore_sign_error: Whether to ignore sign errors in plotting.
            **kwargs: Additional arguments passed to projective_field.

        Returns:
            Matplotlib Figure object.
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
            targets, _ = nodes_edges_utils.order_node_type_list(targets)

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

    def receptive_fields_df(self, target_type: str) -> "ReceptiveFields":
        """Get receptive fields for a target cell type.

        Args:
            target_type: Target cell type.

        Returns:
            ReceptiveFields object.
        """
        return ReceptiveFields(target_type, self.edges.to_df())

    def projective_fields_df(self, source_type: str) -> "ProjectiveFields":
        """Get projective fields for a source cell type.

        Args:
            source_type: Source cell type.

        Returns:
            ProjectiveFields object.
        """
        return ProjectiveFields(source_type, self.edges.to_df())

    def receptive_fields_sum(self, target_type: str) -> Dict[str, int]:
        """Get sum of synapses for each source type in the receptive field.

        Args:
            target_type: Target cell type.

        Returns:
            Dictionary mapping source types to synapse counts.
        """
        return ReceptiveFields(target_type, self.edges.to_df()).sum()

    def projective_fields_sum(self, source_type: str) -> Dict[str, int]:
        """Get sum of synapses for each target type in the projective field.

        Args:
            source_type: Source cell type.

        Returns:
            Dictionary mapping target types to synapse counts.
        """


class ReceptiveFields(Namespace):
    """Dictionary of receptive field dataframes for a specific cell type.

    Args:
        target_type: Target cell type.
        edges: All edges of a Connectome.

    Attributes:
        target_type: The target cell type.
        source_types: List of source cell types.
        _extents: List of extents for each source type.

    Example:
        ```python
        rf = ReceptiveFields("T4a", edges_dataframe)
        ```
    """

    def __init__(self, target_type: str, edges: DataFrame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_extents", [])
        _receptive_fields_edge_dfs(self, target_type, edges)

    @property
    def extents(self) -> Dict[str, int]:
        """Dictionary of extents for each source type."""
        return dict(zip(self.source_types, self._extents))

    @property
    def max_extent(self) -> Optional[int]:
        """Maximum extent across all source types."""
        return max(self._extents) if self._extents else None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.target_type})"

    def sum(self) -> Dict[str, float]:
        """Sum of synapses for each source type."""
        return {key: self[key].n_syn.sum() for key in self}


class ProjectiveFields(Namespace):
    """Dictionary of projective field dataframes for a specific cell type.

    Args:
        source_type: Source cell type.
        edges: All edges of a Connectome.

    Attributes:
        source_type: The source cell type.
        target_types: List of target cell types.
        _extents: List of extents for each target type.

    Example:
        ```python
        pf = ProjectiveFields("Mi9", edges_dataframe)
        ```
    """

    def __init__(self, source_type: str, edges: DataFrame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_extents", [])
        _projective_fields_edge_dfs(self, source_type, edges)

    @property
    def extents(self) -> Dict[str, int]:
        """Dictionary of extents for each target type."""
        return dict(zip(self.target_types, self._extents))

    @property
    def max_extent(self) -> Optional[int]:
        """Maximum extent across all target types."""
        return max(self._extents) if self._extents else None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.source_type})"

    def sum(self) -> Dict[str, float]:
        """Sum of synapses for each target type."""
        return {key: self[key].n_syn.sum() for key in self}


def _receptive_fields_edge_dfs(
    cls: ReceptiveFields, target_type: str, edges: DataFrame
) -> ReceptiveFields:
    """Populate ReceptiveFields with edge dataframes.

    Args:
        cls: ReceptiveFields instance to populate.
        target_type: Target cell type.
        edges: DataFrame containing all edges.

    Returns:
        Populated ReceptiveFields instance.
    """
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
    """Populate ProjectiveFields with edge dataframes.

    Args:
        cls: ProjectiveFields instance to populate.
        source_type: Source cell type.
        edges: DataFrame containing all edges.

    Returns:
        Populated ProjectiveFields instance.
    """
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


def get_avgfilt_connectome(config: dict) -> ConnectomeView:
    """Create a ConnectomeView instance from a config for ConnectomeFromAvgFilters.

    Args:
        config: Containing ConnectomeFromAvgFilters configuration.

    Returns:
        ConnectomeView instance.
    """
    return ConnectomeView(ConnectomeFromAvgFilters(**config))


def is_connectome_protocol(obj: Any) -> bool:
    """
    Check if an object implements the Connectome(Protocol).

    Args:
        obj: The object to check.

    Returns:
        bool: True if the object implements the Connectome(Protocol), False otherwise.

    Note:
        The Connectome(Protocol) requires the following attributes:
        - nodes.index: Union[np.ndarray, ArrayFile]
        - edges.source_index: Union[np.ndarray, ArrayFile]
        - edges.target_index: Union[np.ndarray, ArrayFile]
    """
    if not hasattr(obj, 'nodes'):
        return False, "Missing 'nodes' attribute"
    if not hasattr(obj.nodes, 'index'):
        return False, "Missing 'nodes.index' attribute"
    if not isinstance(obj.nodes.index, (np.ndarray, ArrayFile)):
        return False, "'nodes.index' is not of type np.ndarray or ArrayFile"
    if not hasattr(obj, 'edges'):
        return False, "Missing 'edges' attribute"
    if not hasattr(obj.edges, 'source_index'):
        return False, "Missing 'edges.source_index' attribute"
    if not isinstance(obj.edges.source_index, (np.ndarray, ArrayFile)):
        return False, "'edges.source_index' is not of type np.ndarray or ArrayFile"
    if not hasattr(obj.edges, 'target_index'):
        return False, "Missing 'edges.target_index' attribute"
    if not isinstance(obj.edges.target_index, (np.ndarray, ArrayFile)):
        return False, "'edges.target_index' is not of type np.ndarray or ArrayFile"
    return True, ""


def init_connectome(**kwargs) -> Connectome:
    """Initialize a Connectome instance from a config dictionary.

    Args:
        config: A dictionary containing the connectome configuration.

    Returns:
        An instance of a class implementing the Connectome(Protocol).

    Raises:
        KeyError: If the specified connectome type is not available.

    Example:
        ```python
        config = {
            "type": "ConnectomeFromAvgFilters",
            **config
        }
        connectome = init_connectome(**config)
        ```
    """
    connectome_class = AVAILABLE_CONNECTOMES[kwargs.pop("type")]

    connectome = connectome_class(**kwargs)
    is_valid, error_msg = is_connectome_protocol(connectome)
    assert is_valid, (
        f"Connectome class {connectome} does "
        f"not implement the Connectome(Protocol): {error_msg}"
    )
    return connectome
