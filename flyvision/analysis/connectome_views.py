from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps as cm
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.axis import Axis
from matplotlib.colorbar import Colorbar
from toolz import valmap

from dvs import Namespace
from dvs import utils
from dvs.plots import plots
from dvs.plots import plt_utils


class ConnectomeViews:
    """Wrapper for plots of the connectome.

    Args:
        wrap (Datawrap): wrap of the trained dvs model
                                    or a connectome.
        groups (List[str]): regular expressions to sort the nodes by.

    Parameters:
        ctome (Connectome): wrap of connectome.
        nodes (DataFrame): node table.
        edges (DataFrame): edge table.

    Note: differs between Connectome and NetworkWrap. NetworkWraps can
        contain a 'prior_param_api' and a 'param_api' in which case the
        DataFrames are populated with prior and trained parameter columns.
    """

    def __init__(
        self,
        wrap,
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
        if "ctome" in wrap:
            self.ctome = wrap.ctome
        else:
            self.ctome = wrap
        assert "nodes" in self.ctome and "edges" in self.ctome

        self.edges = self.ctome.edges

        self.nodes = self.ctome.nodes
        if "prior_param_api" in wrap:
            self.nodes.update(wrap.prior_param_api.nodes, suffix="_prior")
            self.edges.update(wrap.prior_param_api.edges, suffix="_prior")
            if "weight_prior" not in self.edges:
                self.edges.weight_prior = self._weights(n_syn=False, trained=False)

        if "param_api" in wrap:
            self.nodes.update(wrap.param_api.nodes, suffix="_trained")
            self.edges.update(wrap.param_api.edges, suffix="_trained")
            if "weight_trained" not in self.edges:
                self.edges.weight_trained = self._weights(n_syn=False, trained=True)

        self.node_types_unsorted = self.ctome.unique_node_types[:].astype(str)

        (
            self.node_types_sorted,
            self.node_types_sort_index,
        ) = utils.order_nodes_list(self.ctome.unique_node_types[:].astype(str))

        if "layout" not in self.ctome:
            layout = []
            layout.extend(
                list(
                    zip(
                        self.ctome.input_node_types,
                        [b"retina" for _ in range(len(self.ctome.input_node_types))],
                    )
                )
            )
            layout.extend(
                list(
                    zip(
                        self.ctome.intermediate_node_types,
                        [
                            b"intermediate"
                            for _ in range(len(self.ctome.intermediate_node_types))
                        ],
                    )
                )
            )
            layout.extend(
                list(
                    zip(
                        self.ctome.output_node_types,
                        [b"output" for _ in range(len(self.ctome.output_node_types))],
                    )
                )
            )
            self.ctome.layout = np.string_(layout)
        self.layout = dict(self.ctome.layout[:].astype(str))
        self.node_indexer = utils.nodes_edges_utils.NodeIndexer(self.ctome)

    def _weights(self, trained=False, n_syn=False):
        if n_syn and not trained:
            return self.edges.sign[:] * self.edges.n_syn[:]
        elif n_syn and trained:
            return (
                self.edges.syn_count_trained[:]
                * self.edges.sign_trained[:]
                * self.edges.syn_strength_trained[:]
                / (self.edges.syn_strength_prior[:] + 1e-15)
            )
        elif not n_syn and not trained:
            return (
                self.edges.syn_count_prior[:]
                * self.edges.syn_strength_prior[:]
                * self.edges.sign_prior[:]
            )
        elif not n_syn and trained:
            return (
                self.edges.syn_count_trained[:]
                * self.edges.syn_strength_trained[:]
                * self.edges.sign_trained[:]
            )

    def get_uv(self, node_type):
        """return coordinates of a particular node type"""
        nodes = self.nodes.to_df()
        nodes = nodes[nodes.type == node_type]
        u, v = nodes[["u", "v"]].values.T
        return u, v

    # ---- CONNECTIVITY MATRIX

    def connectivity_matrix(
        self,
        plot_type="count",
        only_sign=None,
        trained=False,
        heatmap=True,
        cmap=None,
        size_scale=None,
        title=None,
        cbar_label=None,
        node_types=None,
        no_symlog=False,
        min_number=None,
        **kwargs,
    ) -> Tuple[Figure, Axis, Colorbar, np.ndarray]:
        """Plots the connectivity matrix as counts or weights.

        Args:
            plot_type (str): 'count' referring to number of input neurons,
                 'weight' referring to summed weight per kernel, 'n_syn' referring to number of input synapses.
        """

        _kwargs = dict(
            n_syn=dict(
                symlog=1e-5,
                # midpoint=0,
                grid=True,
                cmap=cmap or cm.get_cmap("seismic"),
                title=title or "Number of Input Synapses",
                cbar_label=cbar_label or r"$\pm\sum_{pre} N_\mathrm{syn.}^{pre, post}$",
                size_scale=size_scale or 0.05,
            ),
            weight=dict(
                symlog=1e-5,
                grid=True,
                cmap=cmap or cm.get_cmap("seismic"),
                title=title or "Summed Weight per Kernel",
                cbar_label=r"$\sum_{pre} w^{pre, post}$",
                size_scale=size_scale or 2,
            ),
            count=dict(
                grid=True,
                cmap=cmap or cm.get_cmap("seismic"),
                midpoint=0,
                title=title or "Number of Input Neurons",
                cbar_label=cbar_label or "$\sum_{pre} 1$",
                size_scale=size_scale or 0.05,
            ),
        )
        n_syn = True if plot_type in ["n_syn", "count"] else False
        kwargs.update(_kwargs[plot_type])
        if no_symlog:
            kwargs.update(symlog=None)
            kwargs.update(midpoint=0)

        # Get edges to central nodes.
        edges = self.edges.to_df()
        edges = edges[(edges.target_u == 0) & (edges.target_v == 0)]
        node_types = node_types or self.node_types_sorted

        edges = utils.filter_df_by_list(
            node_types,
            utils.filter_df_by_list(node_types, edges, column="source_type"),
            column="target_type",
        )

        type_index = {
            node_typ: i for i, node_typ in enumerate(node_types)
        }  # lookup table for key -> (i, j)

        matrix = np.zeros([len(type_index), len(type_index)])
        weights = self._weights(trained, n_syn)[edges.index]

        for srctyp, tgttyp, weight in zip(
            edges.source_type.values, edges.target_type.values, weights
        ):
            if plot_type == "count":
                # count the number of nodes
                matrix[type_index[srctyp], type_index[tgttyp]] += 1
            elif plot_type in ["weight", "n_syn"]:
                matrix[type_index[srctyp], type_index[tgttyp]] += weight
            else:
                raise ValueError

        # kick out all that fall below min_number
        if min_number is not None:
            matrix[np.abs(matrix) <= min_number] = np.nan

        if only_sign == "+":
            matrix[matrix < 0] = 0
            kwargs.update(symlog=None, midpoint=0)
        if only_sign == "-":
            matrix[matrix > 0] = 0
            kwargs.update(symlog=None, midpoint=0)

        if heatmap:
            return plots.heatmap(matrix, node_types, **kwargs)
        return plots.heatmap_uniform(matrix, node_types, **kwargs)

    # ---- NODE LAYOUTS

    def node_layout(
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
        Plot the node_types layout on the regular hex grid.
        """
        nodes = self.nodes.to_df()
        node_condition = nodes.type == node_type
        u, v = nodes.u[node_condition], nodes.v[node_condition]
        max_extent = utils.get_extent(u, v) if max_extent is None else max_extent
        extent_condition = (
            (-max_extent <= u)
            & (u <= max_extent)
            & (-max_extent <= v)
            & (v <= max_extent)
            & (-max_extent <= u + v)
            & (u + v <= max_extent)
        )
        u, v = u[extent_condition].values, v[extent_condition].values
        return plots.hex_scatter(
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

    def node_layout_all(
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
        """Plot all node type layouts."""
        node_types = self.node_types_sorted if node_types is None else node_types
        if not (fig and axes):
            fig, axes, (gw, gh) = plt_utils.get_axis_grid(self.node_types_sorted)
        if anatomic_order:
            node_types = [key for key in self.layout.keys() if key in node_types]
        for i, node_type in enumerate(node_types):
            self.node_layout(
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
        return fig, axes

    # ---- NETWORK GRAPHS

    def node_layout_graph(
        self,
        node_types=None,
        edge_color_key="sign",
        edge_cmap=cm.get_cmap("seismic"),
        max_extent=5,
        fig=None,
        **kwargs,
    ):
        """Plot all node type layouts in a graph."""
        import networkx as nx
        from matplotlib.patches import Circle

        fig = fig or plt.figure(figsize=[20, 10])

        if node_types:
            nodes = utils.filter_df_by_list(
                node_types, self.nodes.to_df(), column="type"
            )
            edges = utils.filter_df_by_list(
                node_types, self.edges.to_df(), column="source_type"
            )
            edges = utils.filter_df_by_list(
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
        _ = self.node_layout_all(
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

        if edge_color_key is not None:
            edges = edges.groupby(
                by=["source_type", "target_type"], sort=False, as_index=False
            ).mean()
            edge_color = edges[edge_color_key].values
            edge_vmin = -np.max(edge_color) if np.any(edge_color < 0) else 0
            edge_vmax = np.max(edge_color)
        else:
            edge_color = "black"
            edge_vmin = None
            edge_vmax = None
        nodes, vertices = plots._network_graph(nodes, edges)
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(vertices)
        edge_ax = fig.add_axes([0, 0, 1, 1])
        edge_ax.set_zorder(0)
        edge_ax = plt_utils.rm_spines(edge_ax, rm_xticks=True, rm_yticks=True)
        edge_ax.patch.set_alpha(0.0)

        nx.draw_networkx_edges(
            graph,
            pos=new_pos,
            ax=edge_ax,
            width=2 * edges["n_syn"].values / edges["n_syn"].values.max(),
            edge_color=edge_color,
            edge_cmap=edge_cmap,
            edge_vmin=edge_vmin,
            edge_vmax=edge_vmax,
            alpha=0.2,
        )
        edge_sm, edge_norm = plt_utils.get_scalarmapper(
            cmap=edge_cmap, vmin=edge_vmin, vmax=edge_vmax
        )
        # _ = nx.draw_networkx_edges(graph, pos=new_pos, ax=edge_ax, alpha=0.1)

        # hacky way to add reccurent connections
        for a, b in graph.edges:
            if a == b:
                x, y = new_pos[a]
                x += 0.02
                y += 0.01
                cond = (edges.source_type == a) & (edges.target_type == b)
                sign = edges[cond][edge_color_key]
                edge_color = edge_sm.to_rgba(sign.item()) if edge_color_key else "black"
                width = 2 * edges[cond]["n_syn"].values / edges["n_syn"].values.max()
                edge_ax.add_artist(
                    Circle(
                        [x, y],
                        radius=0.02,
                        linewidth=width,
                        facecolor="None",
                        edgecolor=edge_color,
                    )
                )

        return fig, axes

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
        n_syn=True,
        rfs=None,
        max_extent=None,
        vmin=None,
        vmax=None,
        title="{source} :→ {target}",
        trained=False,
        **kwargs,
    ):
        """
        Plots the receptive field from 'taregt' from 'source'.
        """
        if rfs is None:
            rfs = receptive_fields_edge_dfs(target, self.edges.to_df())
            max_extent = max_extent or rfs.max_extent
        # weights
        weights = self._weights(trained, n_syn)

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
        plt_utils.patch_type_texts(ax)
        return fig, ax, (label_text, scalarmapper)

    def receptive_fields_grid(
        self,
        target,
        sources=None,
        sort_alphabetically=True,
        scale=5,
        aspect_ratio=1,
        trained=False,
        n_syn=True,
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

        rfs = receptive_fields_edge_dfs(target, self.edges.to_df())
        max_extent = max_extent or rfs.max_extent
        weights = self._weights(trained, n_syn)

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
            sources, _ = utils.nodes_edges_utils.order_nodes_list(sources)
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
                    trained=trained,
                    n_syn=n_syn,
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
        return fig, axes, (vmin, vmax)

    # ---- PROJECTIVE FIELDS

    def projective_field(
        self,
        source="Mi9",
        target="T4a",
        n_syn=True,
        title="{source} →: {target}",
        trained=False,
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
            prfs = projective_fields_edge_dfs(source, self.edges.to_df())
            max_extent = max_extent or prfs.max_extent
        if max_extent is None:
            return None
        weights = self._weights(trained, n_syn)

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
        plt_utils.patch_type_texts(ax)
        return fig, ax, (label_text, scalarmapper)

    def projective_fields_grid(
        self,
        source,
        targets=None,
        scale=5,
        fig=None,
        axes=None,
        aspect_ratio=1,
        figsize=[20, 20],
        n_syn=True,
        ax_titles="{source} →: {target}",
        max_h_axes=None,
        max_v_axes=None,
        hspace=0,
        wspace=0.0,
        min_axes=-1,
        keep_nan_axes=True,
        trained=False,
        max_extent=None,
        sort_alphabetically=False,
        **kwargs,
    ):
        """
        Plots all projective field of 'source' inside a regular grid of axes.
        """
        prfs = projective_fields_edge_dfs(source, self.edges.to_df())
        max_extent = max_extent or prfs.max_extent
        weights = self._weights(trained, n_syn)
        sorted_sum_of_outputs = dict(
            sorted(
                valmap(lambda v: weights[v.index].sum(), prfs).items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        # to sort alphabetically in case sources is specified
        if sort_alphabetically:
            targets, _ = utils.nodes_edges_utils.order_nodes_list(targets)

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

        for i, target in enumerate(targets):
            self.projective_field(
                source=source,
                target=target,
                fig=fig,
                ax=axes[i],
                title=ax_titles,
                trained=trained,
                n_syn=n_syn,
                prfs=prfs,
                max_extent=max_extent,
                vmin=vmin,
                vmax=vmax,
                annotate_coords=False,
                **kwargs,
            )
        return fig, axes, (vmin, vmax)

    def filter_df_by_list(self, startswith, node_types):
        """Some logic to filter the node-frame by type in a convenient way."""
        if node_types == "input":
            node_types = self.ctome.input_node_types[:].astype(str)
        elif node_types == "output":
            node_types = self.ctome.output_node_types[:].astype(str)
        elif not node_types:
            _types = self.node_types_sorted
            _match = np.core.defchararray.find(_types, startswith) == 0
            node_types = _types[_match]
        assert isinstance(node_types, Iterable)
        valid_node_types = [nt for nt in node_types if nt in self.node_types_sorted]
        return (
            utils.filter_df_by_list(node_types, self.nodes.to_df()),
            valid_node_types,
        )

    def receptive_fields_df(self, target_type):
        return receptive_fields_edge_dfs(target_type, self.edges.to_df())

    def projective_fields_df(self, source_type):
        return projective_fields_edge_dfs(source_type, self.edges.to_df())

    def receptive_fields_sum(self, target_type):
        return receptive_fields_sum(target_type, self.edges.to_df())

    def projective_fields_sum(self, source_type):
        return projective_fields_sum(source_type, self.edges.to_df())


# -- other plots


def input_T4(edges):
    """Shinomiya et al. 2019 - Figure 2"""
    cells = [
        "Mi1",
        "Mi4",
        "Mi9",
        "Mi13",
        "Tm3",
        "CT1(M10)",
        "TmY15",
        "TmY18",  # equals TmY18
        "C3",
        "T4a",
        "T4b",
        "T4c",
        "T4d",
        "other",
    ]
    n_syn = np.zeros([4, len(cells), 1])
    other = {}
    for i, t4typ in enumerate(["T4a", "T4b", "T4c", "T4d"]):
        other[t4typ] = []
        rfs = receptive_fields_edge_dfs(t4typ, edges)
        for j, cell in enumerate(cells[:-1]):
            if cell in rfs:
                n_syn[i, j] = np.sum(rfs[cell].n_syn)
            else:
                n_syn[i, j] = np.nan
        for cell in rfs:
            if cell not in cells:
                n_syn[i, -1] += np.sum(rfs[cell].n_syn)
                other[t4typ].append(cell)
    n_syn = np.swapaxes(n_syn, 0, 1)

    fig, ax, C = plots.violin_groups(
        n_syn,
        xticklabels=cells,
        as_bars=True,
        colors=["#2AB155", "#EB4E2F", "#FAD23A", "#2174B6"],
        legend=["T4a", "T4b", "T4c", "T4d"],
        legend_kwargs={"fontsize": 10, "loc": 1},
        width=0.7,
    )
    ax.set_ylabel("number of synapses")
    fig.suptitle("input to T4", x=0.25, y=0.75)
    return fig, ax


def output_T4(edges, invert_y=False, ylim=None):
    """Shinomiya et al. 2019 - Figure 2"""
    cells = [
        "Mi1",
        "Mi4",
        "Mi9",
        "Mi13",
        "Tm3",
        "CT1(M10)",
        "TmY15",
        "TmY18",  # equals TmY18
        "C3",
        "T4a",
        "T4b",
        "T4c",
        "T4d",
        "other",
    ]
    n_syn = np.zeros([4, len(cells), 1])
    other = {}
    for i, t4typ in enumerate(["T4a", "T4b", "T4c", "T4d"]):
        other[t4typ] = []
        prfs = projective_fields_edge_dfs(t4typ, edges)
        for j, cell in enumerate(cells[:-1]):
            if cell in prfs:
                n_syn[i, j] = np.sum(prfs[cell].n_syn)
            else:
                n_syn[i, j] = np.nan
        for cell in prfs:
            if cell not in cells:
                n_syn[i, -1] += np.sum(prfs[cell].n_syn)
                other[t4typ].append(cell)
    n_syn = np.swapaxes(n_syn, 0, 1)
    fig, ax, C = plots.violin_groups(
        n_syn,
        xticklabels=cells,
        as_bars=True,
        colors=["#2AB155", "#EB4E2F", "#FAD23A", "#2174B6"],
        legend=["T4a", "T4b", "T4c", "T4d"],
        legend_kwargs={"fontsize": 10, "loc": 2},
        width=0.7,
    )
    ax.set_ylabel("number of synapses")
    fig.suptitle("output from T4", y=0.25, x=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if invert_y:
        ax.invert_yaxis()
        plt_utils.rm_spines(ax, ["bottom"], rm_xticks=True)
    return fig, ax


def input_T5(edges):
    """Shinomiya et al. 2019 - Figure 2"""
    cells = [
        "Tm1",
        "Tm2",
        "Tm4",
        "Tm9",
        "CT1(Lo1)",
        "TmY15",
        "LT33",
        "Tm23",
        "T2",
        "T5a",
        "T5b",
        "T5c",
        "T5d",
        "other",
    ]
    n_syn = np.zeros([4, len(cells), 1])
    other = {}
    for i, t5typ in enumerate(["T5a", "T5b", "T5c", "T5d"]):
        other[t5typ] = []
        rfs = receptive_fields_edge_dfs(t5typ, edges)
        for j, cell in enumerate(cells[:-1]):
            if cell in rfs:
                n_syn[i, j] = np.sum(rfs[cell].n_syn)
            else:
                n_syn[i, j] = np.nan
        for cell in rfs:
            if cell not in cells:
                n_syn[i, -1] += np.sum(rfs[cell].n_syn)
                other[t5typ].append(cell)
    n_syn = np.swapaxes(n_syn, 0, 1)
    fig, ax, C = plots.violin_groups(
        n_syn,
        xticklabels=cells,
        as_bars=True,
        colors=["#2AB155", "#EB4E2F", "#FAD23A", "#2174B6"],
        legend=["T5a", "T5b", "T5c", "T5d"],
        legend_kwargs={"fontsize": 10, "loc": 2},
        width=0.7,
    )
    ax.set_ylabel("number of synapses")
    fig.suptitle("input to T5", x=0.25, y=0.75)
    return fig, ax


def output_T5(edges, invert_y=False, ylim=None, fig=None, ax=None):
    """Shinomiya et al. 2019 - Figure 2"""
    cells = [
        "Tm1",
        "Tm2",
        "Tm4",
        "Tm9",
        "CT1(Lo1)",
        "TmY15",
        "LT33",
        "Tm23",
        "T2",
        "T5a",
        "T5b",
        "T5c",
        "T5d",
        "other",
    ]
    n_syn = np.zeros([4, len(cells), 1])
    other = {}
    for i, t5typ in enumerate(["T5a", "T5b", "T5c", "T5d"]):
        other[t5typ] = []
        prfs = projective_fields_edge_dfs(t5typ, edges)
        for j, cell in enumerate(cells[:-1]):
            if cell in prfs:
                n_syn[i, j] = np.sum(prfs[cell].n_syn)
            else:
                n_syn[i, j] = np.nan
        for cell in prfs:
            if cell not in cells:
                n_syn[i, -1] += np.sum(prfs[cell].n_syn)
                other[t5typ].append(cell)
    n_syn = np.swapaxes(n_syn, 0, 1)
    fig, ax, C = plots.violin_groups(
        n_syn,
        xticklabels=cells,
        as_bars=True,
        colors=["#2AB155", "#EB4E2F", "#FAD23A", "#2174B6"],
        legend=["T5a", "T5b", "T5c", "T5d"],
        legend_kwargs={"fontsize": 10, "loc": 2},
        width=0.7,
        fig=fig,
        ax=ax,
    )
    ax.set_ylabel("number of synapses")
    fig.suptitle("output from T5", y=0.25, x=0.25)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if invert_y:
        ax.invert_yaxis()
        plt_utils.rm_spines(ax, ["bottom"], rm_xticks=True)
    return fig, ax


# -- utility functions
class ReceptiveFields(Namespace):
    "Mapping of source types to pandas DataFrames"

    def __init__(self, target_type, source_types, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "target_type", target_type)
        object.__setattr__(self, "source_types", source_types)
        object.__setattr__(self, "_extents", [])

    @property
    def extents(self):
        return dict(zip(self.source_types, self._extents))

    @property
    def max_extent(self):
        return max(self._extents) if self._extents else None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.target_type})"


def receptive_fields_edge_dfs(target_type, edges) -> ReceptiveFields:
    """Return all receptive fields edges as separate dataframes.

    Args:
        target_type (str): name of target neuron type.
        edges (DataFrame): edges DataFrame.

    Returns:
        dict: receptive fields dict with source neuron type - edge dataframe
                pairs, and all columns contained in the edges dataframe.
        int: maximal extent of all source receptive fields for plotting on
                the same grid.
    """

    edges = edges[edges.target_type == target_type]
    source_types = edges.source_type.unique()

    rfs = ReceptiveFields(target_type, source_types)

    for source_type in source_types:
        _edges = edges[edges.source_type == source_type]

        most_central_edge = _edges.iloc[
            np.argmin(np.abs(_edges.target_u) + np.abs(_edges.target_v))
        ]
        target_u_min = most_central_edge.target_u
        target_v_min = most_central_edge.target_v

        rfs[source_type] = _edges[
            (_edges.target_u == target_u_min) & (_edges.target_v == target_v_min)
        ]
        rfs._extents.append(utils.get_extent(rfs[source_type].du, rfs[source_type].dv))
    # max_extent = max(extents) if extents else None

    return rfs


class ProjectiveFields(Namespace):
    "Mapping of source types to pandas DataFrames"

    def __init__(self, source_type, target_types, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "source_type", source_type)
        object.__setattr__(self, "target_types", target_types)
        object.__setattr__(self, "_extents", [])

    @property
    def extents(self):
        return dict(zip(self.target_types, self._extents))

    @property
    def max_extent(self):
        return max(self._extents) if self._extents else None

    def __repr__(self):
        return f"{self.__class__.__name__}({self.source_type})"


def projective_fields_edge_dfs(source_type, edges) -> ProjectiveFields:
    """Return all projective fields edges as separate dataframes.

    Args:
        source_type (str): name of target neuron type.
        edges (DataFrame): edges DataFrame.

    Returns:
        dict: projective fields dict with target neuron type - edge dataframe
                pairs, and all columns contained in the edges dataframe.
        int: maximal extent of all target projective fields for plotting on
                the same grid.
    """

    edges = edges[edges.source_type == source_type]
    target_types = edges.target_type.unique()

    prfs = ProjectiveFields(source_type, target_types)

    for target_type in target_types:
        _edges = edges[edges.target_type == target_type]
        most_central_edge = _edges.iloc[
            np.argmin(np.abs(_edges.source_u) + np.abs(_edges.source_v))
        ]
        source_u_min = most_central_edge.source_u
        source_v_min = most_central_edge.source_v
        prfs[target_type] = _edges[
            (_edges.source_u == source_u_min) & (_edges.source_v == source_v_min)
        ]
        prfs._extents.append(
            utils.get_extent(prfs[target_type].du, prfs[target_type].dv)
        )

    # max_extent = max(extents) if extents else None

    return prfs  # , max_extent


def receptive_fields_sum(target, edges):
    rfs = receptive_fields_edge_dfs(target, edges)
    return {key: rfs[key].n_syn.sum() for key in rfs}


def projective_fields_sum(source, edges):
    prfs = projective_fields_edge_dfs(source, edges)
    return {key: prfs[key].n_syn.sum() for key in prfs}
