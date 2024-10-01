import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from flyvision.plots import plots, plt_utils
from flyvision.utils import hex_utils


def loss_curves(
    losses,
    smooth=0.05,
    subsample=1,
    mean=False,
    grid=True,
    colors=None,
    cbar=False,
    cmap=None,
    norm=None,
    fig=None,
    ax=None,
    xlabel=None,
    ylabel=None,
):
    """Plot loss traces.

    Args:
        losses: tensor of shape (n_models, n_iters)
        smooth: smoothing factor
        subsample: subsample factor
        mean: plot mean
        grid: show grid
        colors: list of colors
        cbar: add colorbar
        cmap: colormap
        norm: normalization
        fig: figure
        ax: axis
    """
    losses = np.array([loss[::subsample] for loss in losses])

    max_n_iters = max([len(loss) for loss in losses])

    _losses = np.zeros([len(losses), max_n_iters]) * np.nan
    for i, loss in enumerate(losses):
        n_iters = len(loss)
        _losses[i, :n_iters] = loss[:]
    fig, ax, _, _ = plots.traces(
        _losses[::-1],
        x=np.arange(max_n_iters) * subsample,
        fontsize=5,
        figsize=[1.2, 1],
        smooth=smooth,
        fig=fig,
        ax=ax,
        color=colors[::-1],
        linewidth=0.5,
        highlight_mean=mean,
    )

    ax.set_ylabel(ylabel, fontsize=5)
    ax.set_xlabel(xlabel, fontsize=5)

    if cbar and cmap is not None and norm is not None:
        plt_utils.add_colorbar_to_fig(
            fig,
            cmap=cmap,
            norm=norm,
            label="min task error",
            fontsize=5,
            tick_length=1,
            tick_width=0.5,
            x_offset=2,
            y_offset=0.25,
        )

    if grid:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.grid(True, linewidth=0.5)

    return fig, ax


def histogram(
    array,
    bins=None,
    fill=False,
    histtype="step",
    figsize=[1, 1],
    fontsize=5,
    fig=None,
    ax=None,
    xlabel=None,
    ylabel=None,
):
    fig, ax = plt_utils.init_plot(figsize=figsize, fontsize=fontsize, fig=fig, ax=ax)
    ax.hist(
        array,
        bins=bins if bins is not None else len(array),
        linewidth=0.5,
        fill=fill,
        histtype=histtype,
    )
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    return fig, ax


def violins(
    variable_names,
    variable_values,
    ylabel=None,
    title=None,
    max_per_ax=20,
    colors=None,
    cmap=plt.cm.viridis_r,
    fontsize=5,
    violin_width=0.7,
    legend=None,
    scatter_extent=[-0.35, 0.35],
    figwidth=10,
    fig=None,
    axes=None,
    ylabel_offset=0.2,
    **kwargs,
):
    """ """

    # variable first, samples second
    variable_values = variable_values.T
    if len(variable_values.shape) == 2:
        # add empty group dimension
        variable_values = variable_values[:, None]

    n_variables, n_groups, n_samples = variable_values.shape
    if max_per_ax is None:
        max_per_ax = n_variables
    max_per_ax = min(max_per_ax, n_variables)
    n_axes = int(n_variables / max_per_ax)
    max_per_ax += int(np.ceil((n_variables % max_per_ax) / n_axes))

    # breakpoint()
    fig, axes, _ = plt_utils.get_axis_grid(
        gridheight=n_axes,
        gridwidth=1,
        figsize=[figwidth, n_axes * 1.2],
        hspace=1,
        alpha=0,
        fig=fig,
        axes=axes,
    )

    for i in range(n_axes):
        ax_values = variable_values[i * max_per_ax : (i + 1) * max_per_ax]
        ax_names = variable_names[i * max_per_ax : (i + 1) * max_per_ax]

        fig, ax, C = plots.violin_groups(
            ax_values,
            ax_names,
            rotation=90,
            scatter=False,
            fontsize=fontsize,
            width=violin_width,
            scatter_edge_color="white",
            scatter_radius=5,
            scatter_edge_width=0.25,
            cdist=100,
            colors=colors,
            cmap=cmap,
            showmedians=True,
            showmeans=False,
            violin_marker_lw=0.25,
            legend=(legend if legend else None if i == 0 else None),
            legend_kwargs=dict(
                fontsize=5,
                markerscale=10,
                loc="lower left",
                bbox_to_anchor=(0.75, 0.75),
            ),
            fig=fig,
            ax=axes[i],
            **kwargs,
        )

        violin_locations, _ = plots.get_violin_x_locations(
            n_groups, len(ax_names), violin_width
        )

        for group in range(n_groups):
            plt_utils.scatter_on_violins_or_bars(
                ax_values[:, group].T,
                ax,
                xticks=violin_locations[group],
                facecolor="none",
                edgecolor="k",
                zorder=100,
                alpha=0.35,
                uniform=scatter_extent,
                marker="o",
                linewidth=0.5,
            )

        ax.grid(False)

        plt_utils.trim_axis(ax, yaxis=False)
        plt_utils.set_spine_tick_params(
            ax,
            tickwidth=0.5,
            ticklength=3,
            ticklabelpad=2,
            spinewidth=0.5,
        )

    # since axes are split, we need to manually add the ylabel
    lefts, bottoms, rights, tops = np.array([ax.get_position().extents for ax in axes]).T
    fig.text(
        lefts.min() - ylabel_offset * lefts.min(),
        (tops.max() - bottoms.min()) / 2,
        ylabel,
        rotation=90,
        fontsize=fontsize,
        ha="right",
        va="center",
    )

    # top axis gets the title
    axes[0].set_title(title, y=0.91, fontsize=fontsize)

    return fig, axes


def plot_strf(
    time,
    rf,
    hlines=True,
    vlines=True,
    time_axis=True,
    fontsize=6,
    fig=None,
    axes=None,
    figsize=[5, 1],
    wspace=0,
    y_offset_time_axis=0,
):
    max_extent = hex_utils.get_hextent(rf.shape[-1])
    t_steps = np.arange(0.0, 0.2, 0.01)[::2]

    u, v = hex_utils.get_hex_coords(max_extent)
    x, y = hex_utils.hex_to_pixel(u, v)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    elev = 0
    azim = 0

    if fig is None or axes is None:
        fig, axes = plt_utils.divide_figure_to_grid(
            np.arange(10).reshape(1, 10),
            wspace=wspace,
            as_matrix=True,
            figsize=figsize,
        )

    crange = np.abs(rf).max()

    for i, t in enumerate(t_steps):
        mask = np.where(np.abs(time - t) <= 1e-15, True, False)
        _rf = rf[mask]
        plots.quick_hex_scatter(
            _rf,
            cmap=plt.cm.coolwarm,
            edgecolor=None,
            vmin=-crange,
            vmax=crange,
            midpoint=0,
            cbar=False,
            max_extent=max_extent,
            fig=fig,
            ax=axes[0, i],
            fill=True,
            fontsize=fontsize,
        )

        if hlines:
            axes[0, i].hlines(elev, xmin, xmax, color="grey", linewidth=0.25)
        if vlines:
            axes[0, i].vlines(azim, ymin, ymax, color="grey", linewidth=0.25)

    if time_axis:
        left = fig.transFigure.inverted().transform(
            axes[0, 0].transData.transform((0, 0))
        )[0]
        right = fig.transFigure.inverted().transform(
            axes[0, -1].transData.transform((0, 0))
        )[0]

        lefts, bottoms, rights, tops = np.array([
            ax.get_position().extents for ax in axes.flatten()
        ]).T
        time_axis = fig.add_axes((
            left,
            bottoms.min() + y_offset_time_axis * bottoms.min(),
            right - left,
            0.01,
        ))
        plt_utils.rm_spines(
            time_axis,
            ("left", "top", "right"),
            rm_yticks=True,
            rm_xticks=False,
        )

        data_centers_in_points = np.array([
            ax.transData.transform((0, 0)) for ax in axes.flatten()
        ])
        time_axis.tick_params(axis="both", labelsize=fontsize)
        ticks = time_axis.transData.inverted().transform(data_centers_in_points)[:, 0]
        time_axis.set_xticks(ticks)
        time_axis.set_xticklabels(np.arange(0, 200, 20))
        time_axis.set_xlabel("time (ms)", fontsize=fontsize, labelpad=2)
        plt_utils.set_spine_tick_params(
            time_axis,
            spinewidth=0.25,
            tickwidth=0.25,
            ticklength=3,
            ticklabelpad=2,
            spines=("top", "right", "bottom", "left"),
        )

    return fig, axes
