import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from flyvision.plots import plots, plt_utils


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
