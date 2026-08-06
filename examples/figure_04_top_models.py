"""Figure 4a,b for the best models of an ensemble instead of the task-optimal cluster.

Reproduces the arrangement of figure 4a (T4/T5 angular tuning) and figure 4b (T4c
currents) of Lappalainen et al. (2024), but selects models by task error --- the
best `--top-fraction` of the ensemble --- rather than by the task-optimal cluster
of the naturalistic-stimuli clustering. Figure 4a shows both the mean and the
median across the selected models.

Usage:
    python examples/figure_04_top_models.py
    python examples/figure_04_top_models.py --top-fraction 0.2 --output-dir figures

Requires the pretrained models (`flyvis download-pretrained`). Response
normalization constants are loaded from the constants shipped with flyvis, so no
naturalistic stimuli have to be simulated.
"""

import argparse
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from flyvis import EnsembleView
from flyvis.analysis.moving_bar_responses import (
    angular_tuning,
    peak_responses,
    plot_angular_tuning,
)
from flyvis.analysis.moving_edge_currents import CellTypeArray, MovingEdgeCurrentView
from flyvis.analysis.visualization import plt_utils
from flyvis.datasets.moving_bar import MovingEdge
from flyvis.utils.color_utils import OFF, ON

logging.basicConfig(format="[%(asctime)s] %(message)s", level=logging.INFO)
logging = logging.getLogger(__name__)

T4_TYPES = ["T4a", "T4b", "T4c", "T4d"]
T5_TYPES = ["T5a", "T5b", "T5c", "T5d"]
CURRENT_TYPE = "T4c"
SINGLE_MODEL_COLOR = "#a0a0a0"


def best_models(ensemble: EnsembleView, top_fraction: float) -> np.ndarray:
    """Indices of the models with the lowest task error.

    Args:
        ensemble: The ensemble to select from.
        top_fraction: Fraction of the ensemble to select, e.g. 0.2 for the best 20%.

    Returns:
        Indices into the ensemble, ordered from best to worst task error.
    """
    n_models = max(1, int(round(top_fraction * len(ensemble))))
    return ensemble.argsort()[:n_models]


def plot_tuning_row(peaks, cell_types, intensity, axes, color, model_ids=None):
    """Plot the angular tuning of one row of cell types into polar axes.

    Without `model_ids` this is the arrangement of the paper: the mean fills the
    axis, the median is dashed, and the experimental tuning curve is black.

    With `model_ids` the single models are drawn as well, and then every model
    curve of a panel --- singles, mean and median --- is divided by the *same*
    constant, so that the drawn mean really is the average of the drawn singles.
    Normalizing each single to its own peak instead would hide that models differ
    in amplitude by up to a factor of 30 and would make the mean look much less
    directional than it is.

    Args:
        peaks: Peak responses of the selected models.
        cell_types: Cell types, one per axis.
        intensity: 1 for ON edges, 0 for OFF edges.
        axes: Polar axes, one per cell type.
        color: Color of the mean and median traces.
        model_ids: Network ids to draw as single models. None to draw none.
    """
    for ax, cell_type in zip(axes, cell_types):
        shared = dict(dataset=None, cell_type=cell_type, intensity=intensity, ax=ax)
        tuning = dict(peak_responses_da=peaks, cell_type=cell_type, intensity=intensity)

        if model_ids is not None:
            # common scale for all model curves, set by the strongest single model
            # so that nothing is clipped and amplitudes stay comparable
            scale = float(angular_tuning(**tuning).max())
            shared.update(normalize_by=scale, ymin=0, ymax=1.0)
            for model_id in model_ids:
                plot_angular_tuning(
                    peak_responses_da=peaks.sel(network_id=[model_id]),
                    groundtruth=False,
                    colors=SINGLE_MODEL_COLOR,
                    linewidth=0.3,
                    zorder=10,
                    **shared,
                )

        # median across models
        plot_angular_tuning(
            peak_responses_da=peaks,
            average_models=True,
            model_reduction="median",
            groundtruth=False,
            colors=color,
            linestyle="--",
            linewidth=1.0,
            zorder=110,
            **shared,
        )
        # mean across models, with the experimental tuning curve in black
        plot_angular_tuning(
            peak_responses_da=peaks,
            average_models=True,
            model_reduction="mean",
            groundtruth=model_ids is None,
            groundtruth_linewidth=1.0,
            colors=color,
            linewidth=1.2,
            zorder=100,
            **shared,
        )
        ax.set_xlabel(cell_type, fontsize=6, labelpad=1)


def tuning_grid(peaks, model_ids, subtitle):
    """Create the 4x2 polar grid of T4a-d and T5a-d used by both tuning figures.

    Args:
        peaks: Peak responses of the selected models.
        model_ids: Network ids to draw as single models, or None to draw none.
        subtitle: Second line of the figure title.

    Returns:
        The figure.
    """
    fig, axes, _ = plt_utils.get_axis_grid(
        range(8),
        projection="polar",
        gridwidth=4,
        gridheight=2,
        figsize=[2.95, 1.75],
        wspace=0.25,
        hspace=0.5,
        as_matrix=True,
    )
    plot_tuning_row(peaks, T4_TYPES, 1, axes[0], ON, model_ids)
    plot_tuning_row(peaks, T5_TYPES, 0, axes[1], OFF, model_ids)

    handles = [Line2D([], [], color="0.35", linewidth=1.2, label="model mean")]
    handles.append(
        Line2D([], [], color="0.35", linewidth=1.0, linestyle="--", label="median")
    )
    if model_ids is None:
        handles.insert(0, Line2D([], [], color="k", linewidth=1.0, label="experiment"))
    else:
        handles.append(
            Line2D([], [], color=SINGLE_MODEL_COLOR, linewidth=0.6, label="single models")
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.12),
    )
    fig.suptitle(subtitle, fontsize=6, y=1.03)
    return fig


def figure_04a(peaks, n_models, output_dir):
    """Angular tuning of T4a-d and T5a-d, arranged and normalized as in the paper.

    Args:
        peaks: Peak responses of the selected models.
        n_models: Number of selected models, for the caption.
        output_dir: Directory to write the figure to.

    Returns:
        The figure.
    """
    fig = tuning_grid(
        peaks,
        model_ids=None,
        subtitle=(
            f"angular tuning, mean and median of the {n_models} models\n"
            "with the lowest task error"
        ),
    )
    save(fig, output_dir / f"figure_04a_angular_tuning_best_{n_models}_models")
    return fig


def figure_04a_spread(peaks, model_ids, n_models, output_dir):
    """The same tuning with all single models on a common per-panel scale.

    Args:
        peaks: Peak responses of the selected models.
        model_ids: Network ids of the selected models.
        n_models: Number of selected models, for the caption.
        output_dir: Directory to write the figure to.

    Returns:
        The figure.
    """
    fig = tuning_grid(
        peaks,
        model_ids=model_ids,
        subtitle=(
            f"spread across the {n_models} models\n"
            "(all curves of a panel on one scale, set by the strongest model)"
        ),
    )
    save(fig, output_dir / f"figure_04a_angular_tuning_spread_best_{n_models}_models")
    return fig


def figure_04b(ensemble, model_ids, n_models, output_dir):
    """Input currents of T4c to a moving edge, arranged as in the paper.

    Args:
        ensemble: The full ensemble.
        model_ids: Network ids of the selected models.
        n_models: Number of selected models, for file names.
        output_dir: Directory to write the figures to.

    Returns:
        The three figures: response, temporal contributions, spatial contributions.
    """
    experiment_data = ensemble.moving_edge_currents()
    dataset = MovingEdge(**experiment_data[0].config)
    norm = CellTypeArray(ensemble.responses_norm(rectified=False), ensemble[0].connectome)

    view = MovingEdgeCurrentView(
        ensemble, CURRENT_TYPE, experiment_data, dataset.arg_df
    ).divide_by_given_norm(norm)
    view = view.model_selection(model_ids)

    fig_response, ax = view.plot_response(1, 90, t_end=1.0)
    ax.set_ylabel("voltage (a.u.)", fontsize=5)
    ax.set_xlabel("time (s)", fontsize=5)
    save(fig_response, output_dir / f"figure_04b_{CURRENT_TYPE}_response_best_{n_models}")

    fig_temporal, ax, legend_fig, _ = view.plot_temporal_contributions(
        1,
        90,
        t_start=0,
        t_end=1,
        model_average=True,
        legend=False,
        sum_exc_inh=False,
        only_sum=False,
        max_figure_height_cm=3.3941,
        panel_height_cm=3.3941,
        max_figure_width_cm=4.0513,
        panel_width_cm=4.0513,
        hide_source_types=None,
    )
    ax.set_ylabel("current (a.u.)", fontsize=5)
    save(
        fig_temporal,
        output_dir / f"figure_04b_{CURRENT_TYPE}_temporal_currents_best_{n_models}",
    )
    if legend_fig is not None:
        save(
            legend_fig,
            output_dir / f"figure_04b_{CURRENT_TYPE}_temporal_currents_legend",
        )

    fig_spatial, _, _ = view.plot_spatial_contribution_grid(t_start=0, t_end=1)
    save(
        fig_spatial,
        output_dir / f"figure_04b_{CURRENT_TYPE}_spatial_currents_best_{n_models}",
    )
    return fig_response, fig_temporal, fig_spatial


def save(fig, path: Path) -> None:
    """Write a figure as PDF and PNG.

    Args:
        fig: The figure to write.
        path: Target path without suffix.
    """
    for suffix in (".pdf", ".png"):
        fig.savefig(path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    logging.info("Wrote %s.pdf", path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ensemble", type=str, default="flow/0000")
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.2,
        help="Fraction of the ensemble with the lowest task error to plot.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument(
        "--panels",
        nargs="+",
        default=["a", "a-spread", "b"],
        choices=["a", "a-spread", "b"],
        help=(
            "'a' is the paper arrangement, 'a-spread' the same tuning with all "
            "single models on a common per-panel scale, 'b' the T4c currents."
        ),
    )
    args = parser.parse_args()

    mpl.rcParams["pdf.fonttype"] = 42
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ensemble = EnsembleView(
        args.ensemble,
        best_checkpoint_fn_kwargs={
            "validation_subdir": "validation",
            "loss_file_name": "loss",
        },
    )
    model_ids = best_models(ensemble, args.top_fraction)
    n_models = len(model_ids)
    task_error = ensemble.min_validation_losses()
    logging.info(
        "Best %d of %d models by task error: %s",
        n_models,
        len(ensemble),
        ", ".join(f"{ensemble.names[i]} ({task_error[i]:.3f})" for i in model_ids),
    )

    if {"a", "a-spread"} & set(args.panels):
        stims_and_resps = ensemble.moving_edge_responses()
        # per model and cell type, so that amplitudes are comparable across models
        stims_and_resps["responses"] /= ensemble.responses_norm(rectified=True)
        peaks = peak_responses(stims_and_resps.sel(network_id=model_ids))
        if "a" in args.panels:
            figure_04a(peaks, n_models, args.output_dir)
        if "a-spread" in args.panels:
            figure_04a_spread(peaks, model_ids, n_models, args.output_dir)

    if "b" in args.panels:
        figure_04b(ensemble, model_ids, n_models, args.output_dir)

    plt.close("all")


if __name__ == "__main__":
    main()
