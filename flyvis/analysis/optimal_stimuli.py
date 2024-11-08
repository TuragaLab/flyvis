from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

import flyvis
from flyvis.analysis.visualization import plots, plt_utils
from flyvis.datasets.datasets import StimulusDataset
from flyvis.datasets.sintel import AugmentedSintel
from flyvis.network.stimulus import Stimulus
from flyvis.utils import hex_utils, tensor_utils
from flyvis.utils.activity_utils import LayerActivity

__all__ = ["FindOptimalStimuli", "GenerateOptimalStimuli", "plot_stim_response"]


class FindOptimalStimuli:
    """Methods to derive optimal stimuli for cells from stimuli dataset.

    Args:
        network_view: Network view.
        stimuli: Stimuli dataset. "default" uses AugmentedSintelLum.

    Attributes:
        nv (flyvis.NetworkView): Network view.
        network (flyvis.Network): Initialized network.
        central_cells_index (list): Central cells index.
        stimuli (StimulusDataset): Stimulus dataset.
    """

    def __init__(
        self,
        network_view: flyvis.NetworkView,
        stimuli: StimulusDataset | str = "default",
    ):
        self.nv = network_view
        self.network = network_view.init_network()  # type: flyvis.Network
        for param in self.network.parameters():
            param.requires_grad = False
        self.central_cells_index = self.network.connectome.central_cells_index[:]
        self.stimuli = (
            AugmentedSintel(tasks=["lum"], dt=1 / 100, temporal_split=True)
            if stimuli == "default"
            else stimuli
        )

    def optimal_stimuli(
        self,
        cell_type: str,
        dt: float = 1 / 100,
        indices: list[int] | None = None,
    ) -> OptimalStimulus:
        """Finds optimal stimuli for a given cell type in stimuli dataset.

        Args:
            cell_type: Node type.
            dt: Time step.
            indices: Indices of stimuli.

        Returns:
            OptimalStimulus object containing the stimulus and response.
        """
        responses = self.nv.naturalistic_stimuli_responses()
        cell_responses = responses['responses'].custom.where(cell_type=cell_type)

        argmax = cell_responses.argmax(dim=("sample", "frame"))['sample'].item()
        if indices is not None:
            argmax = indices[argmax]
        nat_opt_stim = self.stimuli[argmax]["lum"]

        n_frames = nat_opt_stim.shape[0]
        initial_state = self.network.steady_state(1.0, dt, 1)
        stimulus = Stimulus(self.network.connectome, 1, n_frames)
        stimulus.zero()
        stimulus.add_input(nat_opt_stim[None])
        response = self.network(stimulus(), dt, state=initial_state).detach().cpu()
        response = LayerActivity(response, self.network.connectome, keepref=True)[
            cell_type
        ]

        return OptimalStimulus(nat_opt_stim[None, :], response[:, :, None])

    def regularized_optimal_stimuli(
        self,
        cell_type: str,
        l2_act: float = 1,
        lr: float = 1e-2,
        l2_stim: float = 1,
        n_iters: int = 100,
        dt: float = 1 / 100,
        indices: list[int] | None = None,
    ) -> RegularizedOptimalStimulus:
        """Regularizes the optimal stimulus for a given cell type.

        Maintains central node activity while minimizing mean square of input pixels.

        Args:
            cell_type: Node type.
            l2_act: L2 regularization strength for the activity.
            lr: Learning rate.
            l2_stim: L2 regularization strength for the stimulus.
            n_iters: Number of iterations.
            dt: Time step.
            indices: Indices of stimuli.

        Returns:
            RegularizedOptimalStimulus object.
        """

        optim_stimuli = self.optimal_stimuli(
            cell_type=cell_type,
            dt=dt,
            indices=indices,
        )
        non_nan = ~torch.isnan(
            optim_stimuli.stimulus[0, :, 0, optim_stimuli.stimulus.shape[-1] // 2]
        )
        reg_opt_stim = optim_stimuli.stimulus.clone()
        reg_opt_stim = reg_opt_stim[:, non_nan]
        reg_opt_stim.requires_grad = True

        central_target_response = (
            optim_stimuli.response.to(non_nan.device)[
                :, non_nan, :, optim_stimuli.response.shape[-1] // 2
            ]
            .clone()
            .detach()
            .squeeze()
        )

        optim = torch.optim.Adam([reg_opt_stim], lr=lr)

        n_frames = reg_opt_stim.shape[1]

        stim = Stimulus(self.network.connectome, 1, n_frames)

        layer_activity = LayerActivity(None, self.network.connectome, keepref=True)

        initial_state = self.network.steady_state(1.0, dt, 1)

        losses = []
        for _ in range(n_iters):
            optim.zero_grad()
            stim.zero()
            stim.add_input(reg_opt_stim)
            activities = self.network(stim(), dt, state=initial_state)
            layer_activity.update(activities)
            central_predicted_response = layer_activity.central[cell_type].squeeze()

            act_loss = (
                l2_act
                * ((central_predicted_response - central_target_response) ** 2).sum()
            )
            stim_loss = l2_stim * ((reg_opt_stim - 0.5) ** 2).mean(dim=0).sum()
            loss = act_loss + stim_loss
            loss.backward(retain_graph=True)
            optim.step()
            losses.append(loss.detach().cpu().numpy().item())

        stim.zero()
        reg_opt_stim.requires_grad = False
        stim.add_input(reg_opt_stim)
        activities = self.network(stim(), dt, state=initial_state)
        layer_activity.update(activities)

        reg_opt_stim = reg_opt_stim.detach().cpu()
        rnmei_response = layer_activity[cell_type].detach().cpu()
        central_predicted_response = central_predicted_response.detach().cpu()
        central_target_response = central_target_response.detach().cpu()
        return RegularizedOptimalStimulus(
            optim_stimuli,
            reg_opt_stim,
            rnmei_response,
            central_predicted_response,
            central_target_response,
            losses,
        )


class GenerateOptimalStimuli:
    """Methods to generate optimal stimuli for cells from random noise."""

    def __init__(self, network_view: flyvis.NetworkView):
        self.network = network_view.init_network()  # type: flyvis.Network

        for param in self.network.parameters():
            param.requires_grad = False

    def artificial_optimal_stimuli(
        self,
        cell_type: str,
        t_stim: float = 49 / 200,
        dt: float = 1 / 100,
        lr: float = 1e-2,
        weight_central: float = 1.0,
        weight_mei: float = 600 * 5,
        n_iters: int = 200,
        random_seed: int = 0,
        last_only: bool = True,
    ) -> GeneratedOptimalStimulus:
        """Generate artificial stimuli maximally exciting the central node of a type.

        Args:
            cell_type: Node type.
            t_stim: Stimulus duration.
            dt: Time step.
            lr: Learning rate.
            weight_central: Weight for central node optimization.
            weight_mei: Weight for MEI optimization.
            n_iters: Number of iterations.
            random_seed: Random seed for initialization.
            last_only: If True, optimize only the last frame.

        Returns:
            GeneratedOptimalStimulus object.
        """
        n_frames = int(t_stim / dt)
        n_hexals = hex_utils.get_num_hexals(self.network.config.connectome.extent)

        initial_state = self.network.steady_state(1.0, dt, 1)

        # Prepare pre stimulus.
        stimulus = Stimulus(self.network.connectome, 1, n_frames)

        # Initialize maximally excitatory tensors per time bin.
        torch.manual_seed(random_seed)
        art_opt_stim = torch.rand(1, n_frames, 1, n_hexals, device=flyvis.device)

        art_opt_stim.data.clamp_(0, 1)
        art_opt_stim.requires_grad = True

        # Initialize optimizer for the stimuli.
        optim = torch.optim.Adam([art_opt_stim], lr=lr)

        # Memorize surround_index for the loop.
        central_index = stimulus.central_cells_index[cell_type]

        optimize_frames = [range(n_frames)[-1]] if last_only else list(range(n_frames))

        def optimize(mei):
            # Reset stimulus.
            stimulus.zero()
            stimulus.add_input(mei)
            # Reset optimizer.
            optim.zero_grad()

            central_loss = 0
            # Stimulate network.
            activity = self.network(stimulus(), dt, state=initial_state)
            central_activity = activity[:, :, central_index].squeeze()

            central_loss -= (
                weight_central * torch.exp(central_activity[optimize_frames]).mean()
            )

            mei_loss = weight_mei * ((mei - 0.5) ** 2).mean()
            loss = central_loss + mei_loss
            loss.backward(retain_graph=True)
            optim.step()
            mei.data.clamp_(0, 1)
            return (
                loss.detach().cpu().numpy(),
                central_loss.detach().cpu().numpy(),
                mei_loss.detach().cpu().numpy(),
            )

        losses = []
        for _ in range(n_iters):
            loss = optimize(art_opt_stim)
            losses.append(loss)

        losses = np.array(losses)

        art_opt_stim.requires_grad = False
        # Stimulate network with whole sequence.
        stimulus.zero()
        stimulus.add_input(art_opt_stim)
        activity = self.network(stimulus(), dt, state=initial_state)
        responses = activity.detach().cpu().numpy()[:, :, stimulus.layer_index[cell_type]]

        art_opt_stim = art_opt_stim.cpu().numpy()
        return GeneratedOptimalStimulus(art_opt_stim, responses, losses)


@dataclass
class OptimalStimulus:
    """Optimal stimulus and response."""

    stimulus: np.ndarray
    response: np.ndarray


@dataclass
class RegularizedOptimalStimulus:
    """Regularized optimal stimulus and related data."""

    stimulus: OptimalStimulus
    regularized_stimulus: np.ndarray
    response: np.ndarray
    central_predicted_response: np.ndarray
    central_target_response: np.ndarray
    losses: np.ndarray


@dataclass
class GeneratedOptimalStimulus:
    """Generated optimal stimulus, response, and optimization losses."""

    stimulus: np.ndarray
    response: np.ndarray
    losses: np.ndarray


@dataclass
class StimResponsePlot:
    """Stimulus-response plot data and methods."""

    stim: np.ndarray
    response: np.ndarray
    dt: float
    u: np.ndarray
    v: np.ndarray
    time: np.ndarray
    t_step: np.ndarray
    t_steps_stim: np.ndarray
    t_steps_response: np.ndarray
    xmin_lattice: float
    xmax_lattice: float
    ymin_lattice: float
    ymax_lattice: float
    subtraced_baseline: bool
    steps: int
    fig: Any
    axes: Any
    time_axis: Any
    trace_axis: Any
    argmax: int
    t_argmax: float

    def __iter__(self):
        """Yield figure and axes."""
        yield from [self.fig, self.axes]

    def add_to_trace_axis(
        self,
        other: "StimResponsePlot",
        color: str | None = None,
        label: str | None = None,
        linewidth: float | None = None,
    ):
        """Add another StimResponsePlot's trace to this plot's trace axis."""
        xticks = self.trace_axis.get_xticks()
        mask = (other.time >= other.t_step.min()) & (other.time <= other.t_step.max())
        time = np.linspace(xticks.min(), xticks.max(), mask.sum())
        self.trace_axis.plot(
            time,
            other.response[mask, other.response.shape[-1] // 2],
            color=color,
            label=label,
            linewidth=linewidth,
        )


def plot_stim_response(
    stim: np.ndarray,
    response: np.ndarray,
    dt: float,
    u: np.ndarray,
    v: np.ndarray,
    max_extent: int = 6,
    subtract_baseline: bool = True,
    seconds: float = 0.2,
    steps: int = 10,
    columns: int = 10,
    suptitle: str = "",
    plot_resp: bool = True,
    hlines: bool = True,
    vlines: bool = True,
    time_axis: bool = True,
    peak_central: bool = False,
    wspace: float = -0.2,
    peak_last: bool = True,
    fontsize: int = 5,
    ylabel: str = "",
    ylabelrotation: int = 90,
    figsize: list[float] = [5, 1],
    label_peak_response: bool = False,
    fig: plt.Figure | None = None,
    axes: np.ndarray | None = None,
    crange: float | None = None,
    trace_axis: bool = False,
    trace_label: str | None = None,
    trace_axis_offset: float = 0.1,
    trace_color: str | None = None,
) -> StimResponsePlot:
    """Plot spatio-temporal stimulus and response on regular hex lattices.

    Args:
        stim: Stimulus array.
        response: Response array.
        dt: Time step.
        u: Hexagonal u-coordinates.
        v: Hexagonal v-coordinates.
        max_extent: Maximum extent of the hexagonal grid.
        subtract_baseline: Whether to subtract baseline from response.
        seconds: Duration to plot in seconds.
        steps: Number of time steps to plot.
        columns: Number of columns in the plot.
        suptitle: Super title for the plot.
        plot_resp: Whether to plot response.
        hlines: Whether to plot horizontal lines.
        vlines: Whether to plot vertical lines.
        time_axis: Whether to add a time axis.
        peak_central: Whether to center the plot around the peak.
        wspace: Width space between subplots.
        peak_last: Whether to show the peak in the last frame.
        fontsize: Font size for labels and titles.
        ylabel: Y-axis label.
        ylabelrotation: Rotation angle for y-axis label.
        figsize: Figure size.
        label_peak_response: Whether to label the peak response.
        fig: Existing figure to plot on.
        axes: Existing axes to plot on.
        crange: Color range for the plot.
        trace_axis: Whether to add a trace axis.
        trace_label: Label for the trace.
        trace_axis_offset: Offset for the trace axis.
        trace_color: Color for the trace.

    Returns:
        StimResponsePlot object containing plot data and figure.
    """
    stim = tensor_utils.to_numpy(stim).squeeze()
    mask = ~np.isnan(stim).any(axis=-1).squeeze()
    response = tensor_utils.to_numpy(response).squeeze()
    stim = stim[mask]
    response = response[mask]

    if subtract_baseline:
        response -= response[0]

    argmax = np.nanargmax(response[:, response.shape[-1] // 2])

    n_frames = response.shape[0]
    time = np.arange(n_frames) * dt
    steps = int(seconds / dt)
    t_argmax = time[argmax]

    if peak_central:
        start = argmax - steps // 2
        end = argmax + steps // 2
        if start < 0:
            start = 0
            end = steps
        peak_last = False

    if peak_last:
        start = argmax - steps
        end = argmax
        if start < 0:
            start = 0
            end = steps

    _t_steps = time[start:end]

    # resample in time in case seconds, number of columns, dt does not match
    time_index = np.linspace(0, len(_t_steps), 2 * columns, endpoint=False).astype(int)
    _t_steps = _t_steps[time_index]

    # breakpoint()
    t_steps_stim = _t_steps[0::2]
    t_steps_resp = _t_steps[1::2]

    _u, _v = hex_utils.get_hex_coords(max_extent)
    x, y = hex_utils.hex_to_pixel(_u, _v)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    elev = 0
    azim = 0

    if fig is None or axes is None:
        if plot_resp:
            x, y = hex_utils.hex_rows(2, columns)
            fig, axes, pos = plt_utils.ax_scatter(
                x,
                y,
                figsize=figsize,
                hpad=0,
                wpad=0.07,
                wspace=-0.7,
                hspace=-0.5,
            )
            axes = np.array(axes).reshape(2, columns)

        else:
            fig, axes = plt_utils.divide_figure_to_grid(
                np.arange(10).reshape(1, 10),
                wspace=wspace,
                as_matrix=True,
                figsize=figsize,
            )

    crange = crange or np.abs(np.nanmax(response))
    for i, t in enumerate(t_steps_stim):
        # plot stimulus
        mask = np.where(np.abs(time - t) <= 1e-15, True, False)
        _stim = stim[mask].squeeze()
        plots.quick_hex_scatter(
            _stim,
            vmin=0,
            vmax=1,
            cbar=False,
            max_extent=max_extent,
            fig=fig,
            ax=axes[0, i],
        )

        if hlines:
            axes[0, i].hlines(elev, xmin, xmax, color="#006400", linewidth=0.25)
        if vlines:
            axes[0, i].vlines(azim, ymin, ymax, color="#006400", linewidth=0.25)

        if plot_resp:
            # --- plot response

            mask = np.where(np.abs(time - t_steps_resp[i]) <= 1e-15, True, False)
            _resp = response[mask].squeeze()
            plots.hex_scatter(
                u,
                v,
                _resp,
                fill=True,
                # edgecolor="0.3",
                # edgewidth=0.1,
                cmap=plt.cm.coolwarm,
                vmin=-crange,
                vmax=crange,
                midpoint=0,
                cbar=False,
                max_extent=max_extent,
                fig=fig,
                ax=axes[1, i],
            )
            if t_steps_resp[i] == t_argmax and label_peak_response:
                axes[1, i].set_title("peak", fontsize=fontsize)

            if hlines:
                axes[1, i].hlines(elev, xmin, xmax, color="#006400", linewidth=0.25)
            if vlines:
                axes[1, i].vlines(azim, ymin, ymax, color="#006400", linewidth=0.25)

    if trace_axis:
        left = fig.transFigure.inverted().transform(
            axes[0, 0].transData.transform((0, 0))
        )[0]
        right = fig.transFigure.inverted().transform(
            axes[-1, -1].transData.transform((0, 0))
        )[0]

        lefts, bottoms, rights, tops = np.array([
            ax.get_position().extents for ax in axes.flatten()
        ]).T

        trace_axis = fig.add_axes(
            (
                left,
                bottoms.min() - trace_axis_offset,
                right - left,
                trace_axis_offset - 0.05 * trace_axis_offset,
            ),
            label="trace_axis",
        )
        plt_utils.rm_spines(
            trace_axis, ("top", "right"), rm_yticks=False, rm_xticks=False
        )

        data_centers_in_points = np.array([
            ax.transData.transform((0, 0)) for ax in axes.flatten(order="F")
        ])
        trace_axis.tick_params(axis="both", labelsize=fontsize)
        if plot_resp:
            xticks = trace_axis.transData.inverted().transform(data_centers_in_points)[
                1::2, 0
            ]
            trace_axis.set_xticks(xticks)
            ticklabels = np.round(_t_steps * 1000, 0)
            trace_axis.set_xticklabels((ticklabels - ticklabels.max())[1::2])
        else:
            xticks = trace_axis.transData.inverted().transform(data_centers_in_points)[
                :, 0
            ]
            trace_axis.set_xticks(xticks)
            ticklabels = np.round(t_steps_stim * 1000, 0)
            trace_axis.set_xticklabels((ticklabels - ticklabels.max()))
        trace_axis.set_xlabel("time (ms)", fontsize=fontsize, labelpad=2)
        plt_utils.set_spine_tick_params(
            trace_axis,
            spinewidth=0.25,
            tickwidth=0.25,
            ticklength=3,
            ticklabelpad=2,
            spines=("top", "right", "bottom", "left"),
        )
        xlim = trace_axis.get_xlim()
        mask = (time >= _t_steps.min()) & (time <= _t_steps.max())

        time = np.linspace(xticks.min(), xticks.max(), mask.sum())

        trace_axis.plot(
            time,
            response[mask, response.shape[-1] // 2],
            label=trace_label,
            color=trace_color,
        )
        trace_axis.set_xlim(*xlim)
        trace_axis.set_ylabel("central\nresponse", fontsize=fontsize)
        # flyvis.plots.trim_axis(trace_axis)

        time_axis = False

    if time_axis:
        left = fig.transFigure.inverted().transform(
            axes[0, 0].transData.transform((0, 0))
        )[0]
        right = fig.transFigure.inverted().transform(
            axes[-1, -1].transData.transform((0, 0))
        )[0]

        lefts, bottoms, rights, tops = np.array([
            ax.get_position().extents for ax in axes.flatten()
        ]).T

        time_axis = fig.add_axes((left, bottoms.min(), right - left, 0.01))
        plt_utils.rm_spines(
            time_axis, ("left", "top", "right"), rm_yticks=True, rm_xticks=False
        )

        data_centers_in_points = np.array([
            ax.transData.transform((0, 0)) for ax in axes.flatten(order="F")
        ])
        time_axis.tick_params(axis="both", labelsize=fontsize)
        if plot_resp:
            time_axis.set_xticks(
                time_axis.transData.inverted().transform(data_centers_in_points)[1::2, 0]
            )
            ticklabels = np.round(_t_steps * 1000, 0)
            time_axis.set_xticklabels((ticklabels - ticklabels.max())[1::2])
        else:
            time_axis.set_xticks(
                time_axis.transData.inverted().transform(data_centers_in_points)[:, 0]
            )
            ticklabels = np.round(t_steps_stim * 1000, 0)
            time_axis.set_xticklabels((ticklabels - ticklabels.max()))
        time_axis.set_xlabel("time (ms)", fontsize=fontsize, labelpad=2)
        plt_utils.set_spine_tick_params(
            time_axis,
            spinewidth=0.25,
            tickwidth=0.25,
            ticklength=3,
            ticklabelpad=2,
            spines=("top", "right", "bottom", "left"),
        )

    if ylabel:
        lefts, bottoms, rights, tops = np.array([
            ax.get_position().extents for ax in axes.flatten()
        ]).T
        ylabel_axis = fig.add_axes((
            lefts.min(),
            bottoms.min(),
            0.01,
            tops.max() - bottoms.min(),
        ))
        plt_utils.rm_spines(
            ylabel_axis,
            ("left", "top", "right", "bottom"),
            rm_yticks=True,
            rm_xticks=True,
        )
        ylabel_axis.set_ylabel(ylabel, fontsize=fontsize, rotation=ylabelrotation)
        ylabel_axis.patch.set_alpha(0)

    if plot_resp and ylabel is not None:
        axes[0, 0].annotate(
            "stimulus",
            xy=(0, 0.5),
            ha="right",
            va="center",
            fontsize=fontsize,
            rotation=90,
            xycoords="axes fraction",
        )
        axes[1, 0].annotate(
            "response",
            xy=(0, 0.5),
            ha="right",
            va="center",
            fontsize=fontsize,
            rotation=90,
            xycoords="axes fraction",
        )

    if suptitle:
        lefts, bottoms, rights, tops = np.array([
            ax.get_position().extents for ax in axes.flatten()
        ]).T
        fig.suptitle(suptitle, fontsize=fontsize, y=tops.max(), va="bottom")

    plt_utils.set_spine_tick_params(
        fig.axes[-1],
        spinewidth=0.25,
        tickwidth=0.25,
        ticklength=3,
        ticklabelpad=2,
        spines=("top", "right", "bottom", "left"),
    )

    fig.crange = crange

    return StimResponsePlot(
        stim,
        response,
        dt,
        u,
        v,
        time,
        _t_steps,
        t_steps_stim,
        t_steps_resp,
        xmin,
        xmax,
        ymin,
        ymax,
        subtract_baseline,
        steps,
        fig,
        axes,
        time_axis,
        trace_axis,
        argmax,
        t_argmax,
    )
