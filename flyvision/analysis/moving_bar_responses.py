from dataclasses import dataclass
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datamate import Namespace

from flyvision.analysis.simple_correlation import (
    correlation,
    quick_correlation_one_to_many,
)
from flyvision.datasets.moving_bar import mask_between_seconds, time_window
from flyvision.plots import plt_utils
from flyvision.plots.plots import polar, violin_groups
from flyvision.utils import groundtruth_utils, nodes_edges_utils
from flyvision.utils.activity_utils import StimulusResponseIndexer
from flyvision.utils.color_utils import OFF, ON, adapt_color_alpha
from flyvision.utils.nodes_edges_utils import CellTypeArray
from flyvision.utils.tensor_utils import select_along_axes


class MovingBarResponseView(StimulusResponseIndexer):
    def __init__(
        self,
        arg_df: pd.DataFrame,
        config: Namespace,
        responses: CellTypeArray,
        stim_sample_dim=0,
        temporal_dim=1,
        time=None,
    ):
        self.config = config
        if time is None:
            if len(config.offsets) == 2:
                n_offsets = config.offsets[1] - config.offsets[0]
            else:
                n_offsets = len(config.offsets)
            t_stim = (n_offsets * np.radians(2.25)) / (
                np.array(config.speeds) * np.radians(5.8)
            )
            time = np.arange(
                -config.t_pre, np.max(t_stim) + config.t_post - config.dt, config.dt
            )
        super().__init__(
            arg_df=arg_df,  # could also construct from config
            responses=responses,
            dt=config.dt,
            t_pre=config.t_pre,
            stim_sample_dim=stim_sample_dim,
            temporal_dim=temporal_dim,
            time=time,
        )

    def view(
        self,
        arg_df: pd.DataFrame = None,
        config: Namespace = None,
        responses: Union[np.ndarray, CellTypeArray] = None,
        stim_sample_dim=None,
        temporal_dim=None,
        time=None,
    ) -> "MovingBarResponseView":
        if isinstance(responses, (np.ndarray, np.ma.MaskedArray)):
            responses = CellTypeArray(responses, cell_types=self.responses.cell_types)

        return self.__class__(
            arg_df if np.any(arg_df) else self.arg_df,
            config if config is not None else self.config,
            responses if responses is not None else self.responses,
            stim_sample_dim if np.any(stim_sample_dim) else self.stim_sample_dim,
            temporal_dim or self.temporal_dim,
            time if np.any(time) else self.time,
        )

    def angular(self) -> "MovingBarResponseView":
        angles = np.radians(self.arg_df.angle.to_numpy())
        angles = angles.reshape(*[
            (-1 if i == self.stim_sample_dim else 1) for i in range(len(self.shape))
        ])
        complex_responses = self.responses[:] * np.exp(angles * 1j)
        return self.view(
            responses=complex_responses, stim_sample_dim=self.stim_sample_dim
        )

    def peak_responses(
        self,
        norm=None,
        from_degree=None,
        to_degree=None,
    ) -> "MovingBarResponseView":
        """Peak responses from rectified voltages that are normalized per cell
        if a norm is provided. The normalization does not change the
        scale-invariant
        direction selectivity index (DSI) of the responses, but if responses
        are averaged it will make the average more represetnative of the
        population.
        """
        from_degree = from_degree or self.config.offsets[0] * 2.25
        to_degree = to_degree or (self.config.offsets[1] - 1) * 2.25
        masks = self.get_time_masks(
            from_degree / 5.8,
            to_degree / 5.8,
        )
        # original way from the paper
        view = self.masked(
            masks, mask_dims=(self.stim_sample_dim, self.temporal_dim)
        ).rectify()

        if norm is not None:
            view = view.divide_by_given_array(norm, dims=(0, -1))

        peak = view.peak()
        return peak

    def tuning_curves(
        self,
        norm=None,
        from_degree=None,
        to_degree=None,
    ):
        peak_responses = self.peak_responses(
            norm=norm, from_degree=from_degree, to_degree=to_degree
        )
        stim_tunings = [
            stim_peaks[:]
            for _, stim_peaks in peak_responses.groupby(["angle", "intensity"])
        ]
        stim_tunings = np.stack(stim_tunings, axis=0)
        stim_tunings = stim_tunings.reshape(
            len(self.config.angles),
            len(self.config.intensities),
            *stim_tunings.shape[1:],
        )  # angles, intensities, ..., stim_sample_dim, temporal_dim, ..., cell_type
        tuning_curve = peak_responses.view(
            responses=stim_tunings,
            stim_sample_dim=peak_responses.stim_sample_dim + 2,
            temporal_dim=peak_responses.temporal_dim + 2,
            arg_df=peak_responses.arg_df.loc[
                peak_responses.arg_df[["width", "speed"]].drop_duplicates().index
            ],
        )
        angles = sorted(peak_responses.arg_df.angle.unique())
        intensities = sorted(peak_responses.arg_df.intensity.unique())
        return TuningCurveData(tuning_curve, angles, intensities)

    def get_time_masks(self, from_column=-1.5, to_column=1.5):
        masks = {}
        for speed in self.arg_df.speed.unique():
            masks[speed] = mask_between_seconds(
                *time_window(
                    speed,
                    from_column=from_column,
                    to_column=to_column,
                    start=self.config.offsets[0],
                    end=self.config.offsets[1],
                ),
                self.time,
            )
        return np.array([masks[speed] for speed in self.arg_df.speed.to_list()])

    def dsi(self, average=True):
        Zpeak = self.peak_responses().angular()
        vector_sum = Zpeak.groupby(by=["width", "intensity", "speed"]).sum()
        vector_sum = vector_sum.sum(dims=vector_sum.temporal_dim)
        vector_length = vector_sum.abs()

        normalization = Zpeak.abs().groupby(by=["width", "intensity", "speed"]).sum()
        normalization = normalization.sum(dims=normalization.temporal_dim)
        normalization = normalization.groupby(by=["width", "speed"]).max()
        dsi = vector_length.groupby(by=["intensity"]).apply(
            lambda gpby: gpby / normalization
        )
        if average:
            dsi = dsi.groupby(by=["intensity"]).mean()
        return dsi

    def preferred_direction(self, average=True):
        Zpeak = self.peak_responses().angular()
        vector_sum = Zpeak.groupby(by=["width", "intensity", "speed"]).sum()
        if average:
            # average over widths and speeds
            theta_pref = np.angle(vector_sum.groupby(by=["intensity"]).sum()[:])
        else:
            theta_pref = np.angle(vector_sum[:])
        return np.ma.filled(theta_pref, np.nan)

    def plot_traces(
        self, cell_type, t_start=None, t_end=None, plot_kwargs=dict(), **stim_kwargs
    ):
        if "title" not in plot_kwargs:
            plot_kwargs["title"] = f"{cell_type} moving-bar response"
        return super().plot_traces(
            cell_type=cell_type,
            t_start=t_start,
            t_end=t_end,
            plot_kwargs=plot_kwargs,
            **stim_kwargs,
        )

    def plot_angular_tuning(
        self,
        cell_type,
        intensity,
        quantile=None,
        figsize=[1, 1],
        fontsize=5,
        linewidth=1,
        anglepad=-7,
        xlabelpad=-1,
        groundtruth=False,
        groundtruth_linewidth=1.0,
        fig=None,
        ax=None,
        compare_across_contrasts=False,
        weighted_average=None,
        average_models=True,
        model_dim=None,
        colors=None,
        zorder=10,
        tuning_curves=None,
        **kwargs,
    ):
        """
        Args:
            tuning (optional): If provided, use this tuning instead of computing it.
                Must be a MovingBarResponseView of shape
                (12, 50, 3, 2, 6, 1, 65).
        """
        if tuning_curves is None:
            tuning_curves = self.tuning_curves()
        tuning_curves, angles, intensities = (
            tuning_curves.tuning_curve,
            tuning_curves.angles,
            tuning_curves.intensities,
        )
        assert intensity in intensities
        angles = np.array(angles)

        # squeeze out the cell type dimension
        cell_tuning = tuning_curves[cell_type]
        # average over speeds and widths
        average_tuning = cell_tuning.mean(dims=(cell_tuning.stim_sample_dim,))
        # average over models
        if model_dim is not None and average_models:
            average_tuning = average_tuning.average(
                dims=(model_dim,), weights=weighted_average
            )

        if quantile:
            # average over speeds, widths and compute the quantile over models
            quantile = cell_tuning.mean(dims=(cell_tuning.stim_sample_dim,)).quantile(
                quantile, dims=(1,)
            )[:]

        # being verbose here to avoid misunderstandings cause intensity and their
        # index are identical
        intensity_index = list(intensities).index(intensity)
        r_predicted = average_tuning.take_single(indices=intensity_index, dims=1)

        if compare_across_contrasts:
            r_predicted = r_predicted[:].squeeze() / (
                rabsmax := average_tuning.abs()
                .max(
                    dims=(
                        0,
                        -1,
                    )
                )[:]
                .squeeze()
            )

        else:
            r_predicted = r_predicted[:].squeeze() / (
                rabsmax := r_predicted.abs()
                .max(
                    dims=(
                        0,
                        -1,
                    )
                )[:]
                .squeeze()
            )

        color = (ON if intensity == 1 else OFF) if colors is None else colors

        fig, ax = polar(
            angles,
            r_predicted,
            figsize=figsize,
            fontsize=fontsize,
            linewidth=linewidth,
            anglepad=anglepad,
            xlabelpad=xlabelpad,
            color=color,
            zorder=zorder,
            fig=fig,
            ax=ax,
            **kwargs,
        )

        if np.any(quantile):
            quantile = np.take(quantile, intensity, axis=-1)
            quantile = quantile / rabsmax
            closed = angles[-1] % 360 == angles[0]
            if not closed:
                angles = np.array([*angles, angles[0]])
                quantile = np.append(quantile, quantile[:, 0][:, None], axis=1)
            ax.fill_between(
                np.radians(angles),
                quantile[0],
                quantile[1],
                facecolor=adapt_color_alpha(color, 0.1),
                edgecolor=color,
                linewidth=0.25,
                zorder=0,
            )

        if groundtruth and cell_type in groundtruth_utils.tuning_curves:
            r = np.array(groundtruth_utils.tuning_curves[cell_type])
            r = r / np.max(np.abs(r))
            theta = np.arange(0, 360, 360 / len(r))
            polar(
                theta,
                r,
                figsize=figsize,
                fontsize=fontsize,
                linewidth=groundtruth_linewidth,
                anglepad=anglepad,
                xlabelpad=xlabelpad,
                color="k",
                zorder=100,
                fig=fig,
                ax=ax,
                **kwargs,
            )
        return fig, ax

    def plot_t4_tuning(self, tuning):
        fig, axes, _ = plt_utils.get_axis_grid(
            range(4),
            projection="polar",
            aspect_ratio=4,
            figsize=[2.95, 0.83],
            wspace=0.25,
        )
        for i, cell_type in enumerate(["T4a", "T4b", "T4c", "T4d"]):
            self.plot_angular_tuning(
                cell_type,
                intensity=1,
                fig=fig,
                ax=axes[i],
                groundtruth=True,
                aggregate_models="mean",
                linewidth=1.0,
                scale_to_max_over_all_stimuli=False,
                tuning=tuning,
            )
            axes[i].set_xlabel(cell_type)

        for ax in axes:
            ax.xaxis.label.set_fontsize(8)
            [i.set_linewidth(0.5) for i in ax.spines.values()]
            ax.grid(True, linewidth=0.5)

    def plot_t5_tuning(self, tuning):
        fig, axes, _ = plt_utils.get_axis_grid(
            range(4),
            projection="polar",
            aspect_ratio=4,
            figsize=[2.95, 0.83],
            wspace=0.25,
        )
        for i, cell_type in enumerate(["T5a", "T5b", "T5c", "T5d"]):
            self.plot_angular_tuning(
                cell_type,
                intensity=0,
                fig=fig,
                ax=axes[i],
                groundtruth=True,
                aggregate_models="mean",
                linewidth=1.0,
                scale_to_max_over_all_stimuli=False,
                tuning=tuning,
            )
            axes[i].set_xlabel(cell_type)

        for ax in axes:
            ax.xaxis.label.set_fontsize(8)
            [i.set_linewidth(0.5) for i in ax.spines.values()]
            ax.grid(True, linewidth=0.5)


class MovingEdgeResponseView(MovingBarResponseView):
    def plot_traces(
        self, cell_type, t_start=None, t_end=None, plot_kwargs=dict(), **stim_kwargs
    ):
        if "title" not in plot_kwargs:
            plot_kwargs["title"] = f"{cell_type} moving-edge response"
        return super().plot_traces(
            cell_type=cell_type,
            t_start=t_start,
            t_end=t_end,
            plot_kwargs=plot_kwargs,
            **stim_kwargs,
        )


# -- correlation ------------


def dsi_correlation_to_known(
    dsis,  # TODO: maybe make into CellTypeArray
    cell_types,
    respect_contrast=False,
    aggregate="mean",
    agg_dim=None,
):
    """
    Computes the correlation between predicted DSIs and known DSIs.

    Args:
        dsis (np.ndarray): Array of shape (2, n_cell_types) containing the DSIs for
            ON and OFF intensities.
        cell_types (list): List of cell types.
        respect_contrast (bool): If True, respect T4 and T5 contrast preference, i.e.,
            only compute DSI for preferred contrast.
        aggregate (str): Aggregation method for the correlation values. Can be
            "mean" or "median".
        agg_dim (int): Dimension along which to aggregate the correlation values.

    Note: known DSIs are binary, either 0 or 1, depending on whether the cell type
    is known to be motion-tuned or not.
    """
    assert dsis.shape[0] == 2, (
        "First dimension of `dsis` array should correspond to stimulus intensity (on/"
        "off) and thus have length 2"
    )
    assert dsis.shape[-1] == len(
        cell_types
    ), "Last dimension of `dsis` array should correspond to cell type"
    motion_tuning = groundtruth_utils.motion_tuning
    cell_types_list = list(cell_types)

    if not respect_contrast:
        # note: includes T4 and T5 major inputs
        known_dsi_types = groundtruth_utils.known_dsi_types

        known_idxs = [cell_types_list.index(cell_type) for cell_type in known_dsi_types]
        dsis_for_known = dsis[..., known_idxs]

        if agg_dim is not None:
            if aggregate == "max":
                dsis_for_known = np.max(dsis_for_known, axis=agg_dim)
            elif aggregate == "median":
                dsis_for_known = np.median(dsis_for_known, axis=agg_dim)
            else:
                raise ValueError

        groundtruth_mt = np.array([
            1.0 if nt in motion_tuning else 0.0 for nt in known_dsi_types
        ])

        shape = dsis_for_known.shape
        dsis_for_known = dsis_for_known.reshape(
            2, -1, len(groundtruth_mt)
        )  # flatten all model dims
        dsis_for_known = dsis_for_known.swapaxes(0, 1).reshape(-1, len(groundtruth_mt))

        corr_dsi, _ = correlation(groundtruth_mt, dsis_for_known)

        corr_dsi = corr_dsi.reshape(-1, 2).T.reshape(*shape[:-1])
    else:
        on_motion_tuning = groundtruth_utils.on_motion_tuning
        off_motion_tuning = groundtruth_utils.off_motion_tuning
        no_motion_tuning = groundtruth_utils.no_motion_tuning

        on_idxs = [cell_types_list.index(cell_type) for cell_type in on_motion_tuning]
        off_idxs = [cell_types_list.index(cell_type) for cell_type in off_motion_tuning]
        no_idxs = [cell_types_list.index(cell_type) for cell_type in no_motion_tuning]
        # pick ON
        on_motion_dsi = dsis[..., on_idxs][1]
        # pick OFF
        off_motion_dsi = dsis[..., off_idxs][0]
        # aggregate over ON and OFF
        no_motion_dsi = np.median(dsis[..., no_idxs], axis=(0,))

        groundtruth_mt = [
            1.0 if cell_type in motion_tuning else 0.0
            for cell_type in [
                *on_motion_tuning,
                *off_motion_tuning,
                *no_motion_tuning,
            ]
        ]

        dsis_for_known = np.concatenate(
            (on_motion_dsi, off_motion_dsi, no_motion_dsi), axis=-1
        )

        shape = dsis_for_known.shape
        dsis_for_known = dsis_for_known.reshape(
            -1, len(groundtruth_mt)
        )  # flatten all model dims

        corr_dsi, _ = correlation(groundtruth_mt, dsis_for_known)

        if len(shape) > 3:
            corr_dsi = corr_dsi.T.reshape(*shape[1:-1])

        if aggregate == "max":
            corr_dsi = np.max(corr_dsi, axis=agg_dim)
        elif aggregate == "median":
            corr_dsi = np.median(corr_dsi, axis=agg_dim)
    return corr_dsi


def tuning_curve_correlation_to_known(
    tuning: MovingBarResponseView,
    # TODO: maybe make specific tuningresponseview subclass for safety
    angles=None,
    intensities=None,
    mode="independent",
    aggregate="absmax",
    aggregate_dims=None,
    select_stimuli_indices=None,
    respect_contrast=True,
    concatenate=False,
    opposite_contrast=False,
    interpolate=False,
    fill_missing=0.0,
):
    """
    Computes the correlation between predicted tuning curves and known tuning curves.

    Args:
        tuning: Tuning responses.
        angles: Angles for which to compute the correlation.
        intensities: Intensities for which to compute the correlation.
        mode: independent, joint -- either compute correlations for each cell type
            independently or jointly
        aggregate: median, mean, max -- aggregate over stim params
        aggregate_dims: dimension along which to aggregate the correlation values
        select_stimuli_indices: can select specific stimuli indices to compute
            correlations for.
        respect_contrast: if True, respect to compute correlations for preferred
            contrast tunings instead of all contrasts
        concatenate: if True, concatenate all cell types into one array
        opposite_contrast: if True, compute correlations for opposite contrast
        interpolate: if True, interpolate groundtruth tuning curves to match the
            predicted tuning curve angles.
        fill_missing: value to fill missing values with.
    """
    if interpolate:
        from scipy.interpolate import interp1d
    else:
        interp1d = None

    gt_angles = np.arange(0, 360, 30)
    if angles is None:
        angles = gt_angles
    if intensities is None:
        intensities = [0, 1]
        assert tuning.shape[1]

    if aggregate_dims is not None and isinstance(aggregate_dims, int):
        aggregate_dims = (aggregate_dims,)

    cell_types = tuning.responses.cell_types
    tuning = (
        tuning[:].squeeze(tuning.temporal_dim).filled(fill_missing)
    )  # remove temporal dim
    (
        n_angles,
        n_intensities,
        *sample_stim_shape,
        n_cell_types,
    ) = tuning.shape
    # we compute correlations over all possible stimulus arguments and models
    tuning = tuning.reshape(n_angles, -1, n_cell_types)

    # check shapes
    assert n_angles == len(angles)
    assert n_angles == len(gt_angles) or interpolate
    assert n_intensities == len(intensities)

    correlations = {}

    if mode == "independent":
        if aggregate_dims is not None:
            aggregate_dims = tuple([
                (d - 1) for d in aggregate_dims
            ])  # since lose angle dims
        for cell_type in [
            "T4a",
            "T4b",
            "T4c",
            "T4d",
            "T5a",
            "T5b",
            "T5c",
            "T5d",
        ]:
            if cell_type not in correlations:
                correlations[cell_type] = []
            # has 12 elements, corresponding to n_angles
            gt_tuning = groundtruth_utils.tuning_curves[cell_type]
            if interpolate and n_angles != len(gt_angles):
                gt_tuning = interp1d(x=gt_angles, y=gt_tuning, kind="cubic")(angles)
            cell_idx = list(cell_types).index(cell_type)
            all_predicted_tuning_curves = tuning[..., cell_idx][:].squeeze()
            # (n_models, n_widths, n_intensities, n_speeds)
            correlations[cell_type] = quick_correlation_one_to_many(
                gt_tuning, all_predicted_tuning_curves.T
            ).reshape(n_intensities, *sample_stim_shape)

    elif mode == "joint":
        if aggregate_dims is not None:
            aggregate_dims = tuple([
                (d - 2) for d in aggregate_dims
            ])  # since lose angle and intensity dims
        groundtruth = []
        predicted = []
        for cell_type in [
            "T4a",
            "T4b",
            "T4c",
            "T4d",
            "T5a",
            "T5b",
            "T5c",
            "T5d",
        ]:
            gt_tuning = groundtruth_utils.tuning_curves[cell_type]
            if interpolate and n_angles != len(gt_angles):
                gt_tuning = interp1d(x=gt_angles, y=gt_tuning, kind="cubic")(angles)
            groundtruth.append(gt_tuning)
            cell_idx = list(cell_types).index(cell_type)
            _tuning = tuning[..., cell_idx][:]
            # in this instance, we must select ON for T4 and OFF for T5
            # (or else needed to do all combinarotiral correlations and then aggregate)
            _tuning = _tuning.reshape(n_angles, n_intensities, *sample_stim_shape)
            if "T4" in cell_type:
                _tuning = _tuning[:, [list(intensities).index(1)]]
            elif "T5" in cell_type:
                _tuning = _tuning[:, [list(intensities).index(0)]]
            predicted.append(_tuning.reshape(n_angles, -1))
        # (8, n_angles)
        groundtruth = np.array(groundtruth)
        # (8 * n_angles)
        groundtruth = groundtruth.flatten()
        # (8, n_angles, n_models * n_widths * 1 * n_speeds)
        predicted = np.array(predicted)
        # (8 * n_angles, n_models * n_widths * 1 * n_speeds)
        predicted = predicted.reshape(-1, predicted.shape[-1])
        # (n_models, n_widths, 1, n_speeds)
        corr = quick_correlation_one_to_many(groundtruth, predicted.T).reshape(
            *sample_stim_shape
        )
        correlations["joint"] = corr

    if respect_contrast and mode == "independent":
        # we need to select ON and OFF for T4 and T5 respectively
        for cell_type in correlations:
            if "T4" in cell_type:
                correlations[cell_type] = correlations[cell_type][
                    [list(intensities).index(1)]
                ]
            elif "T5" in cell_type:
                correlations[cell_type] = correlations[cell_type][
                    [list(intensities).index(0)]
                ]
    elif opposite_contrast and mode == "independent":
        # we need to select OFF and ON for T4 and T5 respectively
        for cell_type in correlations:
            if "T4" in cell_type:
                correlations[cell_type] = correlations[cell_type][
                    [list(intensities).index(0)]
                ]
            elif "T5" in cell_type:
                correlations[cell_type] = correlations[cell_type][
                    [list(intensities).index(1)]
                ]

    if aggregate == "median":
        for cell_type in correlations:
            correlations[cell_type] = np.median(
                correlations[cell_type], axis=aggregate_dims
            )
    elif aggregate == "mean":
        for cell_type in correlations:
            correlations[cell_type] = np.mean(
                correlations[cell_type], axis=aggregate_dims
            )
    elif aggregate == "max":
        for cell_type in correlations:
            correlations[cell_type] = np.max(correlations[cell_type], axis=aggregate_dims)
    elif aggregate == "absmax":
        for cell_type in correlations:

            def absmax(arr, *args, **kwargs):
                arrmax = np.max(arr, *args, **kwargs)
                arrmin = np.min(arr, *args, **kwargs)
                return np.where(np.abs(arrmax) > np.abs(arrmin), arrmax, arrmin)

            correlations[cell_type] = absmax(correlations[cell_type], axis=aggregate_dims)
    elif aggregate == "select":
        assert aggregate_dims is not None and select_stimuli_indices is not None
        for cell_type in correlations:
            correlations[cell_type] = select_along_axes(
                correlations[cell_type],
                select_stimuli_indices,
                aggregate_dims,
            )
    elif aggregate is None:
        pass

    if concatenate:
        correlations = np.concatenate(
            [correlations[cell_type] for cell_type in correlations],
            axis=-1,
        )
        return correlations

    return correlations


# -- plot ------------


def plot_dsis(
    dsis: np.ndarray,
    cell_types,
    scatter_best=True,
    scatter_all=True,
    bold_output_type_labels=True,
    output_cell_types=None,
    known_on_off_first=True,
    sorted_type_list=None,
    figsize=[10, 1],
    ylim=(0, 1),
    color_known_types=True,
    fontsize=6,
    fig=None,
    axes=None,
    **kwargs,
):
    if len(dsis.shape) == 2:
        dsis = dsis[None, :]
    if fig is None or axes is None:
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_axes([0, 0.525, 1, 0.475])
        plt_utils.rm_spines(ax1, spines=("bottom",))
        ax2 = fig.add_axes([0, 0, 1, 0.475])
        axes = [ax1, ax2]
    else:
        ax1 = axes[0]
        plt_utils.rm_spines(ax1, spines=("bottom",))
        ax2 = axes[1]

    for i, intensity in enumerate([1, 0]):
        color = ON if intensity == 1 else OFF
        fig, ax, *_ = dsi_violins(
            dsis=dsis[:, [intensity], :],
            cell_types=cell_types,
            cmap=None,
            color=color,
            fig=fig,
            ax=axes[i],
            fontsize=fontsize,
            sorted_type_list=sorted_type_list,
            figsize=figsize,
            scatter_best=scatter_best,
            scatter_all=scatter_all,
            known_on_off_first=known_on_off_first,
            **kwargs,
        )

        ax.grid(False)

        if bold_output_type_labels:
            plt_utils.boldify_labels(output_cell_types, ax)

        ax.set_ylim(*ylim)
        plt_utils.trim_axis(ax)
        plt_utils.set_spine_tick_params(
            ax,
            tickwidth=0.5,
            ticklength=3,
            ticklabelpad=2,
            spinewidth=0.5,
        )
        if color_known_types:
            plt_utils.color_labels(["T4a", "T4b", "T4c", "T4d"], ON, ax)
            plt_utils.color_labels(["T5a", "T5b", "T5c", "T5d"], OFF, ax)

    ax1.set_xticks([])
    ax1.set_yticks(np.arange(0, 1.2, 0.2))
    ax2.set_yticks(np.arange(0, 1.2, 0.2))
    # ax1.set_yticks(ax1.get_yticks()[1:])
    ax2.invert_yaxis()
    return fig, (ax1, ax2)


def dsi_violins(
    dsis,
    cell_types,
    scatter_best=True,
    scatter_all=False,
    cmap=plt.cm.Greens_r,
    colors=None,
    color="b",
    figsize=[10, 1],
    fontsize=6,
    showmeans=False,
    showmedians=True,
    sorted_type_list=None,
    sort_descending=False,
    scatter_edge_width=0.5,
    scatter_best_edge_width=0.75,
    scatter_edge_color="none",
    scatter_face_color="k",
    scatter_alpha=0.35,
    scatter_best_alpha=1.0,
    scatter_all_marker="o",
    scatter_best_index=None,
    scatter_best_marker="o",
    scatter_best_color=None,
    known_on_off_first=True,
    mean_median_linewidth=1.5,
    mean_median_bar_length=1.0,
    violin_alpha=1.0,
    **kwargs,
):
    # always add empty group axis for violin plot unless dsis is provided
    # with 3 axes
    if len(dsis.shape) == 1:
        dsis = dsis[None, None, :]
    elif len(dsis.shape) == 2:
        dsis = dsis[:, None]

    # transpose to #cell_types, #groups, #samples
    if dsis.shape[0] != len(cell_types):
        dsis = np.transpose(dsis, (2, 1, 0))

    if known_on_off_first:
        sorted_type_list = nodes_edges_utils.nodes_list_sorting_on_off_unknown(cell_types)

    if sorted_type_list is not None:
        dsis = nodes_edges_utils.sort_by_mapping_lists(
            cell_types, sorted_type_list, dsis, axis=0
        )
        cell_types = np.array(sorted_type_list)

    if sort_descending:
        medians = np.median(dsis, axis=(-2, -1))
        index = np.argsort(medians)[::-1]
        dsis = dsis[index]
        cell_types = cell_types[index]

    if colors is not None:
        pass
    elif cmap is not None:
        colors = None
    elif color is not None:
        cmap = None
        colors = (color,)

    fig, ax, colors = violin_groups(
        dsis,
        cell_types[:],
        rotation=90,
        scatter=False,
        cmap=cmap,
        colors=colors,
        fontsize=fontsize,
        figsize=figsize,
        width=0.7,
        showmeans=showmeans,
        showmedians=showmedians,
        mean_median_linewidth=mean_median_linewidth,
        mean_median_bar_length=mean_median_bar_length,
        violin_alpha=violin_alpha,
        **kwargs,
    )
    if dsis.shape[1] == 1:
        plt_utils.scatter_on_violins_with_best(
            dsis.T.squeeze(),
            ax,
            scatter_best,
            scatter_all,
            best_index=scatter_best_index,
            linewidth=scatter_edge_width,
            best_linewidth=scatter_best_edge_width,
            edgecolor=scatter_edge_color,
            facecolor=scatter_face_color,
            all_scatter_alpha=scatter_alpha,
            best_scatter_alpha=scatter_best_alpha,
            all_marker=scatter_all_marker,
            best_marker=scatter_best_marker,
            best_color=scatter_best_color,
        )
    return fig, ax, colors, dsis


@dataclass
class TuningCurveData:
    tuning_curve: MovingBarResponseView
    angles: List[float]
    intensities: List[float]

    def __getitem__(self, key):
        return TuningCurveData(self.tuning_curve[key], self.angles, self.intensities)
