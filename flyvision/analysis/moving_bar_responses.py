from __future__ import annotations

from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from flyvision.datasets.moving_bar import mask_between_seconds, time_window
from flyvision.utils import groundtruth_utils, nodes_edges_utils
from flyvision.utils.color_utils import OFF, ON

from .visualization import plt_utils
from .visualization.plots import polar, violin_groups

__all__ = [
    "dsi_correlation_to_known",
    "plot_dsis",
    "dsi_violins",
    "peak_responses",
    "peak_responses_angular",
    "direction_selectivity_index",
    "preferred_direction",
    "plot_angular_tuning",
    "plot_T4_tuning",
    "plot_T5_tuning",
    "get_groundtruth_tuning_curves",
    "correlation_to_known_tuning_curves",
    "angular_distance_to_known",
]

# class MovingBarResponseView(StimulusResponseIndexer):
#     def __init__(
#         self,
#         arg_df: pd.DataFrame,
#         config: Namespace,
#         responses: CellTypeArray,
#         stim_sample_dim=0,
#         temporal_dim=1,
#         time=None,
#     ):
#         self.config = config
#         if time is None:
#             if len(config.offsets) == 2:
#                 n_offsets = config.offsets[1] - config.offsets[0]
#             else:
#                 n_offsets = len(config.offsets)
#             t_stim = (n_offsets * np.radians(2.25)) / (
#                 np.array(config.speeds) * np.radians(5.8)
#             )
#             time = np.arange(
#                 -config.t_pre, np.max(t_stim) + config.t_post - config.dt, config.dt
#             )
#         super().__init__(
#             arg_df=arg_df,  # could also construct from config
#             responses=responses,
#             dt=config.dt,
#             t_pre=config.t_pre,
#             stim_sample_dim=stim_sample_dim,
#             temporal_dim=temporal_dim,
#             time=time,
#         )

#     def view(
#         self,
#         arg_df: pd.DataFrame = None,
#         config: Namespace = None,
#         responses: Union[np.ndarray, CellTypeArray] = None,
#         stim_sample_dim=None,
#         temporal_dim=None,
#         time=None,
#     ) -> "MovingBarResponseView":
#         if isinstance(responses, (np.ndarray, np.ma.MaskedArray)):
#             responses = CellTypeArray(responses, cell_types=self.responses.cell_types)

#         return self.__class__(
#             arg_df if np.any(arg_df) else self.arg_df,
#             config if config is not None else self.config,
#             responses if responses is not None else self.responses,
#             stim_sample_dim if np.any(stim_sample_dim) else self.stim_sample_dim,
#             temporal_dim or self.temporal_dim,
#             time if np.any(time) else self.time,
#         )

#     def angular(self, dim=None) -> "MovingBarResponseView":
#         if dim is None:
#             dim = self.stim_sample_dim
#             view = self.reshape_stim_sample_dim("angle")
#         else:
#             view = self
#         shape = view.shape
#         angles = np.radians(self.arg_df.angle.unique())

#         complex_responses = view.responses[:] * np.exp(
#             np.expand_dims(angles, [i for i in range(len(shape)) if i != dim]) * 1j
#         )
#         return self.view(
#             responses=complex_responses, stim_sample_dim=view.stim_sample_dim
#         )

#     def peak_responses(
#         self,
#         norm=None,
#         from_degree=None,
#         to_degree=None,
#     ) -> "MovingBarResponseView":
#         """Peak responses from rectified voltages that are normalized per cell
#         if a norm is provided. The normalization does not change the
#         scale-invariant
#         direction selectivity index (DSI) of the responses, but if responses
#         are averaged it will make the average more represetnative of the
#         population.
#         """
#         from_degree = from_degree or self.config.offsets[0] * 2.25
#         to_degree = to_degree or (self.config.offsets[1] - 1) * 2.25
#         masks = self.get_time_masks(
#             from_degree / 5.8,
#             to_degree / 5.8,
#         )
#         # original way from the paper
#         view = self.masked(
#             masks, mask_dims=(self.stim_sample_dim, self.temporal_dim)
#         ).rectify()

#         if norm is not None:
#             view = view.divide_by_given_array(norm, dims=(0, -1))

#         peak = view.peak()

#         # from n_models, n_stimuli, n_timesteps, n_cell_types
#         # to n_angle, n_models, n_width, n_intensity, n_speed, n_timesteps, n_cell_types
#         peak = peak.reshape_stim_sample_dim(
#             "angle", "width", "intensity", "speed"
#         ).transpose(1, 0, 2, 3, 4, 5, 6)
#         return peak

#     def peak_responses_angular(
#         self, norm=None, from_degree=None, to_degree=None
#     ) -> "MovingBarResponseView":
#         view = self.peak_responses(norm, from_degree, to_degree)
#         # make complex over angles
#         view = view.angular(dim=0)
#         return view

#     def get_time_masks(self, from_column=-1.5, to_column=1.5):
#         masks = {}
#         for speed in self.arg_df.speed.unique():
#             masks[speed] = mask_between_seconds(
#                 *time_window(
#                     speed,
#                     from_column=from_column,
#                     to_column=to_column,
#                     start=self.config.offsets[0],
#                     end=self.config.offsets[1],
#                 ),
#                 self.time,
#             )
#         return np.array([masks[speed] for speed in self.arg_df.speed.to_list()])

#     def dsi(
#         self, average=True, norm=None, from_degree=None, to_degree=None
#     ) -> "MovingEdgeResponseView":
#         view = self.peak_responses_angular(norm, from_degree, to_degree)
#         # compute DSI
#         vector_sum = view.sum(dims=(0,))
#         vector_length = vector_sum.abs()
#         normalization = view.abs().sum(dims=(0,)).max(dims=(3,), keepdims=True)
#         dsi = vector_length / (normalization + np.array([1e-15]))

#         return dsi
#         if average:
#             # average over widths and speeds
#             dsi = dsi.mean(dims=(2, 4), keepdims=True)
#         return dsi.squeeze()

#     def preferred_direction(
#         self, average=True, norm=None, from_degree=None, to_degree=None
#     ):
#         view = self.peak_responses_angular(norm, from_degree, to_degree)
#         vector_sum = view.sum(dims=(0,))
#         theta_pref = np.angle(vector_sum[:])
#         if average:
#             # average over widths and speeds
#             theta_pref = np.angle(vector_sum.sum(dims=(2, 4))[:])
#         return theta_pref

#     def plot_traces(
#         self, cell_type, t_start=None, t_end=None, plot_kwargs=dict(), **stim_kwargs
#     ):
#         if "title" not in plot_kwargs:
#             plot_kwargs["title"] = f"{cell_type} moving-bar response"
#         return super().plot_traces(
#             cell_type=cell_type,
#             t_start=t_start,
#             t_end=t_end,
#             plot_kwargs=plot_kwargs,
#             **stim_kwargs,
#         )

#     def plot_angular_tuning(
#         self,
#         cell_type,
#         intensity,
#         quantile=None,
#         figsize=[1, 1],
#         fontsize=5,
#         linewidth=1,
#         anglepad=-7,
#         xlabelpad=-1,
#         groundtruth=False,
#         groundtruth_linewidth=1.0,
#         fig=None,
#         ax=None,
#         peak_responses=None,
#         compare_across_contrasts=False,
#         weighted_average=None,
#         average_models=False,
#         colors=None,
#         **kwargs,
#     ):
#         """
#         Args:
#             tuning (optional): If provided, use this tuning instead of computing it.
#                 Must be a MovingEdgeResponseView of shape
#                 (12, 50, 1, 2, 6, 1, 65).
#         """

#         if peak_responses is None:
#             peak_responses = self.peak_responses()

#         peak_responses = peak_responses[cell_type]
#         # squeeze width, time, and cell_type dims
#         # TODO: this will break with width dim > 1
#         peak_responses = peak_responses[cell_type].squeeze(dims=(2, -2, -1))
#         # average over speeds
#         average_tuning = peak_responses.mean(dims=(3,))
#         # average over models
#         if average_models:
#             average_tuning = average_tuning.average(dims=(1,), weights=weighted_average)

#         if quantile:
#             # average over speeds and compute the quantile over models
#             quantile = peak_responses.mean(dims=(3,)).quantile(quantile, dims=(1,))[:]

#         # being verbose here to avoid misunderstandings cause intensity and their
#         # index are identical
#         # TODO: this won't work if intensity is not 0 or 1
#         if intensity == 0:
#             index = 0
#         elif intensity == 1:
#             index = 1
#         else:
#             raise NotImplementedError("Only intensity 0 and 1 are supported")
#         r_predicted = average_tuning.take_single(indices=index, dims=-1)

#         if compare_across_contrasts:
#             r_predicted = r_predicted[:].squeeze() / (
#                 rabsmax := average_tuning.abs()
#                 .max(
#                     dims=(
#                         0,
#                         -1,
#                     )
#                 )[:]
#                 .squeeze()
#             )

#         else:
#             r_predicted = r_predicted[:].squeeze() / (
#                 rabsmax := r_predicted.abs()
#                 .max(
#                     dims=(
#                         0,
#                         -1,
#                     )
#                 )[:]
#                 .squeeze()
#             )

#         color = (ON if intensity == 1 else OFF) if colors is None else colors

#         angles = self.arg_df.angle.unique()

#         fig, ax = polar(
#             angles,
#             r_predicted,
#             figsize=figsize,
#             fontsize=fontsize,
#             linewidth=linewidth,
#             anglepad=anglepad,
#             xlabelpad=xlabelpad,
#             color=color,
#             fig=fig,
#             ax=ax,
#             **kwargs,
#         )

#         if np.any(quantile):
#             quantile = np.take(quantile, intensity, axis=-1)
#             quantile = quantile / rabsmax
#             closed = angles[-1] % 360 == angles[0]
#             if not closed:
#                 angles = np.array([*angles, angles[0]])
#                 quantile = np.append(quantile, quantile[:, 0][:, None], axis=1)
#             ax.fill_between(
#                 np.radians(angles),
#                 quantile[0],
#                 quantile[1],
#                 facecolor=adapt_color_alpha(color, 0.1),
#                 edgecolor=color,
#                 linewidth=0.25,
#                 zorder=0,
#             )

#         if groundtruth and cell_type in groundtruth_utils.tuning_curves:
#             r = np.array(groundtruth_utils.tuning_curves[cell_type])
#             r = r / np.max(np.abs(r))
#             theta = np.arange(0, 360, 360 / len(r))
#             polar(
#                 theta,
#                 r,
#                 figsize=figsize,
#                 fontsize=fontsize,
#                 linewidth=groundtruth_linewidth,
#                 anglepad=anglepad,
#                 xlabelpad=xlabelpad,
#                 color="k",
#                 fig=fig,
#                 ax=ax,
#                 **kwargs,
#             )
#         return fig, ax

#     def plot_t4_tuning(self, tuning):
#         fig, axes, _ = plt_utils.get_axis_grid(
#             range(4),
#             projection="polar",
#             aspect_ratio=4,
#             figsize=[2.95, 0.83],
#             wspace=0.25,
#         )
#         for i, cell_type in enumerate(["T4a", "T4b", "T4c", "T4d"]):
#             self.plot_angular_tuning(
#                 cell_type,
#                 intensity=1,
#                 fig=fig,
#                 ax=axes[i],
#                 groundtruth=True,
#                 aggregate_models="mean",
#                 linewidth=1.0,
#                 scale_to_max_over_all_stimuli=False,
#                 tuning=tuning,
#             )
#             axes[i].set_xlabel(cell_type)

#         for ax in axes:
#             ax.xaxis.label.set_fontsize(8)
#             [i.set_linewidth(0.5) for i in ax.spines.values()]
#             ax.grid(True, linewidth=0.5)

#     def plot_t5_tuning(self, tuning):
#         fig, axes, _ = plt_utils.get_axis_grid(
#             range(4),
#             projection="polar",
#             aspect_ratio=4,
#             figsize=[2.95, 0.83],
#             wspace=0.25,
#         )
#         for i, cell_type in enumerate(["T5a", "T5b", "T5c", "T5d"]):
#             self.plot_angular_tuning(
#                 cell_type,
#                 intensity=0,
#                 fig=fig,
#                 ax=axes[i],
#                 groundtruth=True,
#                 aggregate_models="mean",
#                 linewidth=1.0,
#                 scale_to_max_over_all_stimuli=False,
#                 tuning=tuning,
#             )
#             axes[i].set_xlabel(cell_type)

#         for ax in axes:
#             ax.xaxis.label.set_fontsize(8)
#             [i.set_linewidth(0.5) for i in ax.spines.values()]
#             ax.grid(True, linewidth=0.5)


# class MovingEdgeResponseView(MovingBarResponseView):
#     def plot_traces(
#         self, cell_type, t_start=None, t_end=None, plot_kwargs=dict(), **stim_kwargs
#     ):
#         if "title" not in plot_kwargs:
#             plot_kwargs["title"] = f"{cell_type} moving-edge response"
#         return super().plot_traces(
#             cell_type=cell_type,
#             t_start=t_start,
#             t_end=t_end,
#             plot_kwargs=plot_kwargs,
#             **stim_kwargs,
#         )


# -- correlation ------------


# def dsi_correlation_to_known(
#     dsis,  # TODO: maybe make into CellTypeArray
#     cell_types,
#     respect_contrast=False,
#     aggregate="mean",
#     agg_dim=None,
# ):
#     """
#     Computes the correlation between predicted DSIs and known DSIs.

#     Args:
#         dsis (np.ndarray): Array of shape (2, n_cell_types) containing the DSIs for
#             ON and OFF intensities.
#         cell_types (list): List of cell types.
#         respect_contrast (bool): If True, respect T4 and T5 contrast preference, i.e.,
#             only compute DSI for preferred contrast.
#         aggregate (str): Aggregation method for the correlation values. Can be
#             "mean" or "median".
#         agg_dim (int): Dimension along which to aggregate the correlation values.

#     Note: known DSIs are binary, either 0 or 1, depending on whether the cell type
#     is known to be motion-tuned or not.
#     """
#     assert dsis.shape[0] == 2, (
#         "First dimension of `dsis` array should correspond to stimulus intensity (on/"
#         "off) and thus have length 2"
#     )
#     assert dsis.shape[-1] == len(
#         cell_types
#     ), "Last dimension of `dsis` array should correspond to cell type"
#     motion_tuning = groundtruth_utils.motion_tuning
#     cell_types_list = list(cell_types)

#     if not respect_contrast:
#         # note: includes T4 and T5 major inputs
#         known_dsi_types = groundtruth_utils.known_dsi_types

#         known_idxs = [cell_types_list.index(cell_type) for cell_type in known_dsi_types]
#         dsis_for_known = dsis[..., known_idxs]

#         if agg_dim is not None:
#             if aggregate == "max":
#                 dsis_for_known = np.max(dsis_for_known, axis=agg_dim)
#             elif aggregate == "median":
#                 dsis_for_known = np.median(dsis_for_known, axis=agg_dim)
#             else:
#                 raise ValueError

#         groundtruth_mt = np.array([
#             1.0 if nt in motion_tuning else 0.0 for nt in known_dsi_types
#         ])

#         shape = dsis_for_known.shape
#         dsis_for_known = dsis_for_known.reshape(
#             2, -1, len(groundtruth_mt)
#         )  # flatten all model dims
#         dsis_for_known = dsis_for_known.swapaxes(0, 1).reshape(-1, len(groundtruth_mt))

#         corr_dsi, _ = correlation(groundtruth_mt, dsis_for_known)

#         corr_dsi = corr_dsi.reshape(-1, 2).T.reshape(*shape[:-1])
#     else:
#         on_motion_tuning = groundtruth_utils.on_motion_tuning
#         off_motion_tuning = groundtruth_utils.off_motion_tuning
#         no_motion_tuning = groundtruth_utils.no_motion_tuning

#         on_idxs = [cell_types_list.index(cell_type) for cell_type in on_motion_tuning]
#         off_idxs = [cell_types_list.index(cell_type) for cell_type in off_motion_tuning]
#         no_idxs = [cell_types_list.index(cell_type) for cell_type in no_motion_tuning]
#         # pick ON
#         on_motion_dsi = dsis[..., on_idxs][1]
#         # pick OFF
#         off_motion_dsi = dsis[..., off_idxs][0]
#         # aggregate over ON and OFF
#         no_motion_dsi = np.median(dsis[..., no_idxs], axis=(0,))

#         groundtruth_mt = [
#             1.0 if cell_type in motion_tuning else 0.0
#             for cell_type in [
#                 *on_motion_tuning,
#                 *off_motion_tuning,
#                 *no_motion_tuning,
#             ]
#         ]

#         dsis_for_known = np.concatenate(
#             (on_motion_dsi, off_motion_dsi, no_motion_dsi), axis=-1
#         )

#         shape = dsis_for_known.shape
#         dsis_for_known = dsis_for_known.reshape(
#             -1, len(groundtruth_mt)
#         )  # flatten all model dims

#         corr_dsi, _ = correlation(groundtruth_mt, dsis_for_known)

#         if len(shape) > 3:
#             corr_dsi = corr_dsi.T.reshape(*shape[1:-1])

#         if aggregate == "max":
#             corr_dsi = np.max(corr_dsi, axis=agg_dim)
#         elif aggregate == "median":
#             corr_dsi = np.median(corr_dsi, axis=agg_dim)
#     return corr_dsi


def dsi_correlation_to_known(
    dsis: xr.DataArray, max_aggregate_dims=("intensity",)
) -> xr.DataArray:
    """
    Computes the correlation between predicted DSIs and known DSIs.

    Args:
        dsis (xarray.DataArray): DataArray containing the DSIs for
            ON and OFF intensities, with dimensions including 'intensity' and 'neuron',
            and a coordinate 'cell_type'.
        aggregate_dims (Iterable): Dimensions along which to max-aggregate the
            dsi before computing the correlation. Default is ('intensity',).

    Note: Known DSIs
        Binary, either 0 or 1, depending on whether the cell type
        is known to be motion-tuned or not.
    """
    # Ensure the 'intensity' dimension has length 2
    assert dsis.sizes['intensity'] == 2, "Dimension 'intensity' should have length 2"

    # Retrieve ground truth motion tuning information
    motion_tuning = groundtruth_utils.motion_tuning
    known_dsi_types = groundtruth_utils.known_dsi_types

    # Select dsis for known cell types
    dsis_for_known = dsis.where(dsis['cell_type'].isin(known_dsi_types), drop=True).max(
        dim=max_aggregate_dims
    )

    # Construct ground truth motion tuning array
    groundtruth_mt = xr.DataArray(
        [
            1.0 if ct in motion_tuning else 0.0
            for ct in dsis_for_known['cell_type'].values
        ],
        coords={'neuron': dsis_for_known['neuron']},
        dims=['neuron'],
    )

    # Compute correlation along 'neuron' dimension
    corr_dsi = xr.corr(dsis_for_known, groundtruth_mt, dim='neuron')

    return corr_dsi


# def tuning_curve_correlation_to_known(
#     tuning: MovingBarResponseView,
#     # TODO: maybe make specific tuningresponseview subclass for safety
#     angles=None,
#     intensities=None,
#     mode="independent",
#     aggregate="absmax",
#     aggregate_dims=None,
#     select_stimuli_indices=None,
#     respect_contrast=True,
#     concatenate=False,
#     opposite_contrast=False,
#     interpolate=False,
#     fill_missing=0.0,
# ):
#     """
#     Computes the correlation between predicted tuning curves and known tuning curves.

#     Args:
#         tuning: Tuning responses.
#         angles: Angles for which to compute the correlation.
#         intensities: Intensities for which to compute the correlation.
#         mode: independent, joint -- either compute correlations for each cell type
#             independently or jointly
#         aggregate: median, mean, max -- aggregate over stim params
#         aggregate_dims: dimension along which to aggregate the correlation values
#         select_stimuli_indices: can select specific stimuli indices to compute
#             correlations for.
#         respect_contrast: if True, respect to compute correlations for preferred
#             contrast tunings instead of all contrasts
#         concatenate: if True, concatenate all cell types into one array
#         opposite_contrast: if True, compute correlations for opposite contrast
#         interpolate: if True, interpolate groundtruth tuning curves to match the
#             predicted tuning curve angles.
#         fill_missing: value to fill missing values with.
#     """
#     if interpolate:
#         from scipy.interpolate import interp1d
#     else:
#         interp1d = None

#     gt_angles = np.arange(0, 360, 30)
#     if angles is None:
#         angles = gt_angles
#     if intensities is None:
#         intensities = [0, 1]
#         assert tuning.shape[1]

#     if aggregate_dims is not None and isinstance(aggregate_dims, int):
#         aggregate_dims = (aggregate_dims,)

#     cell_types = tuning.responses.cell_types
#     tuning = (
#         tuning[:].squeeze(tuning.temporal_dim).filled(fill_missing)
#     )  # remove temporal dim
#     (
#         n_angles,
#         n_intensities,
#         *sample_stim_shape,
#         n_cell_types,
#     ) = tuning.shape
#     # we compute correlations over all possible stimulus arguments and models
#     tuning = tuning.reshape(n_angles, -1, n_cell_types)

#     # check shapes
#     assert n_angles == len(angles)
#     assert n_angles == len(gt_angles) or interpolate
#     assert n_intensities == len(intensities)

#     correlations = {}

#     if mode == "independent":
#         if aggregate_dims is not None:
#             aggregate_dims = tuple([
#                 (d - 1) for d in aggregate_dims
#             ])  # since lose angle dims
#         for cell_type in [
#             "T4a",
#             "T4b",
#             "T4c",
#             "T4d",
#             "T5a",
#             "T5b",
#             "T5c",
#             "T5d",
#         ]:
#             if cell_type not in correlations:
#                 correlations[cell_type] = []
#             # has 12 elements, corresponding to n_angles
#             gt_tuning = groundtruth_utils.tuning_curves[cell_type]
#             if interpolate and n_angles != len(gt_angles):
#                 gt_tuning = interp1d(x=gt_angles, y=gt_tuning, kind="cubic")(angles)
#             cell_idx = list(cell_types).index(cell_type)
#             all_predicted_tuning_curves = tuning[..., cell_idx][:].squeeze()
#             # (n_models, n_widths, n_intensities, n_speeds)
#             correlations[cell_type] = quick_correlation_one_to_many(
#                 gt_tuning, all_predicted_tuning_curves.T
#             ).reshape(n_intensities, *sample_stim_shape)

#     elif mode == "joint":
#         if aggregate_dims is not None:
#             aggregate_dims = tuple([
#                 (d - 2) for d in aggregate_dims
#             ])  # since lose angle and intensity dims
#         groundtruth = []
#         predicted = []
#         for cell_type in [
#             "T4a",
#             "T4b",
#             "T4c",
#             "T4d",
#             "T5a",
#             "T5b",
#             "T5c",
#             "T5d",
#         ]:
#             gt_tuning = groundtruth_utils.tuning_curves[cell_type]
#             if interpolate and n_angles != len(gt_angles):
#                 gt_tuning = interp1d(x=gt_angles, y=gt_tuning, kind="cubic")(angles)
#             groundtruth.append(gt_tuning)
#             cell_idx = list(cell_types).index(cell_type)
#             _tuning = tuning[..., cell_idx][:]
#             # in this instance, we must select ON for T4 and OFF for T5
#             # (or else needed to do all combinarotiral correlations and then aggregate)
#             _tuning = _tuning.reshape(n_angles, n_intensities, *sample_stim_shape)
#             if "T4" in cell_type:
#                 _tuning = _tuning[:, [list(intensities).index(1)]]
#             elif "T5" in cell_type:
#                 _tuning = _tuning[:, [list(intensities).index(0)]]
#             predicted.append(_tuning.reshape(n_angles, -1))
#         # (8, n_angles)
#         groundtruth = np.array(groundtruth)
#         # (8 * n_angles)
#         groundtruth = groundtruth.flatten()
#         # (8, n_angles, n_models * n_widths * 1 * n_speeds)
#         predicted = np.array(predicted)
#         # (8 * n_angles, n_models * n_widths * 1 * n_speeds)
#         predicted = predicted.reshape(-1, predicted.shape[-1])
#         # (n_models, n_widths, 1, n_speeds)
#         corr = quick_correlation_one_to_many(groundtruth, predicted.T).reshape(
#             *sample_stim_shape
#         )
#         correlations["joint"] = corr

#     if respect_contrast and mode == "independent":
#         # we need to select ON and OFF for T4 and T5 respectively
#         for cell_type in correlations:
#             if "T4" in cell_type:
#                 correlations[cell_type] = correlations[cell_type][
#                     [list(intensities).index(1)]
#                 ]
#             elif "T5" in cell_type:
#                 correlations[cell_type] = correlations[cell_type][
#                     [list(intensities).index(0)]
#                 ]
#     elif opposite_contrast and mode == "independent":
#         # we need to select OFF and ON for T4 and T5 respectively
#         for cell_type in correlations:
#             if "T4" in cell_type:
#                 correlations[cell_type] = correlations[cell_type][
#                     [list(intensities).index(0)]
#                 ]
#             elif "T5" in cell_type:
#                 correlations[cell_type] = correlations[cell_type][
#                     [list(intensities).index(1)]
#                 ]

#     if aggregate == "median":
#         for cell_type in correlations:
#             correlations[cell_type] = np.median(
#                 correlations[cell_type], axis=aggregate_dims
#             )
#     elif aggregate == "mean":
#         for cell_type in correlations:
#             correlations[cell_type] = np.mean(
#                 correlations[cell_type], axis=aggregate_dims
#             )
#     elif aggregate == "max":
#         for cell_type in correlations:
#             correlations[cell_type] = np.max(correlations[cell_type],
# axis=aggregate_dims)
#     elif aggregate == "absmax":
#         for cell_type in correlations:

#             def absmax(arr, *args, **kwargs):
#                 arrmax = np.max(arr, *args, **kwargs)
#                 arrmin = np.min(arr, *args, **kwargs)
#                 return np.where(np.abs(arrmax) > np.abs(arrmin), arrmax, arrmin)

#             correlations[cell_type] = absmax(correlations[cell_type],
# axis=aggregate_dims)
#     elif aggregate == "select":
#         assert aggregate_dims is not None and select_stimuli_indices is not None
#         for cell_type in correlations:
#             correlations[cell_type] = select_along_axes(
#                 correlations[cell_type],
#                 select_stimuli_indices,
#                 aggregate_dims,
#             )
#     elif aggregate is None:
#         pass

#     if concatenate:
#         correlations = np.concatenate(
#             [correlations[cell_type] for cell_type in correlations],
#             axis=-1,
#         )
#         return correlations

#     return correlations


# -- plot ------------


def plot_dsis(
    dsis: xr.DataArray,
    cell_types: xr.DataArray,
    scatter_best=True,
    scatter_all=True,
    bold_output_type_labels=False,
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
            dsis=dsis.sel(intensity=intensity).values,
            cell_types=cell_types.values,
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
    scatter_best=False,
    scatter_all=True,
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


def get_time_masks(
    dataset: xr.Dataset, from_column: float = -1.5, to_column: float = 1.5
) -> xr.DataArray:
    """
    Generate time masks for each sample based on speed and specified column range.

    Args:
        dataset (xr.Dataset): The input dataset containing 'speed' and 'time' coordinates.
        from_column (float): Start of the column range.
        to_column (float): End of the column range.

    Returns:
        xr.DataArray: A boolean mask with dimensions ('sample', 'frame').
    """
    speeds = dataset['speed'].values
    unique_speeds = np.unique(speeds)
    config = dataset.attrs['config']
    start, end = config['offsets']
    times = dataset['time'].values

    # Precompute masks for unique speeds
    mask_dict = {}
    for speed in unique_speeds:
        t_start, t_end = time_window(
            speed, from_column=from_column, to_column=to_column, start=start, end=end
        )
        mask_dict[speed] = mask_between_seconds(t_start, t_end, times)

    # Map masks to each sample based on its speed
    masks = np.array([mask_dict[speed] for speed in speeds])

    # Create a DataArray for the masks
    mask_da = xr.DataArray(
        data=masks,
        dims=('sample', 'frame'),
        coords={'sample': dataset['sample'], 'frame': dataset['frame']},
    )

    return mask_da


def peak_responses(
    dataset: xr.Dataset,
    norm: xr.DataArray = None,
    from_degree: float = None,
    to_degree: float = None,
) -> xr.DataArray:
    """
    Compute peak responses from rectified voltages, optionally normalized.

    Args:
        dataset (xr.Dataset): The input dataset containing 'responses' and necessary
            coordinates.
        norm (xr.DataArray, optional): Normalization array. Defaults to None.
        from_degree (float, optional): Starting degree for masking. Defaults to None.
        to_degree (float, optional): Ending degree for masking. Defaults to None.

    Returns:
        xr.DataArray: Peak responses with reshaped and transposed dimensions.
    """
    config = dataset.attrs['config']
    from_degree = from_degree if from_degree is not None else config['offsets'][0] * 2.25
    to_degree = to_degree if to_degree is not None else (config['offsets'][1] - 1) * 2.25

    # Generate time masks
    masks = get_time_masks(
        dataset, from_column=from_degree / 5.8, to_column=to_degree / 5.8
    )

    # Apply masks to responses and rectify
    responses = dataset['responses']
    masked = responses.where(masks, other=0)
    rectified = masked.clip(min=0)  # Rectify: max(0, response)

    # Normalize if provided
    if norm is not None:
        rectified = rectified / norm

    # Compute peak (maximum over 'frame')
    peak = rectified.max(dim='frame')
    return peak


def get_groundtruth_tuning_curves(cell_types: List[str], angles: np.ndarray):
    """
    Retrieves the ground truth tuning curves for the specified cell types.
    Optionally interpolates the curves to match the provided angles.
    """
    gt_angles = np.arange(0, 360, 30)
    tuning_curves = []

    for cell_type in cell_types:
        gt_tuning = groundtruth_utils.tuning_curves[cell_type]
        interp_func = interp1d(
            gt_angles, gt_tuning, kind='cubic', fill_value="extrapolate"
        )
        gt_tuning = interp_func(angles)
        tuning_curves.append(gt_tuning)

    dataset = xr.DataArray(
        np.array(tuning_curves),
        dims=['neuron', 'angle'],
        coords={'cell_type': ("neuron", cell_types), 'angle': angles},
    )

    return dataset


def correlation_to_known_tuning_curves(dataset: xr.Dataset, absmax=False) -> xr.DataArray:
    tuning = peak_responses(dataset)
    gt_tuning = get_groundtruth_tuning_curves(
        ["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"], np.arange(0, 360, 30)
    )

    tuning = (
        tuning.set_index(sample=["angle", "intensity", "width", "speed"])
        .unstack("sample")
        .fillna(0.0)
        .custom.where(cell_type=["T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"])
    )

    # reset the neuron axis to make it compatible with the ground truth tuning curves
    tuning["neuron"] = np.arange(tuning.coords["neuron"].size)

    correlation = xr.corr(tuning, gt_tuning, dim="angle")
    correlation = correlation.fillna(0.0)
    if absmax:
        # take speed and width that maximize the magnitude of the correlation, regardless
        # of the sign
        argmax = np.abs(correlation).argmax(dim=("speed", "width"))
    else:
        # take speed and width that maximize the correlation as an experimentalist
        # would do
        argmax = correlation.argmax(dim=("speed", "width"))
    correlation = correlation.isel(argmax)
    return correlation


def peak_responses_angular(
    dataset: xr.Dataset,
    norm: xr.DataArray = None,
    from_degree: float = None,
    to_degree: float = None,
) -> xr.DataArray:
    """
    Compute peak responses and make them complex over angles.
    """
    peak = peak_responses(
        dataset, norm=norm, from_degree=from_degree, to_degree=to_degree
    )

    # Make complex over angles
    angles = peak['angle'].values
    radians = np.deg2rad(angles)
    # Expand dimensions to match broadcasting
    radians = radians[np.newaxis, :, np.newaxis]
    complex_peak = peak * np.exp(1j * radians)

    return complex_peak


def direction_selectivity_index(
    dataset: xr.Dataset,
    average: bool = True,
    norm: xr.DataArray = None,
    from_degree: float = None,
    to_degree: float = None,
) -> xr.DataArray:
    """
    Compute Direction Selectivity Index (DSI).
    """
    view = peak_responses_angular(
        dataset, norm=norm, from_degree=from_degree, to_degree=to_degree
    )
    view = view.set_index(sample=["angle", "width", "intensity", "speed"]).unstack(
        "sample"
    )

    # Compute vector sum over 'angle'
    vector_sum = view.sum(dim='angle')
    vector_length = np.abs(vector_sum)

    # Normalization: sum of absolute responses
    normalization = np.abs(view).sum(dim='angle').max(dim='intensity')
    dsi = vector_length / (normalization + 1e-15)

    if average:
        # Average over 'width' and 'speed'
        dsi = dsi.mean(dim=['width', 'speed'])

    return dsi.squeeze()


def preferred_direction(
    dataset: xr.Dataset,
    average: bool = True,
    norm: xr.DataArray = None,
    from_degree: float = None,
    to_degree: float = None,
) -> xr.DataArray:
    """
    Compute the preferred direction based on peak responses.

    Args:
        dataset (xr.Dataset): The input dataset.
        average (bool, optional): Whether to average over certain dimensions. Defaults
            to True.
        norm (xr.DataArray, optional): Normalization array. Defaults to None.
        from_degree (float, optional): Starting degree for masking. Defaults to None.
        to_degree (float, optional): Ending degree for masking. Defaults to None.

    Returns:
        xr.DataArray: Preferred direction angles in radians.
    """
    view = peak_responses_angular(
        dataset, norm=norm, from_degree=from_degree, to_degree=to_degree
    )
    view = view.set_index(sample=["angle", "width", "intensity", "speed"]).unstack(
        "sample"
    )

    # Compute vector sum over 'angle'
    vector_sum = view.sum(dim='angle')
    theta_pref = np.angle(vector_sum)

    if average:
        # Sum over 'width' and 'speed' before computing angle
        vector_sum = view.sum(dim=['width', 'speed', 'angle'])
        theta_pref = np.angle(vector_sum)

    vector_sum.data = theta_pref
    return vector_sum


def angular_distances(x: xr.DataArray, y: np.array, upper=np.pi):
    assert x.neuron.size == len(y)
    y_da = xr.DataArray(y, dims=['neuron'], coords={'neuron': x.coords['neuron']})

    result = xr.apply_ufunc(
        simple_angle_distance,
        x,
        y_da,
        input_core_dims=[['neuron'], ['neuron']],
        output_core_dims=[['neuron']],
        vectorize=True,
        kwargs={'upper': upper},
    )

    return result


def angular_distance_to_known(pds: xr.DataArray):
    t4s = pds.custom.where(cell_type=["T4a", "T4b", "T4c", "T4d"], intensity=1)
    t4_distances = angular_distances(t4s, np.array([np.pi, 0, np.pi / 2, 3 * np.pi / 2]))
    t5s = pds.custom.where(cell_type=["T5a", "T5b", "T5c", "T5d"], intensity=0)
    t5_distances = angular_distances(t5s, np.array([np.pi, 0, np.pi / 2, 3 * np.pi / 2]))
    # concatenate both xarrays again in the neuron dimension, drop intensity
    return xr.concat(
        [t4_distances.drop('intensity'), t5_distances.drop('intensity')], dim='neuron'
    )


def simple_angle_distance(a, b, upper=np.pi):
    """Element-wise angle distance between 0 and pi radians.

    Args:
        a, b: angle in radians, same shape

    Returns: distance between 0 and pi radians.
    """
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    # a = np.radians(a)
    # b = np.radians(b)
    # make all angles positive between 0 and 2 * pi
    a = a % (2 * np.pi)
    b = b % (2 * np.pi)

    y = np.zeros_like(a)
    # subtract the smaller angle from the larger one
    mask = a >= b
    y[mask] = a[mask] - b[mask]
    y[~mask] = b[~mask] - a[~mask]

    # map distances between pi and 2 pi to 0 and pi
    y[y > np.pi] = 2 * np.pi - y[y > np.pi]

    # map distances between 0 and pi to 0 and upper
    return y / np.pi * upper


def plot_angular_tuning(
    dataset: xr.Dataset,
    cell_type: int,
    intensity: int,
    figsize=(1, 1),
    fontsize: int = 5,
    linewidth: float = 1,
    anglepad: float = -7,
    xlabelpad: float = -1,
    groundtruth: bool = True,
    groundtruth_linewidth: float = 1.0,
    fig: plt.Figure = None,
    ax: plt.Axes = None,
    peak_responses_da: xr.DataArray = None,
    weighted_average: xr.DataArray = None,
    average_models: bool = False,
    colors: str = None,
    zorder: int | Iterable = 0,
    **kwargs,
):
    """
    Plot angular tuning for a specific cell type and intensity.

    Args:
        dataset (xr.Dataset): The input dataset.
        cell_type (int): The neuron index to plot.
        intensity (int): The intensity level (0 or 1).
        figsize (tuple, optional): Figure size. Defaults to (6, 6).
        fontsize (int, optional): Font size. Defaults to 12.
        linewidth (float, optional): Line width. Defaults to 2.
        anglepad (float, optional): Angle padding. Defaults to -7.
        xlabelpad (float, optional): X-label padding. Defaults to -1.
        groundtruth (bool, optional): Whether to plot ground truth. Defaults to False.
        groundtruth_linewidth (float, optional): Line width for ground truth.
            Defaults to 1.0.
        fig (plt.Figure, optional): Existing figure. Defaults to None.
        ax (plt.Axes, optional): Existing axes. Defaults to None.
        peak_responses_da (xr.DataArray, optional): Precomputed peak responses.
            Defaults to None.
        weighted_average (xr.DataArray, optional): Weights for averaging models.
            Defaults to None.
        average_models (bool, optional): Whether to average across models.
            Defaults to False.
        colors (str, optional): Color for the plot. Defaults to None.
        **kwargs: Additional keyword arguments for plotting.

    Returns:
        matplotlib.figure.Figure, matplotlib.axes.Axes: The figure and axes objects.
    """
    if peak_responses_da is None:
        peak_responses_da = peak_responses(dataset)

    peak_responses_da = peak_responses_da.set_index(
        sample=["angle", "width", "intensity", "speed"]
    ).unstack("sample")

    # Select the specific cell type
    peak = peak_responses_da.custom.where(cell_type=cell_type, intensity=intensity)

    # Squeeze irrelevant dimensions
    # peak = peak.squeeze(dim=['width', 'intensity', 'speed'], drop=True)

    # Average over speeds
    average_tuning = peak.mean(dim=('speed', 'width'))

    # Average over models if specified
    if average_models and weighted_average is not None:
        average_tuning = average_tuning.weighted(weighted_average).mean(dim='network_id')
    elif average_models:
        average_tuning = average_tuning.mean(dim='network_id')

    color = (ON if intensity == 1 else OFF) if colors is None else colors

    average_tuning = average_tuning / average_tuning.max()

    angles = average_tuning['angle'].values
    fig, ax = polar(
        angles,
        average_tuning.data.squeeze().T,
        figsize=figsize,
        fontsize=fontsize,
        linewidth=linewidth,
        anglepad=anglepad,
        xlabelpad=xlabelpad,
        color=color,
        fig=fig,
        ax=ax,
        zorder=zorder,
        **kwargs,
    )

    if groundtruth and cell_type in groundtruth_utils.tuning_curves:
        r_gt = np.array(groundtruth_utils.tuning_curves[cell_type])
        r_gt = r_gt / np.max(np.abs(r_gt))
        theta_gt = np.arange(0, 360, 360 / len(r_gt))
        polar(
            theta_gt,
            r_gt,
            figsize=figsize,
            fontsize=fontsize,
            linewidth=groundtruth_linewidth,
            anglepad=anglepad,
            xlabelpad=xlabelpad,
            color="k",
            fig=fig,
            ax=ax,
            # **kwargs,
        )

    return fig, ax


def plot_T4_tuning(dataset):
    fig, axes, _ = plt_utils.get_axis_grid(
        range(4),
        projection="polar",
        aspect_ratio=4,
        figsize=[2.95, 0.83],
        wspace=0.25,
    )
    for i, cell_type in enumerate(["T4a", "T4b", "T4c", "T4d"]):
        plot_angular_tuning(
            dataset,
            cell_type,
            intensity=1,
            fig=fig,
            ax=axes[i],
            groundtruth=True,
            aggregate_models="mean",
            linewidth=1.0,
        )
        axes[i].set_xlabel(cell_type)

    for ax in axes:
        ax.xaxis.label.set_fontsize(5)
        [i.set_linewidth(0.5) for i in ax.spines.values()]
        ax.grid(True, linewidth=0.5)


def plot_T5_tuning(dataset):
    fig, axes, _ = plt_utils.get_axis_grid(
        range(4),
        projection="polar",
        aspect_ratio=4,
        figsize=[2.95, 0.83],
        wspace=0.25,
    )
    for i, cell_type in enumerate(["T5a", "T5b", "T5c", "T5d"]):
        plot_angular_tuning(
            dataset,
            cell_type,
            intensity=0,
            fig=fig,
            ax=axes[i],
            groundtruth=True,
            aggregate_models="mean",
            linewidth=1.0,
        )
        axes[i].set_xlabel(cell_type)

    for ax in axes:
        ax.xaxis.label.set_fontsize(5)
        [i.set_linewidth(0.5) for i in ax.spines.values()]
        ax.grid(True, linewidth=0.5)
