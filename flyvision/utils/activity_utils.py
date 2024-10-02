"""Convenience and efficient access to activity of particular cells.

Example:
    layer_activity = LayerActivity(activity, network.connectome)
    T4a_response = layer_activity.T4a
    T5a_response = layer_activity.T5a
    T4b_central_response = layer_activity.central.T4a
"""

import operator
import weakref
from functools import reduce
from textwrap import wrap
from typing import Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from flyvision.connectome import ConnectomeDir, ReceptiveFields
from flyvision.utils import nodes_edges_utils

__all__ = [
    "CentralActivity",
    "LayerActivity",
    # "StimulusResponseIndexer",
    "SourceCurrentView",
    "asymmetric_weighting",
]


class CellTypeActivity(dict):
    """Base class for attribute-style access to network activity.

    Note, activity is stored as a weakref by default. This is for memory efficienty
    during training. If you want to keep a reference to the activity for analysis,
    set keepref=True.

    Args:
        keepref (bool, optional): Whether to keep a reference to the activity. Defaults
        to False.

    Attributes:
        activity (weakref.ref): Weak reference to the activity.
        keepref (bool): Whether to keep a reference to the activity.
    """

    def __init__(self, keepref=False):
        self.keepref = keepref

    def __dir__(self):
        return list(set([*dict.__dir__(self), *dict.__iter__(self)]))

    def __len__(self):
        return len(self.unique_cell_types)

    def __iter__(self):
        for cell_type in self.unique_cell_types:
            yield cell_type

    def __repr__(self):
        return "Activity of: \n{}".format("\n".join(wrap(", ".join(list(self)))))

    def update(self, activity):
        self.activity = activity

    def _slices(self, n):
        slices = tuple()
        for _ in range(n):
            slices += (slice(None, None, None),)
        return slices

    def __getattr__(self, key):
        activity = self.activity() if not self.keepref else self.activity
        if activity is None:
            return
        if isinstance(key, list):
            index = np.stack(list(map(lambda key: dict.__getitem__(self, key), key)))
            slices = self._slices(len(activity.shape) - 1)
            slices += (index,)
            return activity[slices]
        elif key == slice(None):
            return activity
        elif key in self.unique_cell_types:
            slices = self._slices(len(activity.shape) - 1)
            slices += (dict.__getitem__(self, key),)
            return activity[slices]
        elif key == "output":
            slices = self._slices(len(activity.shape) - 1)
            slices += (self.output_indices,)
            return activity[slices]
        elif key == "input":
            slices = self._slices(len(activity.shape) - 1)
            slices += (self.input_indices,)
            return activity[slices]
        elif "+" in key:
            _cell_types = key.split("+")
            return sum(map(self.__getattr__, _cell_types))
        elif "-" in key:
            _cell_types = key.split("-")
            return reduce(operator.sub, map(self.__getattr__, _cell_types))
        elif "*" in key:
            _cell_types = key.split("*")
            return reduce(operator.mul, map(self.__getattr__, _cell_types))
        elif "/" in key:
            _cell_types = key.split("/")
            return reduce(operator.truediv, map(self.__getattr__, _cell_types))
        elif key in self.__dict__:
            return self.__dict__[key]
        else:
            raise ValueError(f"{key}")

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setattr__(self, key, value):
        if key == "activity" and value is not None:
            if self.keepref is False:
                value = weakref.ref(value)
            object.__setattr__(self, key, value)
        else:
            object.__setattr__(self, key, value)

    def __copy__(self, memodict={}):
        return self

    def __deepcopy__(self, memodict={}):
        return self


class CentralActivity(CellTypeActivity):
    """Attribute-style access to central activity.

    Args:
        activity (array-like): activity of shape (..., n_cells)
        connectome: connectome dir with reference to
                        - connectome.nodes.layer_index
                        - connectome.unique_cell_types
                        - connectome.central_cells_index
        keepref (bool, optional): Whether to keep a reference to the activity.
            Defaults to False.

    Note, activity is stored as a weakref by default. This is for memory efficienty
    during training. If you want to keep a reference to the activity for analysis,
    set keepref=True.

    Attributes:
        activity (array-like): activity of shape (..., n_cells)
        unique_cell_types (array)
        index (NodeIndexer)
        input_indices (array)
        output_indices (array)

    Note: also allows 'virtual types' that are basic operations of individuals
        >>> a = LayerActivity(activity, network.connectome)
        >>> summed_a = a['L2+L4*L3/L5']
    """

    def __init__(
        self,
        activity: Union[NDArray, torch.Tensor],
        connectome: ConnectomeDir,
        keepref=False,
    ):
        super().__init__(keepref)
        self.index = nodes_edges_utils.NodeIndexer(connectome)

        unique_cell_types = connectome.unique_cell_types[:]
        input_cell_types = connectome.input_cell_types[:]
        output_cell_types = connectome.output_cell_types[:]
        self.input_indices = np.array([
            np.nonzero(unique_cell_types == t)[0] for t in input_cell_types
        ])
        self.output_indices = np.array([
            np.nonzero(unique_cell_types == t)[0] for t in output_cell_types
        ])
        self.activity = activity
        self.unique_cell_types = unique_cell_types.astype(str)

    def __getattr__(self, key):
        activity = self.activity() if not self.keepref else self.activity
        if activity is None:
            return
        if isinstance(key, list):
            index = np.stack(list(map(lambda key: self.index[key], key)))
            slices = self._slices(len(activity.shape) - 1)
            slices += (index,)
            return activity[slices]
        elif key == slice(None):
            return activity
        elif key in self.index.unique_cell_types:
            slices = self._slices(len(activity.shape) - 1)
            slices += (self.index[key],)
            return activity[slices]
        elif key == "output":
            slices = self._slices(len(activity.shape) - 1)
            slices += (self.output_indices,)
            return activity[slices]
        elif key == "input":
            slices = self._slices(len(activity.shape) - 1)
            slices += (self.input_indices,)
            return activity[slices]
        elif "+" in key:
            _cell_types = key.split("+")
            return sum(map(self.__getattr__, _cell_types))
        elif "-" in key:
            _cell_types = key.split("-")
            return reduce(operator.sub, map(self.__getattr__, _cell_types))
        elif "*" in key:
            _cell_types = key.split("*")
            return reduce(operator.mul, map(self.__getattr__, _cell_types))
        elif "/" in key:
            _cell_types = key.split("/")
            return reduce(operator.truediv, map(self.__getattr__, _cell_types))
        elif key in self.__dict__:
            return self.__dict__[key]
        else:
            raise ValueError(f"{key}")

    def __setattr__(self, key, value):
        if key == "activity" and value is not None:
            if len(self.index.unique_cell_types) != value.shape[-1]:
                slices = self._slices(len(value.shape) - 1)
                slices += (self.index.central_cells_index,)
                value = value[slices]
                self.keepref = True
            if self.keepref is False:
                value = weakref.ref(value)
            object.__setattr__(self, key, value)
        else:
            object.__setattr__(self, key, value)

    def __len__(self):
        return len(self.unique_cell_types)

    def __iter__(self):
        for cell_type in self.unique_cell_types:
            yield cell_type


class LayerActivity(CellTypeActivity):
    """Attribute-style access to layer activity.

    Args:
        activity (array-like): activity of shape (..., n_cells)
        connectome (Folder): connectome dir with reference to
                        - connectome.nodes.layer_index
                        - connectome.unique_cell_types
                        - connectome.central_cells_index
                        - connectome.input_cell_types
                        - connectome.output_cell_types
        keepref (bool, optional): Whether to keep a reference to the activity.
            Defaults to False.

    Note, activity is stored as a weakref by default. This is for memory efficienty
    during training. If you want to keep a reference to the activity for analysis,
    set keepref=True.

    Attributes:
        central (CentralActivity): central activity mapping,
            giving attribute-style access to central nodes of particular types.
        activity (array-like): activity of shape (..., n_cells)
        connectome (Folder): connectome dir with reference to
                        - connectome.nodes.layer_index
                        - connectome.unique_cell_types
                        - connectome.central_cells_index
                        - connectome.input_cell_types
                        - connectome.output_cell_types
        unique_cell_types (array)
        input_indices (array)
        output_indices (array)

    Note: central activity can be accessed by
    >>> a = LayerActivity(activity, network.connectome)
    >>> central_T4a = a.central.T4a

    Note: also allows 'virtual types' that are the sum of individuals
    >>> a = LayerActivity(activity, network.connectome)
    >>> summed_a = a['L2+L4']
    """

    central = {}
    activity = None
    connectome = None
    unique_cell_types = []
    input_cell_types = []
    output_cell_types = []

    def __init__(self, activity, connectome, keepref=False, use_central=True):
        super().__init__(keepref)
        self.keepref = keepref

        self.use_central = use_central
        if use_central:
            self.central = CentralActivity(activity, connectome, keepref)

        self.activity = activity
        self.connectome = connectome
        self.unique_cell_types = connectome.unique_cell_types[:].astype("str")
        for cell_type in self.unique_cell_types:
            index = connectome.nodes.layer_index[cell_type][:]
            self[cell_type] = index

        _cell_types = self.connectome.nodes.type[:]
        self.input_indices = np.array([
            np.nonzero(_cell_types == t)[0] for t in self.connectome.input_cell_types
        ])
        self.output_indices = np.array([
            np.nonzero(_cell_types == t)[0] for t in self.connectome.output_cell_types
        ])
        self.input_cell_types = self.connectome.input_cell_types[:].astype(str)
        self.output_cell_types = self.connectome.output_cell_types[:].astype(str)
        self.n_nodes = len(self.connectome.nodes.type)

    def __setattr__(self, key, value):
        if key == "activity" and value is not None:
            if self.keepref is False:
                value = weakref.ref(value)

            if self.use_central:
                self.central.__setattr__(key, value)

            object.__setattr__(self, key, value)
        else:
            object.__setattr__(self, key, value)


# class StimulusResponseIndexer:
#     """Indexer for stimulus response data.

#     Args:
#         arg_df (pd.DataFrame): DataFrame with stimulus arguments.
#         responses (CellTypeArray): Array with responses. TODO: might be easier to pass
#             arrays/tensors and the connectome instead of explicitly passing the
#             CellTypeArray. Also, will likely need to be adapted to StateInterface later.
#         dt (float): Time step.
#         t_pre (float): Time before stimulus onset.
#         stim_sample_dim (int, optional): Dimension of stimulus samples. Defaults to 0.
#         temporal_dim (int, optional): Dimension of time. Defaults to 1.
#         time (array-like, optional): Time array. Defaults to None.

#     NOTE: This class is used to index stimulus response data. It allows for easy
#         manipulation of the data, such as masking, reshaping, and normalizing.

#     NOTE: The class is designed to be used in a functional way, i.e. the methods
#         return a new instance of the class with the modified data. This is to avoid
#         modifying the data in place and to allow for easy chaining of operations.
#     """

#     def __init__(
#         self,
#         arg_df: pd.DataFrame,
#         responses: CellTypeArray,
#         dt: float,
#         t_pre: float,
#         stim_sample_dim=0,
#         temporal_dim=1,
#         time=None,
#     ):
#         self.arg_df = arg_df
#         self.responses = responses
#         if responses is not None:
#             assert type(self.responses).__name__ == "CellTypeArray"
#         self.stim_sample_dim = stim_sample_dim
#         self.temporal_dim = temporal_dim
#         self.dt = dt
#         self.t_pre = t_pre
#         self.time = None
#         self.init_time(time)

#     def init_time(self, time=None) -> None:
#         if time is not None:
#             self.time = time
#             return
#         if self.responses:
#             self.time = (
#                 self.time
#                 if np.any(self.time)
#                 and len(self.time) == self.responses.shape[self.temporal_dim]
#                 else (
#                     np.arange(0, self.responses.shape[self.temporal_dim]) * self.dt
#                     - self.t_pre
#                 )
#             )

#     def view(
#         self,
#         arg_df=None,
#         responses: Union[np.ndarray, CellTypeArray] = None,
#         dt=None,
#         t_pre=None,
#         stim_sample_dim=None,
#         temporal_dim=None,
#         time=None,
#     ) -> "StimulusResponseIndexer":
#         if isinstance(responses, np.ndarray):
#             responses = CellTypeArray(responses, cell_types=self.responses.cell_types)

#         return self.__class__(
#             arg_df if np.any(arg_df) else self.arg_df,
#             responses if responses else self.responses,
#             dt or self.dt,
#             t_pre or self.t_pre,
#             stim_sample_dim if np.any(stim_sample_dim) else self.stim_sample_dim,
#             temporal_dim or self.temporal_dim,
#             time if np.any(time) else self.time,
#         )

#     def masked(self, mask, mask_dims) -> "StimulusResponseIndexer":
#         responses = self.responses[:]
#         shape = responses.shape
#         dims = np.arange(len(shape))
#         new_responses = np.ma.masked_array(
#             responses,
#             ~np.tile(
#                 np.expand_dims(mask, tuple(set(dims) - set(mask_dims))),
#                 (shape[d] if d not in mask_dims else 1 for d in dims),
#             ),
#         )
#         if self.temporal_dim in mask_dims:
#             if len(mask_dims) > 1:
#                 new_time = np.expand_dims(
#                     self.time, tuple(set(dims) - {self.temporal_dim})
#                 )
#                 new_time = np.tile(
#                     new_time,
#                     (
#                         (
#                             shape[d]
#                             if d in tuple(set(mask_dims) - {self.temporal_dim})
#                             else 1
#                         )
#                         for d in dims
#                     ),
#                 )
#                 new_time = np.ma.masked_array(
#                     new_time,
#                     ~np.expand_dims(mask, tuple(set(dims) - set(mask_dims))),
#                 )
#             else:
#                 new_time = np.ma.masked_array(self.time, ~mask)
#         return self.view(responses=new_responses, time=new_time)

#     def _prepare_array(self, array: np.ndarray, dims: tuple):
#         assert len(array.shape) == len(dims)
#         shape = self.shape
#         all_dims = np.arange(len(shape))
#         # resolve any negative indexing logic
#         dims = all_dims[np.array(dims)]
#         # expand the array to the dimensions of self
#         return np.expand_dims(array, tuple(set(all_dims) - set(dims)))

#     def divide_by_given_array(self, array: np.ndarray, dims: tuple):
#         """Divide the given dims of self by the given array.

#         E.g. to divide by a protocol normalization constant for stimulus
#         responses per cell hypothesis
#         (i.e. model * cell_type independent values).

#         Args:
#             array: array with elements to divide with.
#             dims: dims in self to divide.
#         """
#         return self.view(responses=self[:] / self._prepare_array(array, dims))

#     def subtract_by_given_array(self, array: np.ndarray, dims: tuple):
#         """Divide the given dims of self by the given array.

#         E.g. to divide by a protocol normalization constant for stimulus
#         responses per cell hypothesis
#         (i.e. model * cell_type independent values).

#         Args:
#             array: array with elements to divide with.
#             dims: dims in self to divide.
#         """
#         return self.view(responses=self[:] - self._prepare_array(array, dims))

#     def cell_type(self, cell_type) -> "StimulusResponseIndexer":
#         if isinstance(cell_type, str):
#             cell_type = [cell_type]
#         new_responses = CellTypeArray(
#             self.responses[cell_type],
#             cell_types=cell_type,
#         )
#         return self.view(responses=new_responses)

#     @property
#     def cell_types(self):
#         return self.responses.cell_types

#     def __truediv__(self, other):
#         return self.view(responses=self[:] / other[:])

#     def __mul__(self, other):
#         return self.view(responses=self[:] * other[:])

#     def __add__(self, other):
#         return self.view(responses=self[:] + other[:])

#     def __sub__(self, other):
#         return self.view(responses=self[:] - other[:])

#     @staticmethod
#     def where_stim_args_index_static(arg_df, **kwargs):
#         return where_dataframe(arg_df, **kwargs)

#     def where_stim_args_index(self, **kwargs):
#         return where_dataframe(self.arg_df, **kwargs)

#     def where_stim_args(self, **kwargs) -> "StimulusResponseIndexer":
#         arg_index = self.where_stim_args_index(**kwargs)

#         # TODO: case when we have already reshaped because we cannot simply index
#         # along the stim_sample_dim --> might require to keep a ref to the original
#         # indexer
#         if isinstance(self.stim_sample_dim, Iterable) and len(self.stim_sample_dim) > 1:
#             # from typing import Iterable
#             # stim_sample_dim = x.stim_sample_dim

#             # if not isinstance(x.stim_sample_dim, Iterable):
#             #     stim_sample_dim = [x.stim_sample_dim]

#             # # assert consecutive dimensions
#             # assert (np.diff(stim_sample_dim) == 1).all()

#             # shape = x.shape
#             # stim_sample_dim_shape = 1
#             # new_shape = []
#             # for i, _shape in enumerate(shape):
#             #     if i in stim_sample_dim:
#             #         stim_sample_dim_shape = np.prod((stim_sample_dim_shape, _shape))
#             #         if i == stim_sample_dim[-1]:
#             #             new_shape.append(stim_sample_dim_shape)
#             #     else:
#             #         new_shape.append(_shape)
#             raise NotImplementedError

#         new_responses = np.take(self.responses[:], arg_index, self.stim_sample_dim)
#         new_arg_df = self.arg_df.iloc[arg_index.values]
#         new_arg_df.reset_index(drop=True, inplace=True)
#         return self.view(responses=new_responses, arg_df=new_arg_df)

#     def where_stim_index(self, stim_index) -> "StimulusResponseIndexer":
#         new_responses = np.take(self.responses[:], stim_index, self.stim_sample_dim)
#         new_arg_df = self.arg_df.iloc[stim_index]
#         new_arg_df.reset_index(drop=True, inplace=True)
#         return self.view(responses=new_responses, arg_df=new_arg_df)

#     def between_seconds(self, t_start, t_end, masked=False) -> "StimulusResponseIndexer":
#         mask = (self.time >= t_start) & (self.time <= t_end)
#         if masked:
#             return self.masked(~mask, (self.temporal_dim,))
#         slice = np.where(mask)[0]
#         new_responses = np.take(self.responses[:], slice, self.temporal_dim)
#         new_time = self.time[slice]
#         return self.view(responses=new_responses, time=new_time)

#     def nonnegative(self, dims=None) -> "StimulusResponseIndexer":
#         if dims is None:
#             dims = (self.temporal_dim, self.stim_sample_dim)

#         offset = np.abs(
#             np.min(
#                 self.responses[:],
#                 axis=dims,
#                 keepdims=True,
#             )
#         )

#         new_responses = self.responses[:] + offset
#         return self.view(responses=new_responses)

#     def rectify(self) -> "StimulusResponseIndexer":
#         new_responses = np.maximum(self.responses[:], 0)
#         return self.view(responses=new_responses)

#     def peak(self, dim=None) -> "StimulusResponseIndexer":
#         time = None
#         if dim is None or dim == self.temporal_dim:
#             dim = self.temporal_dim
#             time = np.nanargmax(self.responses[:], axis=dim, keepdims=True)
#             # TODO?: do this for every operation that collapses dims

#         peak = np.nanmax(self.responses[:], axis=dim, keepdims=True)
#         return self.view(responses=peak, time=time)

#     def baseline(self, dim=None) -> "StimulusResponseIndexer":
#         if dim is None:
#             dim = self.temporal_dim
#         baseline_index = np.argmin(np.abs(self.time) - 0)
#         baseline = np.take(self.responses[:], [baseline_index], dim)
#         return self.view(responses=baseline)

#     def subtract_baseline(self, dim=None) -> "StimulusResponseIndexer":
#         if dim is None:
#             dim = self.temporal_dim
#         baseline_index = np.argmin(np.abs(self.time) - 0)
#         baseline = np.take(self.responses[:], [baseline_index], dim)
#         new_responses = self.responses[:] - baseline
#         return self.view(responses=new_responses)

#     def subtract_mean(self, dims=None) -> "StimulusResponseIndexer":
#         if dims is None:
#             dims = self.temporal_dim
#         means = np.nanmean(self.responses[:], axis=dims, keepdims=True)
#         new_responses = self.responses[:] - means

#         return self.view(responses=new_responses)

#     def divide_by_std(self, dims=None) -> "StimulusResponseIndexer":
#         if dims is None:
#             dims = self.temporal_dim
#         stds = np.nanstd(self.responses[:], axis=dims, keepdims=True)
#         # avoid dividing by zero
#         stds[stds == 0] = 1
#         new_responses = self.responses[:] / stds

#         return self.view(responses=new_responses)

#     def divide_by_norm(self, dims=None) -> "StimulusResponseIndexer":
#         if dims is None:
#             dims = self.temporal_dim
#         stds = np.nanstd(self.responses[:], axis=dims, keepdims=True)
#         # avoid dividing by zero
#         stds[stds == 0] = 1
#         new_responses = self.responses[:] / stds

#         return self.view(responses=new_responses)

#     def divide_by_mean(self, dims=None) -> "StimulusResponseIndexer":
#         if dims is None:
#             dims = self.temporal_dim
#         means = np.nanmean(self.responses[:], axis=dims, keepdims=True)
#         # avoid dividing by zero
#         means[means == 0] = 1
#         new_responses = self.responses[:] / means

#         return self.view(responses=new_responses)

#     def divide_by_percentile(self, q, dims) -> "StimulusResponseIndexer":
#         responses = self.responses[:]
#         if isinstance(responses, np.ma.masked_array):
#             # create copy
#             responses = responses[:]
#             # fill with nans to use nanpercentile cause percentile does not
#             # support masked arrays
#             responses.data[responses.mask] = np.nan
#             percentile = np.nanpercentile(responses.data, q, axis=dims, keepdims=True)
#         else:
#             percentile = np.nanpercentile(responses, q, axis=dims, keepdims=True)
#         # avoid dividing by zero
#         percentile[percentile == 0] = 1
#         new_responses = self.responses[:] / percentile

#         return self.view(responses=new_responses)

#     def standardize(self, dims=None) -> "StimulusResponseIndexer":
#         if dims is None:
#             dims = self.temporal_dim
#         means = np.nanmean(self.responses[:], axis=dims, keepdims=True)
#         stds = np.nanstd(self.responses[:], axis=dims, keepdims=True)
#         # avoid dividing by zero
#         stds[stds == 0] = 1
#         new_responses = (self.responses[:] - means) / stds

#         return self.view(responses=new_responses)

#     def sum(self, dims) -> "StimulusResponseIndexer":
#         sums = np.nansum(self.responses[:], axis=dims, keepdims=True)
#         return self.view(responses=sums)

#     def abs(self) -> "StimulusResponseIndexer":
#         new_responses = np.abs(self.responses[:])
#         return self.view(responses=new_responses)

#     def minmax_scale(self, dims) -> "StimulusResponseIndexer":
#         new_responses = self.responses[:]
#         r_min = np.nanmin(new_responses, axis=dims, keepdims=True)
#         r_max = np.nanmax(new_responses, axis=dims, keepdims=True)
#         # treat zero-cases where r_max == r_min
#         diff = r_max - r_min
#         diff[diff == 0] = 1
#         new_responses = (new_responses - r_min) / diff
#         return self.view(responses=new_responses)

#     def squeeze(self, dims=None) -> "StimulusResponseIndexer":
#         return self.view(responses=np.squeeze(self.responses[:], axis=dims))

#     def mean(self, dims=None, keepdims=False) -> "StimulusResponseIndexer":
#         return self.view(
#             responses=np.nanmean(self.responses[:], axis=dims, keepdims=keepdims)
#         )

#     def average(self, dims, weights=None) -> "StimulusResponseIndexer":
#         all_dims = np.arange(len(self.responses[:].shape))
#         if weights is None:
#             return self.mean(dims, keepdims=True)
#         print("comp weighted average")
#         weights = np.expand_dims(weights, list(set(all_dims) - set(dims)))
#         new_responses = np.nansum(
#             np.multiply(self.responses[:], weights), axis=dims, keepdims=True
#         ) / np.nansum(weights, keepdims=True, axis=dims)
#         return self.view(responses=new_responses)

#     def min(self, dims=None, keepdims=False) -> "StimulusResponseIndexer":
#         return self.view(
#             responses=np.nanmin(self.responses[:], axis=dims, keepdims=keepdims)
#         )

#     def max(self, dims=None, keepdims=False) -> "StimulusResponseIndexer":
#         return self.view(
#             responses=np.nanmax(self.responses[:], axis=dims, keepdims=keepdims)
#         )

#     def median(self, dims=None, keepdims=False) -> "StimulusResponseIndexer":
#         return self.view(
#             responses=np.nanmedian(self.responses[:], axis=dims, keepdims=keepdims)
#         )

#     def quantile(self, q, dims=None) -> "StimulusResponseIndexer":
#         return self.view(responses=np.nanquantile(self.responses[:], q, axis=dims))

#     @property
#     def shape(self):
#         return self.responses[:].shape

#     @property
#     def ndim(self):
#         return self.responses[:].ndim

#     def reshape(self, *args, inplace=False) -> "StimulusResponseIndexer":
#         if inplace:
#             self.responses = CellTypeArray(
#                 self.responses[:].reshape(*args), cell_types=self.responses.cell_types
#             )
#             return self
#         return self.view(responses=self.responses[:].reshape(*args))

#     def filled(self, fill_value) -> "StimulusResponseIndexer":
#         if isinstance(self.responses[:], np.ma.masked_array):
#             return self.view(
#                 responses=self.responses[:].filled(fill_value),
#                 time=self.time.data,
#             )
#         return self

#     def reshape_stim_sample_dim(self, *args) -> "StimulusResponseIndexer":
#         """Reshape the originally flat stimulus sample dimension
#         to unflattened based on arg_df column names.
#         """
#         shape = self.shape
#         stim_sample_dim_shape = tuple()
#         for column in args:
#             assert column in self.arg_df.columns
#             stim_sample_dim_shape += (len(self.arg_df[column].unique()),)
#         if np.prod(stim_sample_dim_shape) == shape[self.stim_sample_dim]:
#             new_shape = (
#                 shape[: self.stim_sample_dim]
#                 + stim_sample_dim_shape
#                 + shape[self.stim_sample_dim + 1 :]
#             )
#             new_stim_sample_dim = np.arange(len(stim_sample_dim_shape)) + len(
#                 shape[: self.stim_sample_dim]
#             )
#         else:
#             new_shape = (
#                 shape[: self.stim_sample_dim]
#                 + stim_sample_dim_shape
#                 + (-1,)
#                 + shape[self.stim_sample_dim + 1 :]
#             )
#             new_stim_sample_dim = len(shape[: self.stim_sample_dim]) + len(
#                 stim_sample_dim_shape
#             )
#         new_temporal_dim = (
#             self.temporal_dim + len(stim_sample_dim_shape) - 1
#             if self.temporal_dim > self.stim_sample_dim
#             else self.temporal_dim
#         )
#         return self.view(
#             responses=self.responses[:].reshape(*new_shape),
#             stim_sample_dim=new_stim_sample_dim,
#             temporal_dim=new_temporal_dim,
#         )

#     def take_single(self, indices, dims) -> "StimulusResponseIndexer":
#         """Select single indices from multiple dims."""
#         if not isinstance(indices, Iterable):
#             indices = [indices]
#         if not isinstance(dims, Iterable):
#             dims = [dims]
#         new_responses = self.responses[:]
#         for index, dim in zip(indices, dims):
#             assert dim != self.temporal_dim
#             new_responses = np.take(new_responses, [index], axis=dim)
#         return self.view(responses=new_responses)

#     def take(self, index, dim) -> "StimulusResponseIndexer":
#         if not isinstance(index, Iterable):
#             index = np.array([index])
#         assert dim not in self.stim_sample_dim
#         new_responses = np.take(self.responses[:], index, axis=dim)
#         return self.view(responses=new_responses)

#     def transpose(self, *args, inplace=False) -> "StimulusResponseIndexer":
#         if isinstance(self.stim_sample_dim, Iterable):
#             # to transpose also stimulus sample dimensions tracked
#             new_stim_sample_dim = np.array([
#                 np.array(args).tolist().index(i) for i in self.stim_sample_dim
#             ])
#         if inplace:
#             self.responses = CellTypeArray(
#                 self.responses[:].transpose(*args),
#                 cell_types=self.responses.cell_types,
#             )
#             self.stim_sample_dim = new_stim_sample_dim
#             return self
#         return self.view(
#             responses=self.responses[:].transpose(*args),
#             stim_sample_dim=new_stim_sample_dim,
#         )

#     def __getitem__(self, key) -> Union[np.ndarray, "StimulusResponseIndexer"]:
#         if (
#             isinstance(key, str)
#             and key in self.responses.cell_types
#             or (
#                 isinstance(key, Iterable)
#                 and all([isinstance(k, str) for k in key])
#                 and all([k in self.responses.cell_types for k in key])
#             )
#         ):
#             return self.cell_type(key)
#         if isinstance(key, slice) and key == slice(None, None, None):
#             return self.responses[:]  # .squeeze()
#         elif (
#             isinstance(key, Iterable)
#             and len(key) <= len(self.responses[:].shape)
#             or isinstance(key, np.ndarray)
#             and len(key.shape) == 1
#         ):
#             return self.view(responses=self.responses[:][key])

#         return object.__getattribute__(self, key)

#     def __repr__(self):
#         return (
#             f"{self.__class__.__name__}({self.arg_df}, {self.responses}, {self.dt}, "
#             f"{self.stim_sample_dim}, {self.temporal_dim})"
#         )

#     def plot_traces(
#         self,
#         cell_type,
#         t_start=None,
#         t_end=None,
#         plot_kwargs=dict(),
#         groupby=None,
#         **stim_kwargs,
#     ):
#         if self.responses.shape[self.temporal_dim] != len(self.time) or np.all(
#             np.isnan(self.time)
#         ):
#             raise ValueError(
#                 "Cannot plot. Previous operations have mis-aligned the "
#                 "response data and timestamps."
#             )
#         cell_trace = self.cell_type(cell_type)
#         t_start = t_start or self.time[0] - 1
#         t_end = t_end or self.time[-1] + 1
#         cell_trace = cell_trace.between_seconds(t_start=t_start, t_end=t_end)
#         if stim_kwargs:
#             cell_trace = cell_trace.where_stim_args(**stim_kwargs)
#         if groupby is None:
#             groupby = [
#                 col
#                 for col in cell_trace.arg_df.columns
#                 if cell_trace.arg_df[col].nunique() > 1
#             ]
#         if len(groupby) > 0:
#             names = sorted([
#                 tuple(values)
#                 for values in cell_trace.arg_df[groupby].drop_duplicates().values.tolist()
#             ])
#             responses = [
#                 cell_trace.where_stim_args(**dict(zip(groupby, name)))[:]
#                 for name in names
#             ]
#             names = [
#                 ", ".join([f"{key}={val}" for key, val in zip(groupby, names)])
#                 for names in names
#             ]

#         else:
#             responses = [cell_trace.responses[:]]
#             names = []
#         responses = [
#             resp.swapaxes(self.temporal_dim, -1).reshape(
#                 -1, resp.shape[self.temporal_dim]
#             )
#             for resp in responses
#         ]
#         defaults = dict(
#             linewidth=1.0,
#             legend=tuple(names),
#             ylabel="activity (a.u.)",
#             xlabel="time (s)",
#             title=f"{cell_type} stimulus response",
#         )
#         defaults.update(plot_kwargs)
#         return grouped_traces(
#             responses,
#             cell_trace.time,
#             **defaults,
#         )


class SourceCurrentView:
    """Create views of source currents for a target type."""

    def __init__(self, rfs: ReceptiveFields, currents):
        self.target_type = rfs.target_type
        self.source_types = list(rfs)
        self.rfs = rfs
        self.currents = currents

    def __getattr__(self, key):
        if key in self.source_types:
            return np.take(self.currents, self.rfs[key].index, axis=-1)
        return object.__getattr__(self, key)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def update(self, currents):
        self.currents = currents


def asymmetric_weighting(tensor, gamma=1.0, delta=0.1):
    """
    Applies asymmetric weighting to the positive and negative elements of a tensor.

    The function is defined as:
    f(x) = gamma * x if x > 0 else delta * x
    """
    return gamma * nn.functional.relu(tensor) - delta * nn.functional.relu(-tensor)
