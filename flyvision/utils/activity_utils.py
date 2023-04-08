"""Convenience and efficient access to activity of particular cells.

Example:
    layer_activity = LayerActivity(activity, network.connectome)
    T4a_response = layer_activity.T4a
    T5a_response = layer_activity.T5a
    T4b_central_response = layer_activity.central.T4a
"""
from textwrap import wrap
from functools import reduce
import operator
import weakref
from typing import Union

import numpy as np
from numpy.typing import NDArray

import torch

from flyvision.utils import nodes_edges_utils
from flyvision.connectome import ConnectomeDir

__all__ = ["CentralActivity", "LayerActivity"]


class CellTypeActivity(dict):
    """Base class for attribute-style access to network activity.

    Note, activity is stored as a weakref by default. This is for memory efficienty
    during training. If you want to keep a reference to the activity for analysis,
    set keepref=True.

    Args:
        keepref (bool, optional): Whether to keep a reference to the activity. Defaults to False.

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
        for i in range(n):
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
        elif key in self.__dict__.keys():
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
        keepref (bool, optional): Whether to keep a reference to the activity. Defaults to False.

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
        self.input_indices = np.array(
            [np.nonzero(unique_cell_types == t)[0] for t in input_cell_types]
        )
        self.output_indices = np.array(
            [np.nonzero(unique_cell_types == t)[0] for t in output_cell_types]
        )
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
        elif key in self.__dict__.keys():
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
        keepref (bool, optional): Whether to keep a reference to the activity. Defaults to False.

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
        self.input_indices = np.array(
            [np.nonzero(_cell_types == t)[0] for t in self.connectome.input_cell_types]
        )
        self.output_indices = np.array(
            [np.nonzero(_cell_types == t)[0] for t in self.connectome.output_cell_types]
        )
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
