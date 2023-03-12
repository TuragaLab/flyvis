"""Attribute-style access to activity of particular node types.

Example:
    layer_activity = LayerActivity(activity, network.ctome)
    T4a = layer_activity.T4a
    T5a = layer_activity.T5a
    T4b_central = layer_activity.central.T4a
"""
from textwrap import wrap
from functools import reduce
import operator
import weakref

import numpy as np

from flyvision.utils import nodes_edges_utils


class _Activity(dict):
    def __init__(self, keepref=False):
        self.keepref = keepref

    def __dir__(self):
        return list(set([*dict.__dir__(self), *dict.__iter__(self)]))

    def __len__(self):
        return len(self.unique_node_types)

    def __iter__(self):
        for node_type in self.unique_node_types:
            yield node_type

    def __repr__(self):
        return "Activity of: \n{}".format(
            "\n".join(wrap(", ".join(list(self))))
        )

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
            index = np.stack(
                list(map(lambda key: dict.__getitem__(self, key), key))
            )
            slices = self._slices(len(activity.shape) - 1)
            slices += (index,)
            return activity[slices]
        elif key == slice(None):
            return activity
        elif key in self.unique_node_types:
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
            _node_types = key.split("+")
            return sum(map(self.__getattr__, _node_types))
        elif "-" in key:
            _node_types = key.split("-")
            return reduce(operator.sub, map(self.__getattr__, _node_types))
        elif "*" in key:
            _node_types = key.split("*")
            return reduce(operator.mul, map(self.__getattr__, _node_types))
        elif "/" in key:
            _node_types = key.split("/")
            return reduce(operator.truediv, map(self.__getattr__, _node_types))
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


class CentralActivity(_Activity):
    """Attribute-style access to central activity.

    Args:
        activity (array-like): activity of shape (..., #nodes)
        ctome (Folder): connectome wrap with reference to
                        - ctome.nodes.layer_index
                        - ctome.unique_node_types
                        - ctome.central_nodes_index


    Attributes:
        activity (array-like): activity of shape (..., #nodes)
        unique_node_types (array)

    Note: also allows 'virtual types' that are basic operations of individuals
    >>> a = LayerActivity(activity, network.ctome)
    >>> summed_a = a['L2+L4*L3/L5']
    """

    def __init__(self, activity, ctome, keepref=False):
        super().__init__(keepref)
        self.index = nodes_edges_utils.NodeIndexer(ctome)

        unique_node_types = ctome.unique_node_types[:]
        input_node_types = ctome.input_node_types[:]
        output_node_types = ctome.output_node_types[:]
        self.input_indices = np.array(
            [np.nonzero(unique_node_types == t)[0] for t in input_node_types]
        )
        self.output_indices = np.array(
            [np.nonzero(unique_node_types == t)[0] for t in output_node_types]
        )
        # breakpoint()
        self.activity = activity
        self.unique_node_types = unique_node_types.astype(str)

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
        elif key in self.index.unique_node_types:
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
            _node_types = key.split("+")
            return sum(map(self.__getattr__, _node_types))
        elif "-" in key:
            _node_types = key.split("-")
            return reduce(operator.sub, map(self.__getattr__, _node_types))
        elif "*" in key:
            _node_types = key.split("*")
            return reduce(operator.mul, map(self.__getattr__, _node_types))
        elif "/" in key:
            _node_types = key.split("/")
            return reduce(operator.truediv, map(self.__getattr__, _node_types))
        elif key in self.__dict__.keys():
            return self.__dict__[key]
        else:
            raise ValueError(f"{key}")

    def __setattr__(self, key, value):
        # TODO: case when value is ReferenceType and whole layers.
        if key == "activity" and value is not None:
            if len(self.index.unique_node_types) != value.shape[-1]:
                slices = self._slices(len(value.shape) - 1)
                slices += (self.index.central_nodes_index,)
                value = value[slices]
                self.keepref = True
            if self.keepref is False:
                value = weakref.ref(value)
            object.__setattr__(self, key, value)
        else:
            object.__setattr__(self, key, value)

    def __len__(self):
        return len(self.unique_node_types)

    def __iter__(self):
        for node_type in self.unique_node_types:
            yield node_type


class LayerActivity(_Activity):
    """Attribute-style access to layer activity.

    Args:
        activity (array-like): activity of shape (..., #nodes)
        ctome (Folder): connectome wrap with reference to
                        - ctome.nodes.layer_index
                        - ctome.unique_node_types
                        - ctome.central_nodes_index
                        - ctome.input_node_types
                        - ctome.output_node_types

    Attributes:
        central (CentralActivity): central activity mapping,
            giving attribute-style access to central nodes of particular types.
        activity (array-like): activity of shape (..., #nodes)
        ctome (Folder): connectome wrap with reference to
                        - ctome.nodes.layer_index
                        - ctome.unique_node_types
                        - ctome.central_nodes_index
                        - ctome.input_node_types
                        - ctome.output_node_types
        unique_node_types (array)
        input_indices (array)
        output_indices (array)
        input (array)
        output (array)
        <node_types> (array)


    Note: central activity can be accessed by
    >>> a = LayerActivity(activity, network.ctome)
    >>> central_T4a = a.central.T4a

    Note: also allows 'virtual types' that are the sum of individuals
    >>> a = LayerActivity(activity, network.ctome)
    >>> summed_a = a['L2+L4']
    """

    central = {}
    activity = None
    ctome = None
    unique_node_types = []
    input_node_types = []
    output_node_types = []

    def __init__(self, activity, ctome, keepref=False, use_central=True):
        super().__init__(keepref)
        self.keepref = keepref

        self.use_central = use_central
        if use_central:
            self.central = CentralActivity(activity, ctome, keepref)

        self.activity = activity
        self.ctome = ctome
        self.unique_node_types = ctome.unique_node_types[:].astype("str")
        for node_type in self.unique_node_types:
            index = ctome.nodes.layer_index[node_type][:]
            self[node_type] = index

        _node_types = self.ctome.nodes.type[:]
        self.input_indices = np.array(
            [
                np.nonzero(_node_types == t)[0]
                for t in self.ctome.input_node_types
            ]
        )
        self.output_indices = np.array(
            [
                np.nonzero(_node_types == t)[0]
                for t in self.ctome.output_node_types
            ]
        )
        self.input_node_types = self.ctome.input_node_types[:].astype(str)
        self.output_node_types = self.ctome.output_node_types[:].astype(str)
        self.n_nodes = len(self.ctome.nodes.type)

    def __setattr__(self, key, value):
        if key == "activity" and value is not None:

            if self.keepref is False:
                value = weakref.ref(value)

            if self.use_central:
                self.central.__setattr__(key, value)

            object.__setattr__(self, key, value)
        else:
            object.__setattr__(self, key, value)
