"""Methods to access and process network activity and currents based on cell types.

Example:
    Get the activity of all cells of a certain cell type
    (note, `LayerActivity` is not the network split into feedforward layers in the
    machine learning sense but the activity of all cells by cell type):
    ```python
    layer_activity = LayerActivity(activity, network.connectome)
    T4a_response = layer_activity.T4a
    T5a_response = layer_activity.T5a
    T4b_central_response = layer_activity.central.T4a
    ```
"""

import weakref
from textwrap import wrap
from typing import List, Union

import numpy as np
import torch
from numpy.typing import NDArray

from flyvis.connectome import ConnectomeFromAvgFilters, ReceptiveFields
from flyvis.utils import nodes_edges_utils

__all__ = [
    "CentralActivity",
    "LayerActivity",
    "SourceCurrentView",
]


class CellTypeActivity(dict):
    """Base class for attribute-style access to network activity based on cell types.

    Args:
        keepref: Whether to keep a reference to the activity. This may not be desired
            during training to avoid memory issues.

    Attributes:
        activity: Weak reference to the activity.
        keepref: Whether to keep a reference to the activity.
        unique_cell_types: List of unique cell types.
        input_indices: Indices of input cells.
        output_indices: Indices of output cells.

    Note:
        Activity is stored as a weakref by default for memory efficiency
        during training. Set keepref=True to keep a reference for analysis.
    """

    def __init__(self, keepref: bool = False):
        self.keepref = keepref
        self.activity: Union[weakref.ref, NDArray, torch.Tensor] = None
        self.unique_cell_types: List[str] = []
        self.input_indices: NDArray = np.array([])
        self.output_indices: NDArray = np.array([])

    def __dir__(self) -> List[str]:
        return list(set([*dict.__dir__(self), *dict.__iter__(self)]))

    def __len__(self) -> int:
        return len(self.unique_cell_types)

    def __iter__(self):
        yield from self.unique_cell_types

    def __repr__(self) -> str:
        return "Activity of: \n{}".format("\n".join(wrap(", ".join(list(self)))))

    def update(self, activity: Union[NDArray, torch.Tensor]) -> None:
        """Update the activity reference."""
        self.activity = activity

    def _slices(self, n: int) -> tuple:
        return tuple(slice(None) for _ in range(n))

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


class CentralActivity(CellTypeActivity):
    """Attribute-style access to central cell activity of a cell type.

    Args:
        activity: Activity of shape (..., n_cells).
        connectome: Connectome directory with reference to required attributes.
        keepref: Whether to keep a reference to the activity.

    Attributes:
        activity: Activity of shape (..., n_cells).
        unique_cell_types: Array of unique cell types.
        index: NodeIndexer instance.
        input_indices: Array of input indices.
        output_indices: Array of output indices.
    """

    def __init__(
        self,
        activity: Union[NDArray, torch.Tensor],
        connectome: ConnectomeFromAvgFilters,
        keepref: bool = False,
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
    """Attribute-style access to hex-lattice activity (cell-type specific).

    Args:
        activity: Activity of shape (..., n_cells).
        connectome: Connectome directory with reference to required attributes.
        keepref: Whether to keep a reference to the activity.
        use_central: Whether to use central activity.

    Attributes:
        central: CentralActivity instance for central nodes.
        activity: Activity of shape (..., n_cells).
        connectome: Connectome directory.
        unique_cell_types: Array of unique cell types.
        input_indices: Array of input indices.
        output_indices: Array of output indices.
        input_cell_types: Array of input cell types.
        output_cell_types: Array of output cell types.
        n_nodes: Number of nodes.

    Note:
        The name `LayerActivity` might change in future as it is misleading.
        This is not a feedforward layer in the machine learning sense but the
        activity of all cells of a certain cell-type.

    Example:

        Central activity can be accessed by:
        ```python
        a = LayerActivity(activity, network.connectome)
        central_T4a = a.central.T4a
        ```

        Also allows 'virtual types' that are the sum of individuals:
        ```python
        a = LayerActivity(activity, network.connectome)
        summed_a = a['L2+L4']
        ```
    """

    def __init__(
        self,
        activity: Union[NDArray, torch.Tensor],
        connectome: ConnectomeFromAvgFilters,
        keepref: bool = False,
        use_central: bool = True,
    ):
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


class SourceCurrentView:
    """Create views of source currents for a target type.

    Args:
        rfs: ReceptiveFields instance.
        currents: Current values.

    Attributes:
        target_type: Target cell type.
        source_types: List of source cell types.
        rfs: ReceptiveFields instance.
        currents: Current values.
    """

    def __init__(self, rfs: ReceptiveFields, currents: Union[NDArray, torch.Tensor]):
        self.target_type = rfs.target_type
        self.source_types = list(rfs)
        self.rfs = rfs
        self.currents = currents

    def __getattr__(self, key: str) -> Union[NDArray, torch.Tensor]:
        if key in self.source_types:
            return np.take(self.currents, self.rfs[key].index, axis=-1)
        return object.__getattr__(self, key)

    def __getitem__(self, key: str) -> Union[NDArray, torch.Tensor]:
        return self.__getattr__(key)

    def update(self, currents: Union[NDArray, torch.Tensor]) -> None:
        """Update the currents."""
        self.currents = currents
