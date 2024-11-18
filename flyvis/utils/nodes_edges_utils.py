import re
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray

from flyvis import connectome
from flyvis.utils import groundtruth_utils


def order_node_type_list(
    node_types: List[str],
    groups: List[str] = [
        r"R\d",
        r"L\d",
        r"Lawf\d",
        r"A",
        r"C\d",
        r"CT\d.*",
        r"Mi\d{1,2}",
        r"T\d{1,2}.*",
        r"Tm.*\d{1,2}.*",
    ],
) -> Tuple[List[str], List[int]]:
    """Orders a list of node types by the regular expressions defined in groups.

    Args:
        node_types: Messy list of nodes.
        groups: Ordered list of regular expressions to sort node_types.

    Returns:
        A tuple containing:
        - Ordered node type list
        - Corresponding sorting indices

    Raises:
        AssertionError: If sorting doesn't include all cell types.
        ValueError: If sorting fails due to length mismatch.
    """
    if node_types is None:
        return None, None

    _len = len(node_types)

    def sort_numeric(string):
        """Used in sorted(list, key=sort_fn) for sorting
        lists including 0 to 4 digits after the character.
        """
        regular_expression = r"\d+"
        match = re.search(regular_expression, string)
        if not match:
            # For example Am types are not numbered.
            return string
        return re.sub(regular_expression, f"{int(match.group()):04}", string)

    #     breakpoint()
    type_groups = {index: [] for index in range(len(groups))}
    type_groups.update({len(groups) + 1: []})  # for unmatched types.
    matched = {cell_type: False for cell_type in node_types}
    for node_index, cell_type in enumerate(node_types):
        for group_index, regular_expression in enumerate(groups):
            if re.match(regular_expression, cell_type):
                type_groups[group_index].append((node_index, cell_type))
                matched[cell_type] = True
        if matched[cell_type]:
            pass
        else:
            type_groups[len(groups) + 1].append((node_index, cell_type))

    # ordered = [y for x in type_groups.values() for y in sorted(x, key=lambda z:
    # sort_fn(z[1]))]
    ordered = []
    for x in type_groups.values():
        for y in sorted(x, key=lambda z: sort_numeric(z[1])):
            ordered.append(y)
    index = [y[0] for y in ordered]
    nodes = [y[1] for y in ordered]

    if set(node_types) - set(nodes):
        print(set(node_types) - set(nodes))
        raise AssertionError(
            "Defined sorting through regular expressions does not include all cell"
            " types."
        )

    if _len != len(nodes) or _len != len(index):
        raise ValueError(
            "sorting failed because the resulting array if of " " different length"
        )

    return nodes, index


def get_index_mapping_lists(from_list: List[str], to_list: List[str]) -> List[int]:
    """Get indices to sort and filter from_list by occurrence of items in to_list.

    The indices are useful to sort or filter another list or tensor that
    is an ordered mapping to items in from_list to the order of items in to_list.

    Args:
        from_list: Original list of items.
        to_list: Target list of items.

    Returns:
        List of indices for sorting.

    Example:
        ```python
        from_list = ["a", "b", "c"]
        mapping_to_from_list = [1, 2, 3]
        to_list = ["c", "a", "b"]
        sort_index = get_index_mapping_lists(from_list, to_list)
        sorted_list = [mapping_to_from_list[i] for i in sort_index]
        # sorted_list will be [3, 1, 2]
        ```
    """
    if isinstance(from_list, np.ndarray):
        from_list = from_list.tolist()
    if isinstance(to_list, np.ndarray):
        to_list = to_list.tolist()
    return [from_list.index(item) for item in to_list]


def sort_by_mapping_lists(
    from_list: List[str],
    to_list: List[str],
    tensor: Union[np.ndarray, torch.Tensor],
    axis: int = 0,
) -> np.ndarray:
    """Sort and filter a tensor along an axis indexed by from_list to match to_list.

    Args:
        from_list: Original list of items.
        to_list: Target list of items.
        tensor: Tensor to be sorted.
        axis: Axis along which to sort the tensor.

    Returns:
        Sorted numpy array.
    """
    tensor = np.array(tensor)
    if axis != 0:
        tensor = np.transpose(tensor, axes=(axis, 0))
    sort_index = get_index_mapping_lists(from_list, to_list)
    tensor = np.array([tensor[i] for i in sort_index])
    if axis != 0:
        tensor = np.transpose(tensor, axes=(axis, 0))
    return tensor


def nodes_list_sorting_on_off_unknown(
    cell_types: Optional[List[str]] = None,
) -> List[str]:
    """Sort node list based on on/off/unknown polarity.

    Args:
        cell_types: List of cell types to sort. If None, uses all types from
                    groundtruth_utils.polarity.

    Returns:
        Sorted list of cell types.
    """
    value = {1: 1, -1: 2, 0: 3}
    preferred_contrasts = groundtruth_utils.polarity
    cell_types = list(preferred_contrasts) if cell_types is None else cell_types
    preferred_contrasts = {
        k: value[v] for k, v in preferred_contrasts.items() if k in cell_types
    }
    preferred_contrasts = dict(sorted(preferred_contrasts.items(), key=lambda k: k[1]))
    nodes_list = list(preferred_contrasts.keys())
    return nodes_list


class NodeIndexer(dict):
    """Attribute-style accessible map from cell types to indices.

    Args:
        connectome: Connectome object. The cell types are taken from the
            connectome and references are created in order.
        unique_cell_types: Array of unique cell types. Optional.
            To specify the mapping from cell types to indices in provided order.

    Attributes:
        unique_cell_types (NDArray[str]): Array of unique cell types.
        central_cells_index (Optional[NDArray[int]]): Array of indices of central cells.

    Raises:
        ValueError: If neither connectome nor unique_cell_types is provided.
    """

    def __init__(
        self,
        connectome: Optional["connectome.ConnectomeFromAvgFilters"] = None,
        unique_cell_types: Optional[NDArray[str]] = None,
    ):
        # if connectome is specified, the indices are taken from the connectome
        # and reference to positions in the entire list of nodes/cells
        if connectome is not None and unique_cell_types is None:
            self.unique_cell_types = connectome.unique_cell_types[:].astype("str")
            self.central_cells_index = connectome.central_cells_index[:]
        # alternatively the mapping can be specified from a list of cell types
        # and reference to positions in order of the list
        elif connectome is None and unique_cell_types is not None:
            self.unique_cell_types = unique_cell_types
            self.central_cells_index = None
        else:
            raise ValueError("either cell types or connectome must be specified")
        for index, cell_type in enumerate(self.unique_cell_types):
            super().__setitem__(cell_type, index)

    def __dir__(self):
        return list(set([*dict.__dir__(self), *dict.__iter__(self)]))

    def __len__(self):
        return len(self.unique_cell_types)

    def __iter__(self):
        for cell_type in self.unique_cell_types:
            yield cell_type

    def __getattr__(self, key):
        if isinstance(key, str):
            pass
        elif isinstance(key, Iterable):
            return [dict.__getitem__(self, _key) for _key in key]
        return dict.__getitem__(self, key)

    def __getitem__(self, key):
        return self.__getattr__(key)


class CellTypeArray:
    """Attribute-style accessible map from cell types to coordinates in array.

    Args:
        array: Has the dim-th axis corresponding to unique cell types
            in the connectome or provided cell types.
        connectome: Connectome object.
        cell_types: Array of cell types.
        dim: Axis corresponding to unique cell types.

    Attributes:
        node_indexer (NodeIndexer): Indexer for cell types.
        array (NDArray): The array of cell type data.
        dim (int): Dimension corresponding to cell types.
        cell_types (NDArray[str]): Array of unique cell types.
    """

    node_indexer: NodeIndexer = None
    array: NDArray = None
    dim: float = None

    def __init__(
        self,
        array: Union[NDArray, torch.Tensor],
        connectome: Optional["connectome.ConnectomeFromAvgFilters"] = None,
        cell_types: Optional[NDArray[str]] = None,
        dim: int = -1,
    ):
        self.array = array
        self.dim = dim
        self.node_indexer = NodeIndexer(connectome, cell_types)
        self.cell_types = self.node_indexer.unique_cell_types

    def __bool__(self):
        return self.array is not None

    def __iter__(self):
        for cell_type in self.node_indexer.unique_cell_types:
            yield cell_type

    def __dir__(self):
        return list(
            set([
                *object.__dir__(self),
                *dict.__dir__(self.node_indexer),
                *dict.__iter__(self.node_indexer),
            ])
        )

    @property
    def shape(self):
        if self.array is not None:
            return self.array.shape
        return []

    def __repr__(self):
        shape = list(self.shape)
        desc = f"Array({tuple(shape)})"
        return {k: desc for k in self}.__repr__()

    def values(self):
        return [self[k] for k in self]

    def keys(self):
        return [k for k in self]

    def items(self):
        return [(k, self[k]) for k in self]

    def __len__(self):
        return len(self.node_indexer.unique_cell_types)

    def __getattr__(self, key):
        if self.node_indexer is not None:
            if isinstance(key, slice) and key == slice(None):
                return self.array
            elif isinstance(key, str) and key in self.node_indexer.unique_cell_types:
                indices = np.int_([dict.__getitem__(self.node_indexer, key)])
            elif isinstance(key, Iterable) and all([
                _key in self.node_indexer.unique_cell_types for _key in key
            ]):
                indices = np.int_([
                    dict.__getitem__(self.node_indexer, _key) for _key in key
                ])
            elif key in self.node_indexer.__dir__():
                return object.__getattribute__(self.node_indexer, key)
            else:
                return object.__getattribute__(self, key)
            return np.take(self.array, indices, axis=self.dim)
        return object.__getattribute__(self, key)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        if self.node_indexer is not None and key in self.node_indexer.unique_cell_types:
            if value.shape[-1] != 1:
                value = np.expand_dims(value, self.dim)
            if self.array is None:
                n_cell_types = len(self.node_indexer.unique_cell_types)
                shape = list(value.shape)
                shape[self.dim] = n_cell_types
                self.array = np.zeros(shape)
            # breakpoint()
            index = dict.__getitem__(self.node_indexer, key)
            np.put_along_axis(
                self.array,
                np.expand_dims(np.array([index]), list(range(len(self.array.shape[1:])))),
                value,
                self.dim,
            )
        else:
            object.__setattr__(self, key, value)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    def from_cell_types(self, cell_types):
        activity = self[cell_types]
        return CellTypeArray(
            activity,
            cell_types=cell_types,
            dim=self.dim,
        )


# cell type layout for visualization
layout = {
    "R1": "retina",
    "R2": "retina",
    "R3": "retina",
    "R4": "retina",
    "R5": "retina",
    "R6": "retina",
    "R7": "retina",
    "R8": "retina",
    "L1": "intermediate",
    "L2": "intermediate",
    "L3": "intermediate",
    "L4": "intermediate",
    "L5": "intermediate",
    "Lawf1": "intermediate",
    "Lawf2": "intermediate",
    "Am": "intermediate",
    "C2": "intermediate",
    "C3": "intermediate",
    "CT1(Lo1)": "intermediate",
    "CT1(M10)": "intermediate",
    "Mi1": "intermediate",
    "Mi2": "intermediate",
    "Mi3": "intermediate",
    "Mi4": "intermediate",
    "Mi9": "intermediate",
    "Mi10": "intermediate",
    "Mi11": "intermediate",
    "Mi12": "intermediate",
    "Mi13": "intermediate",
    "Mi14": "intermediate",
    "Mi15": "intermediate",
    "T4a": "output",
    "T4b": "output",
    "T4c": "output",
    "T4d": "output",
    "T1": "output",
    "T2": "output",
    "T2a": "output",
    "T3": "output",
    "T5a": "output",
    "T5b": "output",
    "T5c": "output",
    "T5d": "output",
    "Tm1": "output",
    "Tm2": "output",
    "Tm3": "output",
    "Tm4": "output",
    "Tm5Y": "output",
    "Tm5a": "output",
    "Tm5b": "output",
    "Tm5c": "output",
    "Tm9": "output",
    "Tm16": "output",
    "Tm20": "output",
    "Tm28": "output",
    "Tm30": "output",
    "TmY3": "output",
    "TmY4": "output",
    "TmY5a": "output",
    "TmY9": "output",
    "TmY10": "output",
    "TmY13": "output",
    "TmY14": "output",
    "TmY15": "output",
    "TmY18": "output",
}
