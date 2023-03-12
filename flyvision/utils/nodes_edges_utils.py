import re
from typing import Iterable
import numpy as np


def order_nodes_list(
    nodes_list,
    groups=[
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
):
    """Orders a list of node types by the regular expressions defined in groups.

    Args:
        nodes_list (list): messy list of nodes.
        groups (list): ordered list of regular expressions that match the nodes.layout

    Returns:
        array: ordered list of nodes.
        array: original indices.
    """
    if nodes_list is None:
        return None, None

    _len = len(nodes_list)

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
    matched = {node_type: False for node_type in nodes_list}
    for node_index, node_type in enumerate(nodes_list):
        for group_index, regular_expression in enumerate(groups):
            if re.match(regular_expression, node_type):
                type_groups[group_index].append((node_index, node_type))
                matched[node_type] = True
        if matched[node_type]:
            pass
        else:
            type_groups[len(groups) + 1].append((node_index, node_type))

    # ordered = [y for x in type_groups.values() for y in sorted(x, key=lambda z: sort_fn(z[1]))]
    ordered = []
    for x in type_groups.values():
        for y in sorted(x, key=lambda z: sort_numeric(z[1])):
            ordered.append(y)
    index = [y[0] for y in ordered]
    nodes = [y[1] for y in ordered]

    if set(nodes_list) - set(nodes):
        print(set(nodes_list) - set(nodes))
        raise AssertionError(
            "Defined sorting through regular expressions does not include all node types."
        )

    if _len != len(nodes) or _len != len(index):
        raise ValueError(
            "sorting failed because the resulting array if of " " different length"
        )

    return nodes, index


class NodeIndexer(dict):
    def __init__(self, ctome=None, unique_node_types=None):
        if ctome is not None and unique_node_types is None:
            self.unique_node_types = ctome.unique_node_types[:].astype("str")
            self.central_nodes_index = ctome.central_nodes_index[:]
        elif ctome is None and unique_node_types is not None:
            self.unique_node_types = unique_node_types
        else:
            raise ValueError("either cell_types or ctome must be specified")
        for index, node_type in enumerate(self.unique_node_types):
            super().__setitem__(node_type, index)

    def __dir__(self):
        return list(set([*dict.__dir__(self), *dict.__iter__(self)]))

    def __len__(self):
        return len(self.unique_node_types)

    def __iter__(self):
        for node_type in self.unique_node_types:
            yield node_type

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
        array: has the dim-th axis corresponding to unique cell types
            in the connectome.
        ctome: Connectome object.
        dim: axis correpsonding to unique cell types.
    """

    node_indexer: NodeIndexer = None
    array: np.ndarray = None
    dim: float = None

    def __init__(self, array, ctome=None, cell_types=None, dim=-1):
        self.array = array
        self.dim = dim
        self.node_indexer = NodeIndexer(ctome, cell_types)

    def __iter__(self):
        for node_type in self.node_indexer.unique_node_types:
            yield node_type

    def __dir__(self):
        return list(
            set(
                [
                    *object.__dir__(self),
                    *dict.__dir__(self.node_indexer),
                    *dict.__iter__(self.node_indexer),
                ]
            )
        )

    @property
    def shape(self):
        return self.array.shape

    def __repr__(self):
        shape = list(self.shape)
        shape.pop(self.dim)
        desc = f"Array({tuple(shape)})"
        return {k: desc for k in self}.__repr__()

    def values(self):
        return [self[k] for k in self]

    def keys(self):
        return [k for k in self]

    def items(self):
        return [(k, self[k]) for k in self]

    def __len__(self):
        return len(self.node_indexer.unique_node_types)

    def __getattr__(self, key):

        if self.node_indexer is not None:
            if isinstance(key, str) and key in self.node_indexer.unique_node_types:
                indices = np.int_([dict.__getitem__(self.node_indexer, key)])
            elif isinstance(key, Iterable) and any(
                [_key in self.node_indexer.unique_node_types for _key in key]
            ):
                indices = np.int_(
                    [dict.__getitem__(self.node_indexer, _key) for _key in key]
                )
            elif key in self.node_indexer.__dir__():
                return object.__getattribute__(self.node_indexer, key)
            else:
                return object.__getattribute__(self, key)
            return np.take(self.array, indices, axis=self.dim)
        return object.__getattribute__(self, key)

    def __getitem__(self, key):
        return self.__getattr__(key)

    def __setitem__(self, key, value):
        if self.node_indexer is not None and key in self.node_indexer.unique_node_types:
            if value.shape[-1] != 1:
                value = np.expand_dims(value, self.dim)
            if self.array is None:
                n_cell_types = len(self.node_indexer.unique_node_types)
                shape = list(value.shape)
                shape[self.dim] = n_cell_types
                self.array = np.zeros(shape)
            # breakpoint()
            index = dict.__getitem__(self.node_indexer, key)
            np.put_along_axis(
                self.array,
                np.expand_dims(
                    np.array([index]), list(range(len(self.array.shape[1:])))
                ),
                value,
                self.dim,
            )
        else:
            object.__setattr__(self, key, value)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)


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
