"""
The parameters that the networks can be initialized with. Each parameter is a
type on its own, because different parameters are shared differently. These
types handle the initialization of indices to perform gather and scatter opera-
tions. Parameter types can be initialized from a range of initial distribution
types.

Note: to maintain compatibility with old configurations, e.g. to reinitialize
    a trained network, careful when refactoring any of these types or syntax.
"""

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Union
import logging
import numpy as np
import pandas as pd

import torch
from torch import Tensor
import torch.nn as nn

from datamate import Namespace

from flyvision.utils.class_utils import forward_subclass
from flyvision.connectome import ConnectomeDir, NodeDir, EdgeDir

logging = logging.getLogger()

# --- Supplementary Error types -------------------------------------------------


class ParameterConfigError(ValueError):
    pass


# --- Initial distribution types ------------------------------------------------


class InitialDistribution:

    """Initial distribution base class.

    Attributes:
        raw_values: initial parameters must store raw_values as attribute in
            their __init__.
        readers: readers will be written by the network during initialization.

    Extension point: to add a new initial distribution type, subclass this
        class and implement the __init__ method. The __init__ method
        should take the param_config as its first argument, and should store
        the attribute raw_values as a torch.nn.Parameter.

    An example of a viable param_config is:
        param_config = Namespace(
            requires_grad=True,
            initial_dist="Normal",
            mean=0,
            std=1,
            mode="sample",
        )
    """

    raw_values: Tensor
    readers: Dict[str, Tensor]

    def __new__(cls, param_config: Namespace, *args, **kwargs):
        return forward_subclass(cls, param_config, subclass_key="initial_dist")

    @property
    def semantic_values(self):
        """Optional reparametrization of raw values invoked for computation."""
        return self.raw_values

    def __repr__(self):
        return f"{self.__class__.__name__} (semantic values): \n{self.semantic_values}"

    def __len__(self):
        return len(self.raw_values)

    def clamp(self, values, param_config):
        """To clamp the raw_values of the parameters at initialization.

        Note, mild clash with raw_values/semantic_values reparametrization.
        Parameters that use reparametrization in terms of semantic_values
        should not use clamp.
        """
        mode = param_config.get("clamp", False)
        if mode == "non_negative":
            values.clamp_(min=0)
        elif isinstance(mode, Iterable) and len(mode) == 2:
            values.clamp_(*mode)
        elif mode in [False, None]:
            return values
        else:
            raise ParameterConfigError(f"{mode} not a valid argument for clamp")
        return values


class Value(InitialDistribution):
    """Initializes parameters with a single value.

    Example param_config:
        param_config = Namespace(
            requires_grad=True,
            initial_dist="Value",
            value=0,
        )
    """

    def __init__(self, param_config: Namespace) -> None:
        _values = torch.tensor(param_config.value).float()
        _values = self.clamp(_values, param_config)
        self.raw_values = nn.Parameter(
            _values, requires_grad=param_config.requires_grad
        )


class Normal(InitialDistribution):
    """Initializes parameters independently from normal distributions.

    Example param_config:
        param_config = Namespace(
            requires_grad=True,
            initial_dist="Normal",
            mean=0,
            std=1,
            mode="sample",
        )
    """

    def __init__(self, param_config: Namespace) -> None:
        if param_config.mode == "mean":
            _values = torch.tensor(param_config.mean).float()
        elif param_config.mode == "sample":
            seed = param_config.get("seed", None)
            if seed is not None:
                torch.manual_seed(seed)
            try:
                _values = torch.distributions.normal.Normal(
                    torch.tensor(param_config.mean).float(),
                    torch.tensor(param_config.std).float(),
                ).sample()
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to sample from normal distribution with mean {param_config.mean} and std {param_config.std}"
                ) from e
        else:
            raise ValueError("Mode must be either mean or sample.")
        _values = self.clamp(_values, param_config)
        self.raw_values = nn.Parameter(
            _values, requires_grad=param_config.requires_grad
        )


class Lognormal(InitialDistribution):
    """Initializes parameters independently from lognormal distributions.

    Example param_config:
        param_config = Namespace(
            requires_grad=True,
            initial_dist="Lognormal",
            mean=0,
            std=1,
            mode="sample",
        )

    Note, the lognormal distribution reparametrizes a normal through semantic
    values.
    """

    def __init__(self, param_config: Namespace) -> None:
        if param_config.get("clamp", False):
            logging.warning(
                f"clamping has no effect for {self.__class__.__name__} parameters"
                " because clamping acts on raw_values"
                " but the lognormal parameter semantic values are raw_values.exp()"
            )

        if param_config.mode == "mean":
            _values = torch.tensor(param_config.mean).float()
        elif param_config.mode == "sample":
            # The log is normally distributed and in the class SynCount we take the log, thus the normal distr. here.
            seed = param_config.get("seed", None)
            if seed is not None:
                torch.manual_seed(seed)
            try:
                _values = torch.distributions.normal.Normal(
                    torch.tensor(param_config.mean).float(),
                    torch.tensor(param_config.std).float(),
                ).sample()
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to sample from normal distribution with mean {param_config.mean} and std {param_config.std}"
                ) from e
        else:
            raise ValueError("Mode must be either mean or sample.")
        self.raw_values = nn.Parameter(
            _values, requires_grad=param_config.requires_grad
        )

    @property
    def semantic_values(self):
        """n_syn ~ self._values.exp()."""
        return self.raw_values.exp()


# --- Parameter types -----------------------------------------------------------


class Parameter:
    """Base class for all parameters.

    Extension point: to add a new parameter type, subclass this
        class and implement the __init__ method. The __init__ method
        should take the param_config as its first argument, and the connectome
        directory as its second argument. From the connectome directory,
        the __init__ method should create an aggregated table of either nodes or
        edges that share parameters based on their type, coordinates or similar
        identifiers. The table is stored in the passed param_config as numpy arrays,
        potentially overriding value of the passed param_config according to the
        required InitialDistribution. The ultimate nn.Parameter is then created
        by passing the param_config to the InitialDistribution. The scatter
        indices are created by passing the param_config and the connectome sub directory
        to the get_scatter_indices function. The __init__ method is also expected
        to store a list of keys that can be used to access
        to access the individual parameter values associated with certain
        identifiers. It is also expected to store an optional
        list of symmetry masks that can be used to apply symmetry constraints
        to the parameter values. TODO: this is really complicated and implicit,
        maybe we can simplify this framework.

        Example:
            class MyParameter(Parameter):
                def __init__(self, param_config, connectome_dir):
                    # create aggregated table of nodes or edges
                    # that share parameters
                    param_config.value = np.array([1, 2, 3])
                    param_config.requires_grad = True
                    param_config.initial_dist = "Value"
                    param_config.mode = "mean"
                    # create parameter
                    self.parameter = InitialDistribution(param_config)
                    # create scatter indices
                    self.indices = get_scatter_indices(param_config, connectome_dir)
                    # store keys
                    self.keys = ["key1", "key2", "key3"]
                    # store symmetry masks
                    self.symmetry_masks = [mask1, mask2, mask3]

    """

    parameter: InitialDistribution
    indices: torch.Tensor
    symmetry_masks: List[torch.Tensor]
    keys: List[Any]

    def __new__(cls, param_config: Namespace, *args, **kwargs):
        obj = forward_subclass(cls, param_config, subclass_key="type")
        object.__setattr__(obj, "_config", param_config)
        object.__setattr__(obj, "config", deepcopy(param_config))
        return obj

    def __repr__(self):
        init_arg_names = list(self.__init__.__annotations__.keys())
        dir_type = self.__init__.__annotations__[init_arg_names[1]].__name__
        return f"{self.__class__.__name__}({self.config}, {dir_type})"

    def __getitem__(self, key):
        if key in self.keys:
            if self.parameter.raw_values.dim() == 0:
                return self.parameter.raw_values
            return self.parameter.raw_values[self.keys.index(key)]
        else:
            raise ValueError(key)

    # -- InitialDistribution API

    @property
    def raw_values(self) -> torch.Tensor:
        return self.parameter.raw_values

    @property
    def semantic_values(self) -> torch.Tensor:
        return self.parameter.semantic_values

    @property
    def readers(self) -> Dict[str, torch.Tensor]:
        return self.parameter.readers

    @readers.setter
    def readers(self, value) -> None:
        self.parameter.readers = value


# --- node / Cell type parameter ------------------------------------------------


class RestingPotential(Parameter):
    """Initialize resting potentials a.k.a. biases for cell types."""

    def __init__(self, param_config: Namespace, connectome: ConnectomeDir):
        nodes_dir = connectome.nodes

        # equals order in connectome.unique_cell_types
        nodes = pd.DataFrame(dict(type=nodes_dir.type[:].astype(str))).drop_duplicates()

        param_config["type"] = nodes["type"].values
        param_config["mean"] = np.repeat(param_config["mean"], len(nodes))
        param_config["std"] = np.repeat(param_config["std"], len(nodes))

        self.symmetry_masks = symmetry_mask_for_nodes(
            param_config.get("symmetric", []), nodes
        )

        self.indices = get_scatter_indices(param_config, nodes_dir)
        self.parameter = InitialDistribution(param_config)
        self.keys = param_config["type"].tolist()


class TimeConstant(Parameter):
    """Initialize time constants for cell types."""

    def __init__(self, param_config: Namespace, connectome: ConnectomeDir):
        nodes_dir = connectome.nodes

        nodes = pd.DataFrame(dict(type=nodes_dir.type[:].astype(str))).drop_duplicates()

        param_config["type"] = nodes["type"].values
        param_config["value"] = np.repeat(param_config["value"], len(nodes))

        self.symmetry_masks = symmetry_mask_for_nodes(
            param_config.get("symmetric", []), nodes
        )

        self.indices = get_scatter_indices(param_config, nodes_dir)
        self.parameter = InitialDistribution(param_config)
        self.keys = param_config["type"].tolist()


# -- edge / synapse type parameter ----------------------------------------------


class SynapseSign(Parameter):
    """Initialize synapse signs for edge types."""

    def __init__(self, param_config: Namespace, connectome: ConnectomeDir) -> None:
        edges_dir = connectome.edges

        edges = pd.DataFrame(
            dict(
                source_type=edges_dir.source_type,
                target_type=edges_dir.target_type,
                sign=edges_dir.sign,
            )
        )
        edges = edges.groupby(
            ["source_type", "target_type"], sort=False, as_index=False
        ).mean()
        assert all(
            np.abs(edges.sign.values) == np.ones(len(edges))
        ), "Inconsistent edge signs."

        param_config.source_type = edges.source_type.values
        param_config.target_type = edges.target_type.values
        param_config.value = edges.sign.values
        self.indices = get_scatter_indices(param_config, edges_dir)
        self.parameter = InitialDistribution(param_config)
        self.keys = list(
            zip(
                param_config.source_type.tolist(),
                param_config.target_type.tolist(),
            )
        )
        self.symmetry_masks = []


class SynapseCount(Parameter):
    """Initialize synapse counts for edge types."""

    def __init__(self, param_config: Namespace, connectome: ConnectomeDir) -> None:
        edges_dir = connectome.edges

        mode = param_config.get("mode", "")
        if mode != "mean":
            raise NotImplementedError(
                f"SynapseCount does not implement {mode}. Implement "
                "a custom Parameter subclass."
            )

        edges = pd.DataFrame(
            dict(
                source_type=edges_dir.source_type,
                target_type=edges_dir.target_type,
                du=edges_dir.du,
                dv=edges_dir.dv,
                n_syn=edges_dir.n_syn,
            )
        )
        offset_keys = ["source_type", "target_type", "du", "dv"]
        edges = edges.groupby(offset_keys, sort=False, as_index=False).mean()

        param_config.source_type = edges.source_type.values
        param_config.target_type = edges.target_type.values
        param_config.du = edges.du.values
        param_config.dv = edges.dv.values

        param_config.mode = "mean"
        param_config.mean = np.log(edges.n_syn.values)

        self.symmetry_masks = symmetry_mask_for_edges(
            param_config.get("symmetric", []), edges
        )
        self.indices = get_scatter_indices(param_config, edges_dir)
        self.parameter = InitialDistribution(param_config)
        self.keys = list(
            zip(
                param_config.source_type.tolist(),
                param_config.target_type.tolist(),
                param_config.du.tolist(),
                param_config.dv.tolist(),
            )
        )


class SynapseCountScaling(Parameter):
    """Initialize synapse count scaling for edge types."""

    def __init__(self, param_config: Namespace, connectome: ConnectomeDir) -> None:
        edges_dir = connectome.edges

        edges = pd.DataFrame(
            dict(
                source_type=edges_dir.source_type,
                target_type=edges_dir.target_type,
                edge_type=edges_dir.edge_type,
                n_syn=edges_dir.n_syn,
            )
        )
        edges = edges.groupby(
            ["source_type", "target_type", "edge_type"],
            sort=False,
            as_index=False,
        ).mean()

        # to initialize synapse strengths with 1/<N>_rf
        syn_strength = 1 / edges.n_syn.values  # 1/<N>_rf

        # scale synapse strengths of chemical and electrical synapses
        # individually
        syn_strength[edges[edges.edge_type == b"chem"].index] *= getattr(
            param_config, "scale_chem", 0.01
        )
        syn_strength[edges[edges.edge_type == b"elec"].index] *= getattr(
            param_config, "scale_elec", 0.01
        )

        param_config.target_type = edges.target_type.values
        param_config.source_type = edges.source_type.values
        param_config.value = syn_strength

        self.symmetry_masks = symmetry_mask_for_edges(
            param_config.get("symmetric", []), edges
        )

        self.indices = get_scatter_indices(param_config, edges_dir)
        self.parameter = InitialDistribution(param_config)
        self.keys = list(
            zip(
                param_config.source_type.tolist(),
                param_config.target_type.tolist(),
            )
        )


def get_scatter_indices(
    param_config: Namespace, nodes_or_edges_dir: Union[NodeDir, EdgeDir]
) -> Tensor:
    """Indices for scattering operations to share parameters.

    Maps each node/edge from the complete computational graph to a parameter
    index.

    Note: 'param_config' must have all the attributes that exist as h5 files in
        'node_or_edges_dir' in order to group by them.
    """
    groupby = set(nodes_or_edges_dir).intersection(set(param_config))

    # to automatically cast the arrays in param_config to the same
    # type that comes from the h5 data. mainly 'U'-strings to 'S'-strings
    dtypes = {k: nodes_or_edges_dir[k][:].dtype for k in groupby}

    def cast(array, key):
        return array.astype(dtypes[key])

    # to get all groupby shared in the connectome part and the param config
    # e.g. ["type"] for nodes and ["source_type", "target_type"] for edges
    # (e.g. to group by all edges to shared edge parameters)
    groupby = set(nodes_or_edges_dir).intersection(set(param_config))
    # creates a table for all groupby from the connectome part
    ctome_elements = zip(
        *[nodes_or_edges_dir[k][:] for k in groupby]
    )  # type: zip[List[Any]]
    # creates a table for all groupby from the param_config
    param_elements = zip(
        *[cast(param_config[k][:], k) for k in groupby]
    )  # type: zip[List[Any]]
    # create dictionary with rows from param_config as keys and indices as values
    mapping = {k: i for i, k in enumerate(param_elements)}
    # to eventually map each row from the expanded connectome graph onto one
    # entry in the parameter config
    return torch.tensor([mapping[k] for k in ctome_elements])


def symmetry_mask_for_nodes(
    symmetric: List[List[str]], nodes: pd.DataFrame
) -> List[torch.Tensor]:
    """One additional way to constrain network elements to have the same
    parameter values.

    Note, this method stores one mask per shared tuple of the size of the
    parameter and should thus be used sparsely because its not very memory
    friendly. The bulk part of parameter sharing is achieved through scatter
    and gather operations.

    Args:
        symmetric: list of tuples of cell types that share parameters.
        nodes: DataFrame containing 'type' column.

    Returns:
        list of masks List[torch.BoolTensor]

    Example:
        symmetric = [["T4a", "T4b", "T4c, "T4d"], ["T5a", "T5b", "T5c", "T5d"]]
        would return two masks, one to constrain all T4 subtypes to the same
        parameter and another to constrain all T5 subtypes to the same parameter.
    """
    if not symmetric:
        return []
    nodes.reset_index(drop=True, inplace=True)
    symmetric = np.array(symmetric)
    symmetry_masks = [
        torch.zeros(len(nodes)).bool() for _ in symmetric
    ]  # type: List[torch.Tensor]
    for i, cell_types in enumerate(symmetric):
        for index, row in nodes.iterrows():
            if row.type in cell_types:
                symmetry_masks[i][index] = torch.tensor(1).bool()
    return symmetry_masks


def symmetry_mask_for_edges(symmetric: List[List[List[str]]], edges: pd.DataFrame):
    """One additional way to constrain network elements to have the same
    parameter values. Particularly for electric synapses.

    Note, this method stores one mask per shared tuple of the size of the
    parameter and should thus be used sparsely because its not very memory
    friendly. The bulk part of parameter sharing is achieved through scatter
    and gather operations.

    Args:
        symmetric: list of tuples of edges that share parameters. Only requires
            the edge in one direction if it is bidirectional.
        edges: DataFrame containing 'source_type' and 'target_type' column.

    Returns:
        list of masks List[torch.BoolTensor]

    Example:
        symmetric = [[("CT1(M10)", "T4a"), ("CT1(M10)", "T4b")],
                     [("CT1(Lo1)", "T5a"), ("CT1(Lo1)", "T5b")]]
        would return two masks, one to constrain all CT1(M10) to T4a and T4b
        connections to the same parameters and another one to constrain
        CT1(Lo1) to T5a and T5b to the same parameter.
    """
    if not symmetric:
        return []
    edges.reset_index(drop=True, inplace=True)
    symmetric = np.array(symmetric).astype("S")
    symmetry_masks = [
        torch.zeros(len(edges)).bool() for _ in symmetric
    ]  # type: List[torch.Tensor]

    for i, edge_types in enumerate(symmetric):
        for index, row in edges.iterrows():
            if (row.source_type, row.target_type) in edge_types or (
                row.target_type,
                row.source_type,
            ) in edge_types:
                symmetry_masks[i][index] = torch.tensor(1).bool()

    return symmetry_masks
