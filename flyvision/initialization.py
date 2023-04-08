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
import functools
from typing import Any, Dict, Iterable, List
import logging
import numpy as np
import pandas as pd

import torch
from torch import Tensor
import torch.nn as nn

from datamate import Namespace

from flyvision.connectome import ConnectomeDir
from flyvision.utils.tensor_utils import atleast_column_vector, where_equal_rows
from flyvision.utils.type_utils import byte_to_str
from flyvision.utils.class_utils import forward_subclass


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


def deepcopy_config(f):
    """Decorator to deepcopy the parameter configuration.

    Note, this is necessary because the parameter configuration is updated
    in the __init__ method of the parameter classes and prevents mutation of the
    param_config in the outer scope.
    """

    @functools.wraps(f)
    def wrapper(cls, param_config, connectome):
        return f(cls, deepcopy(param_config), connectome)

    return wrapper


class Parameter:
    """Base class for all parameters to share across nodes or edges.

    Args:
        param_config: Namespace containing parameter configuration.
        connectome: Connectome object.

    Attributes:
        parameter: InitialDistribution object.
        indices: Indices for parameter sharing.
        keys: Keys to access individual parameter values associated with
            certain identifiers.
        symmetry_masks: Symmetry masks that can be configured optionally to
            apply further symmetry constraints to the parameter values.

    Extension point:
        subclasses must implement __init__(self, param_config, connectome_dir).
        __init__ must configure the above attributes.
        __init__ should be decorated with @deepcopy_config when __init__ updates
        param_config to ensure param_config is not mutated in the outer scope, .
        __init__ udpates param_config to contain key value pairs informed by
        connectome and matching the desired InitialDistribution.
        __init__ stores `parameter` from InitialDistribution(param_config), which
        constructs and holds the nn.Parameter.
        __init__ stores `indices` for parameter sharing through
        `get_scatter_indices(dataframe, grouped_dataframe, groupby)`.
        __init__ stores `keys` to access individual
        parameter values associated with certain identifiers.
        __init__ stores `symmetry_masks` that can be configured optionally
        to apply further symmetry constraints to the parameter values.

        Example:
            class MyParameter(Parameter):
                def __init__(self, param_config, connectome_dir):
                    nodes_dir = connectome.nodes

                    nodes = pd.DataFrame(
                        {k: byte_to_str(nodes_dir[k][:]) for k in param_config.groupby}
                    )
                    grouped_nodes = nodes.groupby(
                        param_config.groupby, as_index=False, sort=False
                    ).first()

                    param_config["type"] = grouped_nodes["type"].values
                    param_config["mean"] = np.repeat(param_config["mean"], len(grouped_nodes))
                    param_config["std"] = np.repeat(param_config["std"], len(grouped_nodes))

                    self.parameter = InitialDistribution(param_config)
                    self.indices = get_scatter_indices(nodes, grouped_nodes, param_config.groupby)
                    self.keys = param_config["type"].tolist()
                    self.symmetry_masks = symmetry_masks(
                        param_config.get("symmetric", []), self.keys
                    )
            param_config = Namespace(
                    type="MyParameter",
                    mean=0,
                    std=1,
                    groupby=["type"],
                    requires_grad=True
                )
            param = Parameter(param_config, connectome)
            type(param) == MyParameter
    """

    parameter: InitialDistribution
    indices: torch.Tensor
    symmetry_masks: List[torch.Tensor]
    keys: List[Any]

    def __new__(cls, param_config: Namespace, connectome: ConnectomeDir):
        obj = forward_subclass(cls, deepcopy(param_config), subclass_key="type")
        # object.__setattr__(obj, "_config", param_config)
        object.__setattr__(obj, "config", deepcopy(param_config))
        return obj

    @deepcopy_config
    def __init__(self, param_config: Namespace, connectome: ConnectomeDir):
        pass

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

    def __len__(self):
        return len(self.raw_values)

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

    def _symmetry(self):
        """Return symmetry constraints from symmetry masks for debugging."""
        keys = np.array(self.keys)
        return [keys[mask.cpu()] for mask in self.symmetry_masks]


# --- node / Cell type parameter ------------------------------------------------


class RestingPotential(Parameter):
    """Initialize resting potentials a.k.a. biases for cell types."""

    @deepcopy_config
    def __init__(self, param_config: Namespace, connectome: ConnectomeDir):
        nodes_dir = connectome.nodes

        nodes = pd.DataFrame(
            {k: byte_to_str(nodes_dir[k][:]) for k in param_config.groupby}
        )
        grouped_nodes = nodes.groupby(
            param_config.groupby, as_index=False, sort=False
        ).first()

        param_config["type"] = grouped_nodes["type"].values
        param_config["mean"] = np.repeat(param_config["mean"], len(grouped_nodes))
        param_config["std"] = np.repeat(param_config["std"], len(grouped_nodes))

        self.parameter = InitialDistribution(param_config)
        self.indices = get_scatter_indices(nodes, grouped_nodes, param_config.groupby)
        self.keys = param_config["type"].tolist()
        self.symmetry_masks = symmetry_masks(
            param_config.get("symmetric", []), self.keys
        )


class TimeConstant(Parameter):
    """Initialize time constants for cell types."""

    @deepcopy_config
    def __init__(self, param_config: Namespace, connectome: ConnectomeDir):
        nodes_dir = connectome.nodes

        nodes = pd.DataFrame(
            {k: byte_to_str(nodes_dir[k][:]) for k in param_config.groupby}
        )
        grouped_nodes = nodes.groupby(
            param_config.groupby, as_index=False, sort=False
        ).first()

        param_config["type"] = grouped_nodes["type"].values
        param_config["value"] = np.repeat(param_config["value"], len(grouped_nodes))

        self.indices = get_scatter_indices(nodes, grouped_nodes, param_config.groupby)
        self.parameter = InitialDistribution(param_config)
        self.keys = param_config["type"].tolist()
        self.symmetry_masks = symmetry_masks(
            param_config.get("symmetric", []), self.keys
        )


# -- edge / synapse type parameter ----------------------------------------------


class SynapseSign(Parameter):
    """Initialize synapse signs for edge types."""

    @deepcopy_config
    def __init__(self, param_config: Namespace, connectome: ConnectomeDir) -> None:
        edges_dir = connectome.edges

        edges = pd.DataFrame(
            {k: byte_to_str(edges_dir[k][:]) for k in [*param_config.groupby, "sign"]}
        )
        grouped_edges = edges.groupby(
            param_config.groupby, as_index=False, sort=False
        ).first()

        param_config.source_type = grouped_edges.source_type.values
        param_config.target_type = grouped_edges.target_type.values
        param_config.value = grouped_edges.sign.values

        self.indices = get_scatter_indices(edges, grouped_edges, param_config.groupby)
        self.parameter = InitialDistribution(param_config)
        self.keys = list(
            zip(
                param_config.source_type.tolist(),
                param_config.target_type.tolist(),
            )
        )
        self.symmetry_masks = symmetry_masks(
            param_config.get("symmetric", []), self.keys
        )


class SynapseCount(Parameter):
    """Initialize synapse counts for edge types."""

    @deepcopy_config
    def __init__(self, param_config: Namespace, connectome: ConnectomeDir) -> None:
        mode = param_config.get("mode", "")
        if mode != "mean":
            raise NotImplementedError(
                f"SynapseCount does not implement {mode}. Implement "
                "a custom Parameter subclass."
            )

        edges_dir = connectome.edges

        edges = pd.DataFrame(
            {k: byte_to_str(edges_dir[k][:]) for k in [*param_config.groupby, "n_syn"]}
        )
        grouped_edges = edges.groupby(
            param_config.groupby, as_index=False, sort=False
        ).mean()

        param_config.source_type = grouped_edges.source_type.values
        param_config.target_type = grouped_edges.target_type.values
        param_config.du = grouped_edges.du.values
        param_config.dv = grouped_edges.dv.values

        param_config.mode = "mean"
        param_config.mean = np.log(grouped_edges.n_syn.values)

        self.indices = get_scatter_indices(edges, grouped_edges, param_config.groupby)
        self.parameter = InitialDistribution(param_config)
        self.keys = list(
            zip(
                param_config.source_type.tolist(),
                param_config.target_type.tolist(),
                param_config.du.tolist(),
                param_config.dv.tolist(),
            )
        )
        self.symmetry_masks = symmetry_masks(
            param_config.get("symmetric", []), self.keys
        )


class SynapseCountScaling(Parameter):
    """Initialize synapse count scaling for edge types."""

    @deepcopy_config
    def __init__(self, param_config: Namespace, connectome: ConnectomeDir) -> None:
        edges_dir = connectome.edges

        edges = pd.DataFrame(
            {k: byte_to_str(edges_dir[k][:]) for k in [*param_config.groupby, "n_syn"]}
        )
        grouped_edges = edges.groupby(
            param_config.groupby, as_index=False, sort=False
        ).mean()

        # to initialize synapse strengths with 1/<N>_rf
        syn_strength = 1 / grouped_edges.n_syn.values  # 1/<N>_rf

        # scale synapse strengths of chemical and electrical synapses
        # individually
        syn_strength[grouped_edges[grouped_edges.edge_type == "chem"].index] *= getattr(
            param_config, "scale_chem", 0.01
        )
        syn_strength[grouped_edges[grouped_edges.edge_type == "elec"].index] *= getattr(
            param_config, "scale_elec", 0.01
        )

        param_config.target_type = grouped_edges.target_type.values
        param_config.source_type = grouped_edges.source_type.values
        param_config.value = syn_strength

        self.indices = get_scatter_indices(edges, grouped_edges, param_config.groupby)
        self.parameter = InitialDistribution(param_config)
        self.keys = list(
            zip(
                param_config.source_type.tolist(),
                param_config.target_type.tolist(),
            )
        )
        self.symmetry_masks = symmetry_masks(
            param_config.get("symmetric", []), self.keys
        )


def get_scatter_indices(
    dataframe: pd.DataFrame, grouped_dataframe: pd.DataFrame, groupby: List[str]
) -> Tensor:
    """Indices for scattering operations to share parameters.

    Maps each node/edge from the complete computational graph to a parameter
    index.

    Args:
        dataframe (pd.DataFrame): dataframe of nodes or edges of the graph.
        grouped_dataframe (list): aggregated version of the same dataframe.
        groupby (list): the same columns from which the grouped_dataframe was
            constructed.

    For N elements that are grouped into M groups, this function
    returns N indices from 0 to M-1 that can be used to scatter the
    parameters of the M groups to the N elements.

    To illustrate, consider the following simplified example:
        elements = ["A", "A", "A", "B", "B", "C", "D", "D", "E"]
        groups = ["A", "B", "C", "D", "E"]
        parameter = [1, 2, 3, 4, 5]
        # get_scatter_indices would return
        scatter_indices = [0, 0, 0, 1, 1, 2, 3, 3, 4]
        scattered_parameters = [parameter[idx] for idx in scatter_indices]
        scattered_parameters == [1, 1, 1, 2, 2, 3, 4, 4, 5]
    """
    ungrouped_elements = zip(*[dataframe[k][:] for k in groupby])
    grouped_elements = zip(*[grouped_dataframe[k][:] for k in groupby])
    to_index = {k: i for i, k in enumerate(grouped_elements)}
    return torch.tensor([to_index[k] for k in ungrouped_elements])


def symmetry_masks(
    symmetric: List[Any], keys: List[Any], as_mask: bool = False
) -> List[torch.Tensor]:
    """Masks subsets of parameters for joint constraints e.g. to their mean.

    Args:
        symmetric: contains subsets of keys that point to the subsets
            of parameters to be indexed.
        keys: list of keys that point to individual parameter values.
        as_mask: if True, returns a boolean mask, otherwise integer indices.

    Returns:
        list of masks List[torch.BoolTensor]

    Note: this is experimental for configuration-based fine-grained shared
    parameter optimization, e.g. for models includig multi-compartment cells
    or gap junctions.
    Example 1:
    for node type parameters with individual node types as keys
        symmetric = [["T4a", "T4b", "T4c", "T4d"],
                     ["T5a", "T5b", "T5c", "T5d"]]
    would be used to constrain the parameter values of all T4 subtypes to
    their joint mean and the parameter values of all T5 subtypes to their
    joint mean.
    Exaple 2:
    for edge type parameters with individual edge types as keys
        symmetric = [[("CT1(M10)", "CT1(Lo1)"), ("CT1(Lo1)", "CT1(M10)")]]
    would be used to constrain the edge parameter of the directed edge from
    CT1(M10) to CT1(Lo1) and the directed edge from CT1(Lo1) to CT1(M10) to
    their joint mean.
    """
    if not symmetric:
        return []
    symmetry_masks = []  # type: List[torch.Tensor]
    keys = atleast_column_vector(keys)
    for i, identifiers in enumerate(symmetric):
        identifiers = atleast_column_vector(identifiers)
        # to allow identifiers like [None, "A", None, 0]
        # for parameters that have tuples as keys
        columns = np.arange(identifiers.shape[1] + 1)[
            np.where((identifiers != None).all(axis=0))
        ]
        try:
            symmetry_masks.append(
                torch.tensor(
                    where_equal_rows(
                        identifiers[:, columns], keys[:, columns], as_mask=as_mask
                    )
                )
            )
        except Exception as e:
            raise ValueError(
                f"{identifiers} cannot be a symmetry constraint"
                f" for parameter with keys {keys}: {e}"
            ) from e
    return symmetry_masks
