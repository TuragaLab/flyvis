"""
The parameters that the networks can be initialized with. Each parameter is a
type on its own, because different parameters are shared differently. These
types handle the initialization of indices to perform gather and scatter opera-
tions. Parameter types can be initialized from a range of initial distribution
types.
"""

import functools
import logging
from copy import deepcopy
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datamate import Namespace
from torch import Tensor

from flyvis import device
from flyvis.connectome import ConnectomeFromAvgFilters
from flyvis.utils.class_utils import forward_subclass
from flyvis.utils.tensor_utils import atleast_column_vector, where_equal_rows
from flyvis.utils.type_utils import byte_to_str

logging = logging.getLogger(__name__)

__all__ = [
    "Parameter",
    "RestingPotential",
    "TimeConstant",
    "SynapseSign",
    "SynapseCount",
    "SynapseCountScaling",
    "InitialDistribution",
    "Value",
    "Normal",
    "Lognormal",
]

# --- Initial distribution types ------------------------------------------------


class InitialDistribution:
    """Initial distribution base class.

    Attributes:
        raw_values (Tensor): Initial parameters must store raw_values as attribute in
            their __init__.
        readers (Dict[str, Tensor]): Readers will be written by the network during
            initialization.

    Note:
        To add a new initial distribution type, subclass this class and implement the
        __init__ method. The __init__ method should take the param_config as its first
        argument, and should store the attribute raw_values as a torch.nn.Parameter.

    Example:
        An example of a viable param_config is:
        ```python
        param_config = Namespace(
            requires_grad=True,
            initial_dist="Normal",
            mean=0,
            std=1,
            mode="sample",
        )
        ```
    """

    raw_values: Tensor
    readers: Dict[str, Tensor]

    @property
    def semantic_values(self):
        """Optional reparametrization of raw values invoked for computation."""
        return self.raw_values

    def __repr__(self):
        return f"{self.__class__.__name__} (semantic values): \n{self.semantic_values}"

    def __len__(self):
        return len(self.raw_values)

    def clamp(self, values, mode):
        """To clamp the raw_values of the parameters at initialization.

        Note, mild clash with raw_values/semantic_values reparametrization.
        Parameters that use reparametrization in terms of semantic_values
        should not use clamp.
        """
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

    Args:
        value: The value to initialize the parameter with.
        requires_grad (bool): Whether the parameter requires gradients.
        clamp (bool, optional): Whether to clamp the values. Defaults to False.
        **kwargs: Additional keyword arguments.

    Example:
        ```python
        param_config = Namespace(
            requires_grad=True,
            initial_dist="Value",
            value=0,
        )
        ```
    """

    def __init__(self, value, requires_grad, clamp=False, **kwargs) -> None:
        _values = torch.tensor(value).float()
        _values = self.clamp(_values, clamp)
        self.raw_values = nn.Parameter(_values, requires_grad=requires_grad)


class Normal(InitialDistribution):
    """Initializes parameters independently from normal distributions.

    Args:
        mean: The mean of the normal distribution.
        std: The standard deviation of the normal distribution.
        requires_grad (bool): Whether the parameter requires gradients.
        mode (str, optional): The initialization mode. Defaults to "sample".
        clamp (bool, optional): Whether to clamp the values. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        **kwargs: Additional keyword arguments.

    Example:
        ```python
        param_config = Namespace(
            requires_grad=True,
            initial_dist="Normal",
            mean=0,
            std=1,
            mode="sample",
        )
        ```
    """

    def __init__(
        self, mean, std, requires_grad, mode="sample", clamp=False, seed=None, **kwargs
    ) -> None:
        if mode == "mean":
            _values = torch.tensor(mean).float()
        elif mode == "sample":
            # set seed for reproducibility and avoid seeding the global RNG
            generator = torch.Generator(device=device)
            if seed is not None:
                generator.manual_seed(seed)
            else:
                generator.seed()
            try:
                _values = torch.normal(
                    torch.tensor(mean).float(),
                    torch.tensor(std).float(),
                    generator=generator,
                )
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to sample from normal with mean {mean} and std {std}"
                ) from e
        else:
            raise ValueError("Mode must be either mean or sample.")
        _values = self.clamp(_values, clamp)
        self.raw_values = nn.Parameter(_values, requires_grad=requires_grad)


class Lognormal(Normal):
    """Initializes parameters independently from lognormal distributions.

    Note:
        The lognormal distribution reparametrizes a normal through semantic values.

    Example:
        ```python
        param_config = Namespace(
            requires_grad=True,
            initial_dist="Lognormal",
            mean=0,
            std=1,
            mode="sample",
        )
        ```
    """

    @property
    def semantic_values(self):
        """n_syn ~ self._values.exp()."""
        return self.raw_values.exp()


# --- Parameter types -----------------------------------------------------------


def deepcopy_config(f):
    """Decorator to deepcopy the parameter configuration.

    Note:
        This decorator is necessary because the `__init__` method of parameter classes
        often modifies the `param_config` object. By creating a deep copy, we ensure
        that these modifications don't affect the original `param_config` object in the
        outer scope. This prevents unintended side effects and maintains the integrity
        of the original configuration.
    """

    @functools.wraps(f)
    def wrapper(cls, param_config, connectome):
        cls.config = deepcopy(param_config)
        return f(cls, deepcopy(param_config), connectome)

    return wrapper


class Parameter:
    """Base class for all parameters to share across nodes or edges.

    Args:
        param_config (Namespace): Namespace containing parameter configuration.
        connectome (Connectome): Connectome object.

    Attributes:
        parameter (InitialDistribution): InitialDistribution object.
        indices (torch.Tensor): Indices for parameter sharing.
        keys (List[Any]): Keys to access individual parameter values associated with
            certain identifiers.
        symmetry_masks (List[torch.Tensor]): Symmetry masks that can be configured
            optionally to apply further symmetry constraints to the parameter values.

    Note:
        Subclasses must implement `__init__(self, param_config, connectome_dir)` with the
        following requirements:

        1. Configure all attributes defined in the base class.
        2. Decorate `__init__` with `@deepcopy_config` if it updates `param_config` to
           prevent mutations in the outer scope.
        3. Update `param_config` with key-value pairs informed by `connectome` and
           matching the desired `InitialDistribution`.
        4. Store `parameter` from `InitialDistribution(param_config)`, which constructs
           and holds the `nn.Parameter`.
        5. Store `indices` for parameter sharing using
           `get_scatter_indices(dataframe, grouped_dataframe, groupby)`.
        6. Store `keys` to access individual parameter values associated with certain
           identifiers.
        7. Store `symmetry_masks` (optional) to apply further symmetry constraints to
           the parameter values.

        Example implementation structure:

        ```python
        @deepcopy_config
        def __init__(self, param_config: Namespace, connectome: Connectome):
            # Update param_config based on connectome data
            # ...

            # Initialize parameter
            self.parameter = InitialDistribution(param_config)

            # Set up indices, keys, and symmetry masks
            self.indices = get_scatter_indices(...)
            self.keys = ...
            self.symmetry_masks = symmetry_masks(...)
        ```
    """

    parameter: InitialDistribution
    indices: torch.Tensor
    symmetry_masks: List[torch.Tensor]
    keys: List[Any]

    @deepcopy_config
    def __init__(self, param_config: Namespace, connectome: ConnectomeFromAvgFilters):
        pass

    def __repr__(self):
        """Return a string representation of the Parameter object."""
        init_arg_names = list(self.__init__.__annotations__.keys())
        dir_type = self.__init__.__annotations__[init_arg_names[1]].__name__
        return f"{self.__class__.__name__}({self.config}, {dir_type})"

    def __getitem__(self, key):
        """Get parameter value for a given key."""
        if key in self.keys:
            if self.parameter.raw_values.dim() == 0:
                return self.parameter.raw_values
            return self.parameter.raw_values[self.keys.index(key)]
        else:
            raise ValueError(key)

    def __len__(self):
        """Return the length of raw_values."""
        return len(self.raw_values)

    @property
    def raw_values(self) -> torch.Tensor:
        """Get raw parameter values."""
        return self.parameter.raw_values

    @property
    def semantic_values(self) -> torch.Tensor:
        """Get semantic parameter values."""
        return self.parameter.semantic_values

    @property
    def readers(self) -> Dict[str, torch.Tensor]:
        """Get parameter readers."""
        return self.parameter.readers

    @readers.setter
    def readers(self, value) -> None:
        """Set parameter readers."""
        self.parameter.readers = value

    def _symmetry(self):
        """Return symmetry constraints from symmetry masks for debugging."""
        keys = np.array(self.keys)
        return [keys[mask.cpu()] for mask in self.symmetry_masks]


# --- node / Cell type parameter ------------------------------------------------


class RestingPotential(Parameter):
    """Initialize resting potentials a.k.a. biases for cell types."""

    @deepcopy_config
    def __init__(self, param_config: Namespace, connectome: ConnectomeFromAvgFilters):
        nodes_dir = connectome.nodes

        nodes = pd.DataFrame({
            k: byte_to_str(nodes_dir[k][:]) for k in param_config.groupby
        })
        grouped_nodes = nodes.groupby(
            param_config.groupby, as_index=False, sort=False
        ).first()

        param_config["type"] = grouped_nodes["type"].values
        param_config["mean"] = np.repeat(param_config["mean"], len(grouped_nodes))
        param_config["std"] = np.repeat(param_config["std"], len(grouped_nodes))

        self.parameter = forward_subclass(
            InitialDistribution, param_config, subclass_key="initial_dist"
        )
        self.indices = get_scatter_indices(nodes, grouped_nodes, param_config.groupby)
        self.keys = param_config["type"].tolist()
        self.symmetry_masks = symmetry_masks(param_config.get("symmetric", []), self.keys)


class TimeConstant(Parameter):
    """Initialize time constants for cell types."""

    @deepcopy_config
    def __init__(self, param_config: Namespace, connectome: ConnectomeFromAvgFilters):
        nodes_dir = connectome.nodes

        nodes = pd.DataFrame({
            k: byte_to_str(nodes_dir[k][:]) for k in param_config.groupby
        })
        grouped_nodes = nodes.groupby(
            param_config.groupby, as_index=False, sort=False
        ).first()

        param_config["type"] = grouped_nodes["type"].values
        param_config["value"] = np.repeat(param_config["value"], len(grouped_nodes))

        self.indices = get_scatter_indices(nodes, grouped_nodes, param_config.groupby)
        self.parameter = forward_subclass(
            InitialDistribution, param_config, subclass_key="initial_dist"
        )
        self.keys = param_config["type"].tolist()
        self.symmetry_masks = symmetry_masks(param_config.get("symmetric", []), self.keys)


# -- edge / synapse type parameter ----------------------------------------------


class SynapseSign(Parameter):
    """Initialize synapse signs for edge types."""

    @deepcopy_config
    def __init__(
        self, param_config: Namespace, connectome: ConnectomeFromAvgFilters
    ) -> None:
        edges_dir = connectome.edges

        edges = pd.DataFrame({
            k: byte_to_str(edges_dir[k][:]) for k in [*param_config.groupby, "sign"]
        })
        grouped_edges = edges.groupby(
            param_config.groupby, as_index=False, sort=False
        ).first()

        param_config.source_type = grouped_edges.source_type.values
        param_config.target_type = grouped_edges.target_type.values
        param_config.value = grouped_edges.sign.values

        self.indices = get_scatter_indices(edges, grouped_edges, param_config.groupby)
        self.parameter = forward_subclass(
            InitialDistribution, param_config, subclass_key="initial_dist"
        )
        self.keys = list(
            zip(
                param_config.source_type.tolist(),
                param_config.target_type.tolist(),
            )
        )
        self.symmetry_masks = symmetry_masks(param_config.get("symmetric", []), self.keys)


class SynapseCount(Parameter):
    """Initialize synapse counts for edge types."""

    @deepcopy_config
    def __init__(
        self, param_config: Namespace, connectome: ConnectomeFromAvgFilters
    ) -> None:
        mode = param_config.get("mode", "")
        if mode != "mean":
            raise NotImplementedError(
                f"SynapseCount does not implement {mode}. Implement "
                "a custom Parameter subclass."
            )

        edges_dir = connectome.edges

        edges = pd.DataFrame({
            k: byte_to_str(edges_dir[k][:]) for k in [*param_config.groupby, "n_syn"]
        })
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
        self.parameter = forward_subclass(
            InitialDistribution, param_config, subclass_key="initial_dist"
        )
        self.keys = list(
            zip(
                param_config.source_type.tolist(),
                param_config.target_type.tolist(),
                param_config.du.tolist(),
                param_config.dv.tolist(),
            )
        )
        self.symmetry_masks = symmetry_masks(param_config.get("symmetric", []), self.keys)


class SynapseCountScaling(Parameter):
    """Initialize synapse count scaling for edge types.

    This class initializes synapse strengths based on the average synapse count for each
    edge type, scaling them differently for chemical and electrical synapses.

    The initialization follows this equation:

    $$\\alpha_{t_it_j} =\\frac{\\rho}{\\langle N \\rangle_{t_it_j}}$$

    where:

    1. $\\alpha_{t_it_j}$ is the synapse strength between neurons $i$ and $j$.
    2. $\\langle N \\rangle_{t_it_j}$ is the average synapse count for the edge type
        across columnar offsets $u_i-u_j$ and $v_i-v_j$
    3. $\\rho$ is a scaling factor (default: 0.01)

    """

    @deepcopy_config
    def __init__(
        self, param_config: Namespace, connectome: ConnectomeFromAvgFilters
    ) -> None:
        edges_dir = connectome.edges

        edges = pd.DataFrame({
            k: byte_to_str(edges_dir[k][:]) for k in [*param_config.groupby, "n_syn"]
        })
        grouped_edges = edges.groupby(
            param_config.groupby, as_index=False, sort=False
        ).mean()

        # to initialize synapse strengths with scale/<N>_rf
        syn_strength = param_config.get("scale", 0.01) / grouped_edges.n_syn.values

        param_config.target_type = grouped_edges.target_type.values
        param_config.source_type = grouped_edges.source_type.values
        param_config.value = syn_strength

        self.indices = get_scatter_indices(edges, grouped_edges, param_config.groupby)
        self.parameter = forward_subclass(
            InitialDistribution, param_config, subclass_key="initial_dist"
        )
        self.keys = list(
            zip(
                param_config.source_type.tolist(),
                param_config.target_type.tolist(),
            )
        )
        self.symmetry_masks = symmetry_masks(param_config.get("symmetric", []), self.keys)


def get_scatter_indices(
    dataframe: pd.DataFrame, grouped_dataframe: pd.DataFrame, groupby: List[str]
) -> Tensor:
    """Get indices for scattering operations to share parameters.

    Maps each node/edge from the complete computational graph to a parameter index.

    Args:
        dataframe: Dataframe of nodes or edges of the graph.
        grouped_dataframe: Aggregated version of the same dataframe.
        groupby: The same columns from which the grouped_dataframe was constructed.

    Returns:
        Tensor of indices for scattering operations.

    Note:
        For N elements that are grouped into M groups, this function returns N indices
        from 0 to M-1 that can be used to scatter the parameters of the M groups to the
        N elements.

    Example:
        ```python
        elements = ["A", "A", "A", "B", "B", "C", "D", "D", "E"]
        groups = ["A", "B", "C", "D", "E"]
        parameter = [1, 2, 3, 4, 5]
        # get_scatter_indices would return
        scatter_indices = [0, 0, 0, 1, 1, 2, 3, 3, 4]
        scattered_parameters = [parameter[idx] for idx in scatter_indices]
        scattered_parameters == [1, 1, 1, 2, 2, 3, 4, 4, 5]
        ```
    """
    ungrouped_elements = zip(*[dataframe[k][:] for k in groupby])
    grouped_elements = zip(*[grouped_dataframe[k][:] for k in groupby])
    to_index = {k: i for i, k in enumerate(grouped_elements)}
    return torch.tensor([to_index[k] for k in ungrouped_elements])


def symmetry_masks(
    symmetric: List[Any], keys: List[Any], as_mask: bool = False
) -> List[torch.Tensor]:
    """Create masks for subsets of parameters for joint constraints.

    Args:
        symmetric: Contains subsets of keys that point to the subsets of parameters
            to be indexed.
        keys: List of keys that point to individual parameter values.
        as_mask: If True, returns a boolean mask, otherwise integer indices.

    Returns:
        List of masks (List[torch.BoolTensor]).

    Note:
        This is experimental for configuration-based fine-grained shared parameter
        optimization, e.g. for models including multi-compartment cells or gap
        junctions.

    Example:
        ```python
        # For node type parameters with individual node types as keys:
        symmetric = [["T4a", "T4b", "T4c", "T4d"], ["T5a", "T5b", "T5c", "T5d"]]
        # This would constrain the parameter values of all T4 subtypes to their joint
        # mean and the parameter values of all T5 subtypes to their joint mean.

        # For edge type parameters with individual edge types as keys:
        symmetric = [[("CT1(M10)", "CT1(Lo1)"), ("CT1(Lo1)", "CT1(M10)")]]
        # This would constrain the edge parameter of the directed edge from CT1(M10) to
        # CT1(Lo1) and the directed edge from CT1(Lo1) to CT1(M10) to their joint mean.
        ```
    """
    if not symmetric:
        return []
    symmetry_masks = []  # type: List[torch.Tensor]
    keys = atleast_column_vector(keys)
    for identifiers in symmetric:
        identifiers = atleast_column_vector(identifiers)
        # to allow identifiers like [None, "A", None, 0]
        # for parameters that have tuples as keys
        columns = np.arange(identifiers.shape[1] + 1)[
            np.where((identifiers is not None).all(axis=0))
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
                f"{identifiers} cannot be a symmetry constraint "
                f"for parameter with keys {keys}: {e}"
            ) from e
    return symmetry_masks


# --- Supplementary Error types -------------------------------------------------


class ParameterConfigError(ValueError):
    pass
