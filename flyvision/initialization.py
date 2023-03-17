"""
The parameters that the networks can be initialized with. Each parameter is a
type on its own, because different parameters are shared differently. These
types handle the initialization of indices to perform gather and scatter opera-
tions. Parameter types can be initialized from a range of initial distribution
types.

Note: to maintain compatibility with old configurations, e.g. to reinitialize
    a trained network, careful when refactoring any of these types or syntax.
"""

from typing import Any, Dict, Iterable, List
import logging
import numpy as np
import pandas as pd

import torch
from torch import Tensor
import torch.nn as nn

from datamate import Directory, Namespace

logging = logging.getLogger()

# -- Supplementary Error types


class ParameterConfigError(ValueError):
    pass


# -- Initial distribution types


class InitialDistribution:

    """Initial distribution base class.

    Types for initial parameters must store raw_values as attribute.
    """

    raw_values: Tensor
    readers: Dict[str, Tensor]

    def __new__(cls, param_config, *args, **kwargs):

        distributions = dict(
            value=Value,
            normal=Normal,
            trunc_normal=TruncatedNormal,
            lognormal=Lognormal,
            tanh=Tanh,
            sign=Sign,
            uniform=Uniform,
            uniform_from_mean_std=UniformFromMeanStd,
        )

        form = param_config.get("form", None)

        if form is None:
            raise ParameterConfigError(
                "Initial distribution not specified. Specify 'form'"
                " attribute in the Parameter configuration. 'form'"
                f" can be one of {list(distributions.keys())}"
            )

        try:
            _type = distributions[form]
        except KeyError as e:
            raise ParameterConfigError(
                f"form={form} is not a valid initial distribution."
                f" form can be one of {list(distributions.keys())}."
            )

        obj = object.__new__(_type)
        return obj

    @property
    def semantic_values(self):
        return self.raw_values

    def __repr__(self):
        return f"{self.__class__.__name__} (semantic values): \n{self.semantic_values}"

    def __len__(self):
        return len(self.raw_values)

    def clamp(self, values, param_config):
        """To clamp the parameter at initialization before training.

        TODO: clashes with raw_values/semantic_values distinction, because
        lognormal implements raw_values.exp() as semantic values and clamp
        would clamp the raw values. I think this could be more straightforward
        by either getting rid of semantic_values and making syn count normally dist.
        parameters that are getting clamped or by returning a temporary clamped
        version as semantic values based on the clamp configuration (complicated).
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
    """Samples uniformly between mean ± 3 * std.

    Args:
        param_config: requires value: array
                               requires_grad: bool
                               min: float optional
                               max: float optional
                               seed: int optional
    """

    def __init__(self, param_config) -> None:
        _values = torch.tensor(param_config.value[:])
        _values = self.clamp(_values, param_config)
        self.raw_values = nn.Parameter(
            _values, requires_grad=param_config.requires_grad
        )


class Uniform(InitialDistribution):
    """Samples uniformly between low and high.

    Args:
        param_config: requires low: float
                               high: float
                               count: int = number of parameters to sample
                               requires_grad: bool
                               min: float optional
                               max: float optional
                               seed: int optional
    """

    def __init__(self, param_config) -> None:
        seed = param_config.get("seed", None)
        if seed is not None:
            torch.manual_seed(seed)
        _values = torch.distributions.uniform.Uniform(
            param_config.low, param_config.high
        ).sample([param_config.count])
        _values = self.clamp(_values, param_config)
        self.raw_values = nn.Parameter(
            _values, requires_grad=param_config.requires_grad
        )


class UniformFromMeanStd(InitialDistribution):
    """Samples uniformly between mean ± 3 * std.

    Args:
        param_config: requires mean: float
                               std:  float
                               requires_grad: bool
                               min: float optional
                               max: float optional
                               seed: int optional
    """

    def __init__(self, param_config) -> None:
        seed = param_config.get("seed", None)
        if seed is not None:
            torch.manual_seed(seed)
        _values = torch.distributions.uniform.Uniform(
            torch.tensor(param_config.mean - 3 * param_config.std),
            torch.tensor(param_config.mean + 3 * param_config.std),
        ).sample()
        _values = self.clamp(_values, param_config)
        self.raw_values = nn.Parameter(
            _values, requires_grad=param_config.requires_grad
        )


class Normal(InitialDistribution):
    """Samples from independent normal distributions defined by mean and std.

    Args:
        param_config: requires mean: float
                               std:  float
                               mode: 'mean' or 'sample'
                               requires_grad: bool
                               min: float optional
                               max: float optional
                               seed: int optional
    """

    def __init__(self, param_config) -> None:
        if param_config.mode == "mean":
            _values = torch.Tensor(param_config.mean[:])
        elif param_config.mode == "sample":
            seed = param_config.get("seed", None)
            if seed is not None:
                torch.manual_seed(seed)
            _values = torch.distributions.normal.Normal(
                torch.Tensor(param_config.mean), torch.Tensor(param_config.std)
            ).sample()
        else:
            raise ValueError("Mode must be either mean or sample.")
        _values = self.clamp(_values, param_config)
        self.raw_values = nn.Parameter(
            _values, requires_grad=param_config.requires_grad
        )


class TruncatedNormal(InitialDistribution):
    """Truncated normal distributions defined by mean, std and boundaries.

    Args:
        param_config: requires mean: array
                               std:  array
                               clamp: Tuple[Optional[float], Optional[float]]
                               mode: 'mean' or 'sample'
                               requires_grad: bool
                               min: float optional
                               max: float optional
                               seed: int optional
    """

    def __init__(self, param_config) -> None:

        if not param_config.get("clamp", False):
            raise ValueError("truncated normal distribution requires bounds")

        # to handle False and None as infinite bounds
        # clamp can be tuple, so to mutate make list
        param_config.clamp = list(param_config.clamp)
        if param_config.clamp[0] is None:
            param_config.clamp[0] = -np.inf

        if param_config.clamp[1] is None:
            param_config.clamp[1] = np.inf

        if param_config.mode == "mean":
            _values = torch.Tensor(param_config.mean[:])
        elif param_config.mode == "sample":
            seed = param_config.get("seed", None)
            if seed is not None:
                torch.manual_seed(seed)
            _values = torch.distributions.normal.Normal(
                torch.Tensor(param_config.mean), torch.Tensor(param_config.std)
            ).sample()
            mask = (_values < param_config.clamp[0]) | (_values > param_config.clamp[1])
            # rejection sampling
            while mask.any():
                _values[mask] = torch.distributions.normal.Normal(
                    torch.Tensor(param_config.mean),
                    torch.Tensor(param_config.std),
                ).sample()[mask]
                mask = (_values < param_config.clamp[0]) | (
                    _values > param_config.clamp[1]
                )
        else:
            raise ValueError("Mode must be either mean or sample.")
        self.raw_values = nn.Parameter(
            _values, requires_grad=param_config.requires_grad
        )


class Lognormal(InitialDistribution):
    """Samples from independent lognormal distributions defined by mean and std.

    Args:
        param_config: requires mean: float
                               std:  float
                               mode: 'mean' or 'sample'
                               requires_grad: bool
                               min: float optional
                               max: float optional
                               seed: int optional
    """

    def __init__(self, param_config) -> None:

        if param_config.get("clamp", False):
            logging.warning(
                f"clamping has no effect for {self.__class__.__name__} parameters"
                " because clamping acts on raw_values"
                " but the lognormal parameter semantic values are raw_values.exp()"
            )

        if param_config.mode == "mean":
            _values = torch.Tensor(param_config.mean[:])
        elif param_config.mode == "sample":
            # The log is normally distributed and in the class SynCount we take the log, thus the normal distr. here.
            seed = param_config.get("seed", None)
            if seed is not None:
                torch.manual_seed(seed)
            _values = torch.distributions.normal.Normal(
                torch.Tensor(param_config.mean), torch.Tensor(param_config.std)
            ).sample()
        else:
            raise ValueError("Mode must be either mean or sample.")
        self.raw_values = nn.Parameter(
            _values, requires_grad=param_config.requires_grad
        )

    @property
    def semantic_values(self):
        """n_syn ~ self._values.exp()."""
        return self.raw_values.exp()


class Tanh(InitialDistribution):
    """Tanh for sign training."""

    def __init__(self, param_config) -> None:
        if param_config.mode == "mean":
            return NotImplementedError
            _values = torch.Tensor(param_config.mean[:])
        elif param_config.mode == "sample":
            seed = param_config.get("seed", None)
            if seed is not None:
                torch.manual_seed(seed)
            _values = torch.distributions.uniform.Uniform(-1, 1).sample(
                param_config.source_type.shape
            )
        else:
            raise ValueError("Mode must be either mean or sample.")
        _values = self.clamp(_values, param_config)
        self.raw_values = nn.Parameter(
            _values, requires_grad=param_config.requires_grad
        )
        self.arg_gain = getattr(param_config, "arg_gain", 1)

    @property
    def semantic_values(self):
        """n_syn ~ self._values.exp()."""
        return torch.tanh(self.arg_gain * self.raw_values)


class Sign(InitialDistribution):
    """Tanh for sign training."""

    def __init__(self, param_config) -> None:
        if param_config.mode == "mean":
            return NotImplementedError
            _values = torch.Tensor(param_config.mean[:])
        elif param_config.mode == "sample":
            seed = param_config.get("seed", None)
            if seed is not None:
                torch.manual_seed(seed)
            _values = torch.distributions.uniform.Uniform(-1, 1).sample(
                param_config.source_type.shape
            )
        else:
            raise ValueError("Mode must be either mean or sample.")
        _values = self.clamp(_values, param_config)
        self.raw_values = nn.Parameter(
            _values, requires_grad=param_config.requires_grad
        )
        self.arg_gain = getattr(param_config, "arg_gain", 1)

    @property
    def semantic_values(self):
        """n_syn ~ self._values.exp()."""
        return torch.sign(self.arg_gain * self.raw_values)


# - Parameter types


class Parameter:
    parameter: InitialDistribution
    indices: torch.Tensor
    symmetry_masks: List[torch.Tensor]
    keys: List[Any]

    def __new__(cls, param_config: Namespace, wrap: Directory):

        obj = super(Parameter, cls).__new__(cls)
        # first create a copy of the mutable param_config to not alter the
        # config object externally
        param_config = param_config.deepcopy()
        # now make this config accesible as _config for debug the mutations
        object.__setattr__(obj, "_config", param_config)
        # make a copy accesible that is not supposed to be altered
        object.__setattr__(obj, "config", param_config.deepcopy())
        return obj

    def __repr__(self):
        init_arg_names = list(self.__init__.__annotations__.keys())
        wrap_type = self.__init__.__annotations__[init_arg_names[1]].__name__
        return f"{self.__class__.__name__}({self.config}, {wrap_type})"

    def __getitem__(self, key):
        if key in self.keys:
            return self.parameter.raw_values[self.keys.index(key)]
        else:
            raise ValueError(key)

    # -- InitialDistribution API

    @property
    def raw_values(self) -> torch.Tensor:
        return self.parameter.raw_values

    # @raw_values.setter
    # def raw_values(self, values: torch.Tensor) -> torch.Tensor:

    #     if type(values) != type(self.parameter.raw_values):
    #         raise ValueError("invalid type")

    #     if values.shape != self.parameter.raw_values.shape:
    #         raise ValueError("invalid shape")

    #     self.parameter.raw_values = values

    @property
    def semantic_values(self) -> torch.Tensor:
        return self.parameter.semantic_values

    @property
    def readers(self) -> Dict[str, torch.Tensor]:
        return self.parameter.readers

    @readers.setter
    def readers(self, value) -> None:
        self.parameter.readers = value


class Nodeswrap(Directory):
    """Part of the Connectome describing the nodes."""

    pass


class Edgeswrap(Directory):
    """Part of the Connectome describing the edges."""

    pass


class NodesOrEdgeswrap(Directory):
    pass


# -- Node / Cell type parameter

# ---_ Node / Cell bias parameter


class RestingPotential(Parameter):
    """Initialize resting potentials a.k.a. biases for cell types."""

    def __init__(self, param_config: Namespace, nodes_wrap: Nodeswrap):
        # equals order in connectome.unique_cell_types
        nodes = pd.DataFrame(
            dict(type=nodes_wrap.type[:].astype(str))
        ).drop_duplicates()

        param_config["type"] = nodes["type"].values.astype("S")
        param_config["mean"] = np.repeat(np.float32(param_config["mean"]), len(nodes))
        param_config["std"] = np.repeat(np.float32(param_config["std"]), len(nodes))

        self.symmetry_masks = symmetry_mask_for_nodes(
            param_config.get("symmetric", []), nodes
        )

        self.indices = gather_indices(param_config, nodes_wrap)
        self.parameter = InitialDistribution(param_config)
        self.keys = param_config["type"].astype(str).tolist()


# ---- Node / Cell time constant parameter


class TimeConstant(Parameter):
    def __init__(self, param_config: Namespace, nodes_wrap: Nodeswrap):
        nodes = pd.DataFrame(
            dict(type=nodes_wrap.type[:].astype(str))
        ).drop_duplicates()

        param_config["type"] = nodes["type"].values.astype("S")
        param_config["value"] = np.repeat(np.float32(param_config["value"]), len(nodes))

        self.symmetry_masks = symmetry_mask_for_nodes(
            param_config.get("symmetric", []), nodes
        )

        self.indices = gather_indices(param_config, nodes_wrap)
        self.parameter = InitialDistribution(param_config)
        self.keys = param_config["type"].astype(str).tolist()


# --- Edge / 'synapse' parameter


# ---- Edge / 'synapse' sign parameter


class SynapseSign(Parameter):
    """A PDF that generates edge signs from edges attributes"""

    def __init__(self, param_config: Namespace, edges_wrap: Edgeswrap) -> None:
        edges = pd.DataFrame(
            dict(
                source_type=edges_wrap.source_type,
                target_type=edges_wrap.target_type,
                sign=edges_wrap.sign,
            )
        )
        edges = edges.groupby(
            ["source_type", "target_type"], sort=False, as_index=False
        ).mean()
        assert all(
            np.abs(edges.sign.values) == np.ones(len(edges))
        ), "Inconsistent edge signs."

        param_config.source_type = edges.source_type.values.astype("S")
        param_config.target_type = edges.target_type.values.astype("S")
        param_config.value = edges.sign.values.astype("f")
        self.indices = gather_indices(param_config, edges_wrap)
        self.parameter = InitialDistribution(param_config)
        self.keys = list(
            zip(
                param_config.source_type.astype(str).tolist(),
                param_config.target_type.astype(str).tolist(),
            )
        )


# ---- Edge / 'synapse' synapse count parameter


class SynapseCount(Parameter):
    """A PDF that generates synapse counts from edges attributes

    param_config contains, e.g.:
        std: float
        "The standard deviation, in log space"
        form: str = "lognormal"
        requires_grad: bool = True
        mode: str = "mean"
        symmetric: List[List[str]] = [[b"CT1", b"CT1L"], [b"CT1L", b"CT1G"]]
    """

    def __init__(self, param_config: Namespace, edges_wrap: Edgeswrap) -> None:

        mode = param_config.get("mode", "")
        if mode != "mean":
            raise NotImplementedError(
                f"SynapseCount does not implement {mode}. Implement "
                "a custom Parameter subclass."
            )

        edges = pd.DataFrame(
            dict(
                source_type=edges_wrap.source_type,
                target_type=edges_wrap.target_type,
                du=edges_wrap.du,
                dv=edges_wrap.dv,
                n_syn=edges_wrap.n_syn,
            )
        )
        offset_keys = ["source_type", "target_type", "du", "dv"]
        edges = edges.groupby(offset_keys, sort=False, as_index=False).mean()

        param_config.source_type = edges.source_type.values.astype("S")
        param_config.target_type = edges.target_type.values.astype("S")
        param_config.du = edges.du.values
        param_config.dv = edges.dv.values

        param_config.mode = "mean"
        param_config.mean = np.log(edges.n_syn.values.astype("f"))
        # param_config.std = param_config.std * np.ones(len(edges))

        self.symmetry_masks = symmetry_mask_for_edges(
            param_config.get("symmetric", []), edges
        )
        self.indices = gather_indices(param_config, edges_wrap)
        self.parameter = InitialDistribution(param_config)
        self.keys = list(
            zip(
                param_config.source_type.astype(str).tolist(),
                param_config.target_type.astype(str).tolist(),
                param_config.du.tolist(),
                param_config.dv.tolist(),
            )
        )


# ---- Edge / 'synapse' synapse scaling parameter


class SynapseCountScaling(Parameter):
    """Synapse count scaling factor.

    One per receptive field for each target cell type.
    Initial values scaled by scale * 1 / mean(n_syn) over receptive field.
    """

    def __init__(self, param_config: Namespace, edges_wrap: Edgeswrap) -> None:
        edges = pd.DataFrame(
            dict(
                source_type=edges_wrap.source_type,
                target_type=edges_wrap.target_type,
                edge_type=edges_wrap.edge_type,
                n_syn=edges_wrap.n_syn,
            )
        )
        # Same if sorting by src-tar or tar-src!
        edges = edges.groupby(
            ["source_type", "target_type", "edge_type"],
            sort=False,
            as_index=False,
        ).mean()
        syn_strength = 1 / edges.n_syn.values  # 1/<N>_rf
        syn_strength[edges[edges.edge_type == b"chem"].index] *= param_config.scale_chem
        syn_strength[edges[edges.edge_type == b"elec"].index] *= param_config.scale_elec

        param_config.target_type = edges.target_type.values.astype("S")
        param_config.source_type = edges.source_type.values.astype("S")
        param_config.value = syn_strength.astype("f")

        self.symmetry_masks = symmetry_mask_for_edges(
            param_config.get("symmetric", []), edges
        )

        self.indices = gather_indices(param_config, edges_wrap)
        self.parameter = InitialDistribution(param_config)
        self.keys = list(
            zip(
                param_config.source_type.astype(str).tolist(),
                param_config.target_type.astype(str).tolist(),
            )
        )


def init_parameter(
    param_config: Namespace, nodes_or_edges_wrap: NodesOrEdgeswrap
) -> Parameter:
    param_config = param_config.deepcopy()
    _type = param_config.type
    param_type = globals()[_type]  # type: Parameter
    param = param_type(param_config, nodes_or_edges_wrap)
    return param


def gather_indices(param_config: Namespace, nodes_or_edges_wrap: Edgeswrap) -> Tensor:
    """
    One way to share parameters is through scatter and gather operation.

    Return a length-'n_elems' vector of indices such that 'result[i]' is the
    index of the row in 'param_config' corresponding to the 'i'th row of 'elems'.
    Indices for gathering operations.

    Note: 'param_config' must have all the attributes that exist as h5 files in
        'node_or_edges_wrap' in order to group by them.

    Note: this is equivalent to
        np.concatenate([i * np.ones_like(index) for i, (key, index)
                        in enumerate(gb.groups.items())])
            where gb is a grouped dataframe object.

    TODO: for speed up and straight-forwardness consider to change to the above.
    """
    # to get all mutual keys between the connectome part and the param config
    mutual_keys = set(nodes_or_edges_wrap).intersection(set(param_config))
    # concatenation of all connectome elements to keys
    ctome_elements = zip(*[nodes_or_edges_wrap[k][:] for k in mutual_keys])
    # concatenation of all param elements to keys
    param_elements = zip(*[param_config[k][:] for k in mutual_keys])
    # create a mapping from ctome_elements to param_elements
    # (e.g. from all edges to shared edge parameters)
    # to create indices mapping each node and edge to the respective parameters
    mapping = {k: i for i, k in enumerate(param_elements)}
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


def symmetry_mask_for_edges(symmetric: List[List[str]], edges: pd.DataFrame):
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
