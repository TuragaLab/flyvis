"""Classes defining the voltage initialization, voltage and current dynamics."""

import torch
from torch import nn

__all__ = ["NetworkDynamics", "StaticSynapses"]

activation_fns = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "softplus": nn.Softplus,
    "leakyrelu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
}

# -- The `NetworkDynamics` interface -------------------------------------------


class NetworkDynamics:
    """
    Defines the initialization and behavior of a network during simulation

    There are two node state variable names with special semantics:

        - "activity" is the state variable written to by network inputs, and
            read by decoders trying to perform tasks based on the network's
            state.
        - "energy_use" is read by regularizers during network training.
    """

    class Config:
        activation: str = "relu"

    def __new__(cls, config):
        return _forward_subclass(cls, config)

    def __init__(self, config: Config):
        config = config.deepcopy()
        self.activation = activation_fns[config.activation.pop("type")](
            **config.activation
        )

    def write_derived_params(self, params) -> None:
        """
        Augment `params`, called once at every forward passs.

        Parameters:
            params: A namespace containing two subnamespaces: `nodes` and
                `edges`, containing node and edges parameters, respectively.

        This is called at the beginning of a stimulus presentation, before
        `write_initial_state`. Parameter transformations can be moved here to
        avoid repeating them at every timestep.
        """
        pass

    def write_initial_state(self, state, params) -> None:
        """
        Initialize a network's state variables from its network parameters.

        Parameters:
            state: A namespace containing two subnamspaces: `nodes` and
                `edges`. Write initial node and edge state variable values, as
                1D tensors, into them, respectively.
            params: A namespace containing four subnamespaces: `nodes`,
                `edges`, `sources`, and `targets`. `nodes` and `edges` contain
                node and edges parameters, respectively. `sources` and
                `targets` provide access to the node parameters associated with
                the source node and target node of each edge, respectively.

        Note: called by Network._initial_state.
        """
        pass

    def write_state_velocity(self, vel, state, params, target_sum) -> None:
        """
        Compute δvar/δtime for each state variable.

        Parameters:
            vel: A namespace containing two subnamspaces: `nodes` and
                `edges`. Write δvar/δtime for node and edge state variables
                into them, respectively.
            state: A namespace containing two subnamspaces: `nodes` and
                `edges`, containing node and edge state variable values,
                respectively.
            params: A namespace containing four subnamespaces: `nodes`,
                `edges`, `sources`, and `targets`. `nodes` and `edges` contain
                node and edges parameters, respectively. `sources` and
                `targets` provide access to the node parameters associated with
                the source node and target node of each edge, respectively.
            target_sum: Sum the entries in a `len(edges)` tensor corresponding to edges
                with the same target node, yielding a `len(nodes)` tensor.

        Note: called by Network._next_state.
        """
        pass


# -- `NetworkDynamics` implementations -----------------------------------------


class PPNeuronIGRSynapses(NetworkDynamics):
    def write_derived_params(self, params, chem_indices, elec_indices, **kwargs):
        params.edges.weight = (
            params.edges.sign * params.edges.syn_count * params.edges.syn_strength
        )

        # CHEMICAL SYNAPSES (DEFAULT)
        weight_chem = torch.zeros_like(params.edges.weight)
        weight_chem[chem_indices] = params.edges.weight.index_select(-1, chem_indices)
        params.edges.weight_chem = weight_chem

        if elec_indices.any():
            # ELECTRICAL SYNAPSES (CT1)
            weight_elec = torch.zeros_like(params.edges.weight)
            weight_elec[elec_indices] = params.edges.weight.index_select(
                -1, elec_indices
            )
            params.edges.weight_elec = weight_elec  # external stimulus

    def write_initial_state(self, state, params, source_indices, mean, std):
        if mean is not None and std is not None:
            state.nodes.activity = (
                torch.distributions.Normal(mean, std)
                .sample(params.nodes.bias.size())
                .clamp_(0.0)
            )
        else:
            state.nodes.activity = params.nodes.bias

    def write_state_velocity(self, vel, state, params, target_sum, x_t, dt, **kwargs):
        vel.nodes.activity = (
            1
            / torch.max(params.nodes.time_const, torch.tensor(dt).float())
            * (
                -state.nodes.activity
                + params.nodes.bias
                + target_sum(
                    params.edges.weight_chem * self.activation(state.sources.activity)
                )  # internal chemical current
                + (
                    target_sum(
                        (state.sources.activity - state.targets.activity)
                        * params.edges.weight_elec
                    )
                    if "weight_elec" in params.edges
                    else 0
                )
                + x_t
            )
        )

    def currents(self, state, params, electric=False, both=False):
        if electric:
            return (
                state.sources.activity - state.targets.activity
            ) * params.edges.weight_elec
        elif both:
            return (
                params.edges.weight_chem * self.activation(state.sources.activity)
                + (state.sources.activity - state.targets.activity)
                * params.edges.weight_elec
            )
        return params.edges.weight_chem * self.activation(state.sources.activity)


def _forward_subclass(cls: type, config: object = {}) -> object:
    target_subclass = config.pop("type", None)
    for subclass in cls.__subclasses__():
        if target_subclass == subclass.__qualname__:
            return object.__new__(subclass)
    return object.__new__(cls)
